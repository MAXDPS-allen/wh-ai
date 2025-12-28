"""
约束条件驱动的材料生成主程序
=============================================
基于用户定义的约束条件生成和筛选铁电材料

功能:
1. 加载用户约束配置文件
2. 搜索已知铁电材料数据库中满足约束的材料
3. 使用CVAE+逆向设计生成新候选材料
4. 根据约束条件筛选候选材料
5. 与Materials Project数据库对比
6. 生成详细报告

使用方法:
    python main.py --constraint config/example_titanate.json
    python main.py --constraint config/example_leadfree.json --iterations 10
"""

import sys
import os
import argparse
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch

# 添加上级目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'code_v5'))

from constraint_parser import load_constraints, ConstraintSet
from constrained_filter import (
    ConstraintFilter, 
    MaterialCandidate, 
    create_material_from_dict,
    FilterResult
)


class ConstrainedMaterialGenerator:
    """约束驱动的材料生成器"""
    
    def __init__(self, constraints: ConstraintSet):
        """
        初始化生成器
        
        Args:
            constraints: 约束集合
        """
        self.constraints = constraints
        self.filter = ConstraintFilter(constraints)
        
        # 设置输出目录
        self.output_dir = Path(constraints.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        if constraints.generation.seed is not None:
            np.random.seed(constraints.generation.seed)
            torch.manual_seed(constraints.generation.seed)
        
        # 模型（延迟加载）
        self.cvae = None
        self.inverse_design = None
        
        # 结果统计
        self.stats = {
            'timestamp': datetime.now().isoformat(),
            'constraint_name': constraints.name,
            'known_materials': 0,
            'generated_total': 0,
            'generated_passed': 0,
            'mp_matched': 0,
            'iterations': []
        }
    
    def load_models(self):
        """加载CVAE和逆向设计模型"""
        print("\n加载模型...")
        
        # 导入模型相关模块
        try:
            from FE_CVAE import CVAE
        except ImportError as e:
            print(f"错误: 无法导入CVAE模块 - {e}")
            print("请确保 FE_CVAE.py 在 code_v5/ 目录中")
            raise
        
        # 模型路径
        base_dir = Path(__file__).parent.parent
        cvae_path = base_dir / 'model_cvae' / 'cvae_best.pt'
        inv_design_path = base_dir / 'invs_dgn_model_v2' / 'inverse_design_v6_best.pt'
        
        # 备用路径
        if not cvae_path.exists():
            cvae_path = base_dir / 'model_v3' / 'cvae_model_v3.pth'
        if not inv_design_path.exists():
            inv_design_path = base_dir / 'invs_dgn_model' / 'inverse_design_network_v7.pth'
        
        # 加载CVAE
        if not cvae_path.exists():
            raise FileNotFoundError(f"CVAE模型不存在: {cvae_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  使用设备: {device}")
        
        print(f"  加载 CVAE: {cvae_path.name}")
        self.cvae = CVAE(
            input_dim=64,
            hidden_dim=512,
            latent_dim=32
        ).to(device)
        
        # 加载 checkpoint（可能是字典格式）
        checkpoint = torch.load(cvae_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.cvae.load_state_dict(checkpoint['model'])
        else:
            self.cvae.load_state_dict(checkpoint)
        self.cvae.eval()
        
        # 加载逆向设计网络
        if not inv_design_path.exists():
            raise FileNotFoundError(f"逆向设计模型不存在: {inv_design_path}")
        
        print(f"  加载 Inverse Design Network: {inv_design_path.name}")
        
        # 导入逆向设计网络
        try:
            sys.path.insert(0, str(base_dir / 'gen'))
            # 使用 v6 版本的模型 (与保存的权重匹配)
            from inverse_design_v6 import InverseDesignModel, Config
            
            config = Config()
            self.inverse_design = InverseDesignModel(config).to(device)
        except ImportError as e:
            print(f"  警告: 无法导入 InverseDesignModel ({e})，使用简化版本")
            # 使用简化的网络定义
            from torch import nn
            
            class InverseDesignNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 简化结构
                    self.network = nn.Sequential(
                        nn.Linear(64, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    )
                    
                    # 输出头
                    self.element_head = nn.Linear(512, 5 * 87)  # 5个位置，87个元素
                    self.lattice_head = nn.Linear(512, 6)  # vol_root, sqrt(b/a), sqrt(c/a), 3个角度
                    self.fraction_head = nn.Linear(512, 5)  # 5个位置的分数
                
                def forward(self, x):
                    features = self.network(x)
                    element_logits = self.element_head(features).view(-1, 5, 87)
                    lattice_params = self.lattice_head(features)
                    fractions = torch.softmax(self.fraction_head(features), dim=-1)
                    return {
                        'element_logits': element_logits,
                        'lattice': lattice_params,
                        'fractions': fractions
                    }
            
            self.inverse_design = InverseDesignNetwork().to(device)
        
        try:
            # 加载 checkpoint（可能是字典格式）
            checkpoint = torch.load(inv_design_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.inverse_design.load_state_dict(checkpoint['model'])
            else:
                self.inverse_design.load_state_dict(checkpoint)
        except Exception as e:
            print(f"  警告: 无法完整加载模型权重 ({e})，尝试部分加载")
            # 尝试部分加载
            checkpoint = torch.load(inv_design_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model_dict = self.inverse_design.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.inverse_design.load_state_dict(model_dict, strict=False)
            print(f"  已加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
        
        self.inverse_design.eval()
        self.device = device
        
        print("  ✓ 模型加载完成")
    
    def search_known_materials(self) -> List[MaterialCandidate]:
        """
        在已知铁电材料数据库中搜索满足约束的材料
        
        Returns:
            满足约束的已知材料列表
        """
        print("\n" + "="*70)
        print("搜索已知铁电材料数据库")
        print("="*70)
        
        # 数据库路径
        base_dir = Path(__file__).parent.parent
        db_paths = [
            base_dir / 'new_data' / 'dataset_original_ferroelectric.jsonl',
            base_dir / 'new_data' / 'dataset_known_FE_rest.jsonl',
            base_dir / 'ferroelectric_database_labeled.csv'
        ]
        
        all_materials = []
        
        # 读取JSONL文件
        for db_path in db_paths[:2]:
            if db_path.exists():
                print(f"\n读取: {db_path.name}")
                try:
                    with open(db_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                data = json.loads(line)
                                material = create_material_from_dict(data, material_id=len(all_materials))
                                all_materials.append(material)
                            except json.JSONDecodeError:
                                continue
                    print(f"  加载 {len(all_materials)} 条数据")
                except Exception as e:
                    print(f"  错误: {e}")
        
        # 读取CSV文件
        csv_path = db_paths[2]
        if csv_path.exists():
            print(f"\n读取: {csv_path.name}")
            try:
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    data = row.to_dict()
                    material = create_material_from_dict(data, material_id=len(all_materials))
                    all_materials.append(material)
                print(f"  加载 {len(df)} 条数据")
            except Exception as e:
                print(f"  错误: {e}")
        
        print(f"\n总计加载 {len(all_materials)} 条已知材料")
        
        # 应用约束筛选
        print("\n应用约束条件筛选...")
        passed_materials, results = self.filter.filter_batch(all_materials)
        
        print(f"  筛选后: {len(passed_materials)} 条材料满足约束")
        
        # 更新统计
        self.stats['known_materials'] = len(passed_materials)
        
        # 保存结果
        if passed_materials:
            output_file = self.output_dir / 'known_materials_matched.csv'
            self._save_materials_to_csv(passed_materials, output_file)
            print(f"  已保存到: {output_file}")
        
        return passed_materials
    
    def generate_iteration(self, iteration: int) -> Tuple[List[MaterialCandidate], List[FilterResult]]:
        """
        执行一次生成迭代
        
        Args:
            iteration: 迭代编号
            
        Returns:
            (生成的材料列表, 过滤结果列表)
        """
        # 导入特征工程模块
        sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
        from feature_engineering import ELEMENT_DATABASE
        
        print(f"\n迭代 {iteration}/{self.constraints.generation.n_iterations}")
        print("-" * 70)
        
        n_samples = self.constraints.generation.n_candidates
        temperature = self.constraints.generation.temperature
        
        # 生成特征向量
        print(f"  生成 {n_samples} 个候选特征向量...")
        with torch.no_grad():
            # 标签: 1 = 铁电 (使用整数张量，因为Embedding层需要整数索引)
            labels = torch.ones(n_samples, dtype=torch.long, device=self.device)
            
            # 使用CVAE采样
            sampled_features = self.cvae.sample(n_samples, labels, self.device)
            
            # 使用逆向设计网络预测材料描述
            outputs = self.inverse_design(sampled_features)
            
            # 处理返回格式（可能是字典或元组）
            if isinstance(outputs, dict):
                element_logits = outputs.get('element_logits', outputs.get('elements'))
                lattice_params = outputs.get('lattice')
                fractions = outputs.get('fractions')
            else:
                element_logits, lattice_params, fractions = outputs
        
        # 解码材料
        print("  解码材料描述...")
        materials = []
        element_list = sorted(ELEMENT_DATABASE.keys())
        
        for i in range(n_samples):
            try:
                # 解码元素
                elements = []
                composition = {}
                
                for pos in range(5):  # 5个元素位置
                    el_idx = torch.argmax(element_logits[i, pos]).item()
                    frac = fractions[i, pos].item()
                    
                    if frac > 0.05:  # 分数阈值
                        element = element_list[el_idx]
                        elements.append(element)
                        composition[element] = frac
                
                # 归一化组成
                total_frac = sum(composition.values())
                if total_frac > 0:
                    composition = {k: v/total_frac for k, v in composition.items()}
                
                # 生成化学式
                formula = ''.join([f"{el}{composition[el]:.2f}" for el in sorted(elements)])
                
                # 解码晶格参数 - 根据模型版本调整
                lat = lattice_params[i].cpu().numpy()
                
                # 尝试不同的晶格参数格式
                if len(lat) >= 6:
                    # 格式: [vol_root, sqrt_b_a, sqrt_c_a, alpha, beta, gamma]
                    vol_root = max(abs(lat[0]), 1.0)
                    volume = vol_root ** 3
                    
                    b_a_ratio = max(abs(lat[1]) ** 2, 0.5) if lat[1] >= 0 else max(lat[1] ** 2, 0.5)
                    c_a_ratio = max(abs(lat[2]) ** 2, 0.5) if lat[2] >= 0 else max(lat[2] ** 2, 0.5)
                    
                    a = (volume / (b_a_ratio * c_a_ratio)) ** (1/3)
                    b = a * b_a_ratio
                    c = a * c_a_ratio
                    
                    alpha = np.clip(lat[3] * 180, 60, 120) if abs(lat[3]) < 2 else np.clip(lat[3], 60, 120)
                    beta = np.clip(lat[4] * 180, 60, 120) if abs(lat[4]) < 2 else np.clip(lat[4], 60, 120)
                    gamma = np.clip(lat[5] * 180, 60, 120) if abs(lat[5]) < 2 else np.clip(lat[5], 60, 120)
                else:
                    # 简化格式
                    volume = max(abs(lat[0]) ** 3, 30)
                    a = b = c = volume ** (1/3)
                    alpha = beta = gamma = 90.0
                
                # 创建材料候选
                material = MaterialCandidate(
                    id=len(materials),
                    formula=formula,
                    elements=elements,
                    composition=composition,
                    volume=volume,
                    a=a, b=b, c=c,
                    alpha=alpha, beta=beta, gamma=gamma
                )
                
                materials.append(material)
                
            except Exception as e:
                # 跳过解码失败的样本
                continue
        
        print(f"  成功解码 {len(materials)} 个材料")
        
        # 应用约束筛选
        print("  应用约束筛选...")
        passed_materials, results = self.filter.filter_batch(materials)
        
        print(f"  筛选后: {len(passed_materials)} 个材料通过")
        
        # 统计
        stats = self.filter.get_statistics(results)
        
        iter_stats = {
            'iteration': iteration,
            'generated': len(materials),
            'passed': len(passed_materials),
            'pass_rate': stats['pass_rate'],
            'failure_reasons': stats['failure_reasons']
        }
        
        self.stats['iterations'].append(iter_stats)
        self.stats['generated_total'] += len(materials)
        self.stats['generated_passed'] += len(passed_materials)
        
        return passed_materials, results
    
    def compare_with_mp(self, materials: List[MaterialCandidate]) -> List[MaterialCandidate]:
        """
        与Materials Project数据库对比
        
        Args:
            materials: 材料列表
            
        Returns:
            匹配到MP的材料列表
        """
        print("\n" + "="*70)
        print("与 Materials Project 数据库对比")
        print("="*70)
        
        try:
            from mp_comparison import MPComparator, MPConfig
        except ImportError:
            print("  警告: 无法导入 mp_comparison 模块，跳过MP对比")
            return []
        
        try:
            comparator = MPComparator(MPConfig())
            
            # 转换为DataFrame格式
            data = []
            for mat in materials:
                data.append({
                    'id': mat.id,
                    'formula': mat.formula,
                    'elements': ','.join(mat.elements),
                    'volume': mat.volume,
                    'a': mat.a,
                    'b': mat.b,
                    'c': mat.c,
                    'alpha': mat.alpha,
                    'beta': mat.beta,
                    'gamma': mat.gamma
                })
            
            df = pd.DataFrame(data)
            
            # 执行对比
            print(f"\n检索 {len(materials)} 个材料...")
            matched_df = comparator.compare_dataframe(df)
            
            # 更新材料信息
            matched_materials = []
            if not matched_df.empty:
                for idx, row in matched_df.iterrows():
                    mat_id = row.get('generated_id', row.get('id', 0))
                    
                    # 找到原材料
                    original_mat = next((m for m in materials if m.id == mat_id), None)
                    if original_mat:
                        # 更新MP信息
                        original_mat.mp_id = row.get('mp_id')
                        original_mat.mp_formula = row.get('mp_formula')
                        original_mat.band_gap = row.get('mp_band_gap')
                        original_mat.energy_above_hull = row.get('mp_energy_above_hull')
                        original_mat.is_stable = row.get('mp_is_stable')
                        original_mat.is_polar = row.get('mp_is_polar')
                        original_mat.spacegroup = row.get('mp_spacegroup')
                        original_mat.point_group = row.get('mp_point_group')
                        
                        matched_materials.append(original_mat)
            
            print(f"  匹配到 {len(matched_materials)} 个MP材料")
            self.stats['mp_matched'] = len(matched_materials)
            
            return matched_materials
            
        except Exception as e:
            print(f"  错误: {e}")
            return []
    
    def generate_report(self, 
                       known_materials: List[MaterialCandidate],
                       generated_materials: List[MaterialCandidate],
                       mp_matched: List[MaterialCandidate]):
        """
        生成详细报告
        
        Args:
            known_materials: 已知材料
            generated_materials: 生成的材料
            mp_matched: 匹配到MP的材料
        """
        print("\n" + "="*70)
        print("生成报告")
        print("="*70)
        
        # 生成文本报告
        report_path = self.output_dir / 'report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("约束条件驱动的铁电材料生成报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"生成时间: {self.stats['timestamp']}\n")
            f.write(f"约束名称: {self.constraints.name}\n")
            f.write(f"约束描述: {self.constraints.description}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("约束条件摘要:\n")
            f.write("-"*70 + "\n")
            f.write(self.constraints.summary)
            f.write("\n\n")
            
            f.write("-"*70 + "\n")
            f.write("结果统计:\n")
            f.write("-"*70 + "\n")
            f.write(f"已知材料数据库匹配: {self.stats['known_materials']} 条\n")
            f.write(f"生成候选材料总数: {self.stats['generated_total']} 个\n")
            f.write(f"通过约束筛选: {self.stats['generated_passed']} 个\n")
            f.write(f"匹配到MP数据库: {self.stats['mp_matched']} 个\n")
            
            if self.stats['generated_total'] > 0:
                pass_rate = self.stats['generated_passed'] / self.stats['generated_total'] * 100
                f.write(f"总体通过率: {pass_rate:.1f}%\n")
            
            f.write("\n")
            
            # 迭代详情
            if self.stats['iterations']:
                f.write("-"*70 + "\n")
                f.write("迭代详情:\n")
                f.write("-"*70 + "\n")
                for iter_stat in self.stats['iterations']:
                    f.write(f"\n迭代 {iter_stat['iteration']}:\n")
                    f.write(f"  生成: {iter_stat['generated']} 个\n")
                    f.write(f"  通过: {iter_stat['passed']} 个\n")
                    f.write(f"  通过率: {iter_stat['pass_rate']*100:.1f}%\n")
                    
                    if iter_stat['failure_reasons']:
                        f.write(f"  失败原因:\n")
                        for reason, count in iter_stat['failure_reasons'].items():
                            f.write(f"    - {reason}: {count}\n")
        
        print(f"  报告已保存: {report_path}")
        
        # 保存统计JSON
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"  统计已保存: {stats_path}")
    
    def _save_materials_to_csv(self, materials: List[MaterialCandidate], output_path: Path):
        """保存材料到CSV"""
        data = []
        for mat in materials:
            data.append({
                'id': mat.id,
                'formula': mat.formula,
                'elements': ','.join(mat.elements),
                'volume': mat.volume,
                'a': mat.a,
                'b': mat.b,
                'c': mat.c,
                'alpha': mat.alpha,
                'beta': mat.beta,
                'gamma': mat.gamma,
                'spacegroup': mat.spacegroup,
                'point_group': mat.point_group,
                'crystal_system': mat.crystal_system,
                'band_gap': mat.band_gap,
                'energy_above_hull': mat.energy_above_hull,
                'is_stable': mat.is_stable,
                'mp_id': mat.mp_id,
                'mp_formula': mat.mp_formula,
                'is_polar': mat.is_polar
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def run(self):
        """运行完整流程"""
        print("\n" + "="*70)
        print("约束条件驱动的铁电材料生成")
        print("="*70)
        print(f"\n约束名称: {self.constraints.name}")
        print(f"描述: {self.constraints.description}")
        print(f"输出目录: {self.output_dir}")
        
        # 1. 加载模型
        self.load_models()
        
        # 2. 搜索已知材料
        known_materials = self.search_known_materials()
        
        # 3. 生成新材料
        print("\n" + "="*70)
        print("生成新候选材料")
        print("="*70)
        
        all_generated = []
        for i in range(1, self.constraints.generation.n_iterations + 1):
            passed_materials, _ = self.generate_iteration(i)
            all_generated.extend(passed_materials)
        
        # 保存所有生成的材料
        if all_generated:
            output_file = self.output_dir / 'generated_materials_all.csv'
            self._save_materials_to_csv(all_generated, output_file)
            print(f"\n已保存所有生成材料: {output_file}")
        
        # 4. MP数据库对比
        mp_matched = []
        if all_generated:
            mp_matched = self.compare_with_mp(all_generated)
            
            if mp_matched:
                # 二次筛选：对MP数据应用约束
                print("\n对MP匹配结果应用约束筛选...")
                final_passed, _ = self.filter.filter_batch(mp_matched)
                
                if final_passed:
                    output_file = self.output_dir / 'mp_matched_filtered.csv'
                    self._save_materials_to_csv(final_passed, output_file)
                    print(f"  最终通过 {len(final_passed)} 个材料")
                    print(f"  已保存: {output_file}")
                    
                    mp_matched = final_passed
        
        # 5. 生成报告
        if self.constraints.output.generate_report:
            self.generate_report(known_materials, all_generated, mp_matched)
        
        # 完成
        print("\n" + "="*70)
        print("完成!")
        print("="*70)
        print(f"\n结果摘要:")
        print(f"  已知材料匹配: {len(known_materials)} 个")
        print(f"  生成新材料: {len(all_generated)} 个")
        print(f"  MP匹配: {len(mp_matched)} 个")
        print(f"\n所有结果已保存到: {self.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='约束条件驱动的铁电材料生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --constraint config/example_titanate.json
  python main.py --constraint config/example_leadfree.json --output results/leadfree
  python main.py --constraint config/example_titanate.json --iterations 10 --candidates 200
        """
    )
    
    parser.add_argument(
        '--constraint',
        type=str,
        required=True,
        help='约束配置文件路径 (JSON)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        help='生成迭代次数（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--candidates',
        type=int,
        help='每次迭代生成的候选数量（覆盖配置文件中的设置）'
    )
    
    args = parser.parse_args()
    
    # 加载约束
    print("加载约束配置...")
    try:
        constraints = load_constraints(args.constraint)
    except Exception as e:
        print(f"错误: 无法加载约束文件 - {e}")
        sys.exit(1)
    
    # 覆盖命令行参数
    if args.output:
        constraints.output.output_dir = args.output
    
    if args.iterations:
        constraints.generation.n_iterations = args.iterations
    
    if args.candidates:
        constraints.generation.n_candidates = args.candidates
    
    # 打印约束摘要
    print("\n" + "="*70)
    print(constraints.summary)
    print("="*70)
    
    # 运行生成器
    generator = ConstrainedMaterialGenerator(constraints)
    
    try:
        generator.run()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
