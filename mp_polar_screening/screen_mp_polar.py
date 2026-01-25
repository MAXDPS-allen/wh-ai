"""
Materials Project 极性材料铁电筛选
=================================
从 Materials Project 获取所有极性材料，使用双模型筛选铁电材料
生成两个版本：
1. 包含训练数据的完整版本
2. 剔除训练数据的新发现版本

同时分析训练集中有多少铁电材料未被模型识别
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set, Tuple

# 路径设置
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'shared'))
sys.path.insert(0, str(ROOT_DIR / 'code_v5'))

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader

# 导入模型
from GCNN_v5 import GCNNClassifierV5, structure_to_graph
from NequIP_Classifier import NequIPClassifier, NequIPConfig, structure_to_nequip_graph
from feature_engineering import UnifiedFeatureExtractor, FEATURE_DIM

# MP API
from mp_api.client import MPRester

# 配置
API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"

# 极性空间群列表 (共68个)
POLAR_SPACE_GROUPS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9,  # Triclinic & Monoclinic polar
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34,  # Orthorhombic mm2
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,  # Orthorhombic
    75, 76, 77, 78, 79, 80,  # Tetragonal 4
    99, 100, 101, 102, 103, 104, 105, 106,  # Tetragonal 4mm
    107, 108, 109, 110,  # Tetragonal -4m2 (not polar but often considered)
    143, 144, 145, 146,  # Trigonal 3
    156, 157, 158, 159, 160, 161,  # Trigonal 3m
    168, 169, 170, 171, 172, 173,  # Hexagonal 6
    183, 184, 185, 186  # Hexagonal 6mm
]


class MPPolarScreener:
    """MP 极性材料筛选器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # 加载 GCNN 模型
        gcnn_path = ROOT_DIR / 'model_gcnn_v5' / 'gcnn_v5_model0_best.pt'
        self.gcnn_model = GCNNClassifierV5().to(self.device)
        self.gcnn_threshold = 0.87
        if gcnn_path.exists():
            checkpoint = torch.load(gcnn_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.gcnn_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                self.gcnn_model.load_state_dict(checkpoint['model'])
            else:
                self.gcnn_model.load_state_dict(checkpoint)
            self.gcnn_threshold = checkpoint.get('threshold', 0.87)
            print(f"Loaded GCNN from {gcnn_path}, threshold={self.gcnn_threshold:.4f}")
        self.gcnn_model.eval()
        
        # 加载 NequIP 模型
        nequip_path = ROOT_DIR / 'model_nequip' / 'nequip_classifier_final.pt'
        self.nequip_model = NequIPClassifier().to(self.device)
        self.nequip_threshold = 0.94
        if nequip_path.exists():
            checkpoint = torch.load(nequip_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.nequip_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                self.nequip_model.load_state_dict(checkpoint['model'])
            else:
                self.nequip_model.load_state_dict(checkpoint)
            self.nequip_threshold = checkpoint.get('threshold', 0.94)
            print(f"Loaded NequIP from {nequip_path}, threshold={self.nequip_threshold:.4f}")
        self.nequip_model.eval()
        
        # 特征提取器
        self.extractor = UnifiedFeatureExtractor()
        
        # 加载训练集中的铁电材料
        self.known_fe_formulas = self._load_known_ferroelectrics()
        print(f"Known ferroelectric formulas: {len(self.known_fe_formulas)}")
        
    def _load_known_ferroelectrics(self) -> Set[str]:
        """加载训练集中的已知铁电材料"""
        known_fe = set()
        
        # 1. dataset_original_ferroelectric.jsonl
        path1 = ROOT_DIR / 'new_data' / 'dataset_original_ferroelectric.jsonl'
        if path1.exists():
            with open(path1, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'formula' in item:
                            known_fe.add(item['formula'])
                        elif 'structure' in item:
                            s = Structure.from_dict(item['structure'])
                            known_fe.add(s.composition.reduced_formula)
                    except:
                        pass
            print(f"  dataset_original_ferroelectric: added {len(known_fe)} formulas")
        
        # 2. dataset_known_FE_rest.jsonl
        path2 = ROOT_DIR / 'new_data' / 'dataset_known_FE_rest.jsonl'
        if path2.exists():
            count_before = len(known_fe)
            with open(path2, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'formula' in item:
                            known_fe.add(item['formula'])
                        elif 'structure' in item:
                            s = Structure.from_dict(item['structure'])
                            known_fe.add(s.composition.reduced_formula)
                    except:
                        pass
            print(f"  dataset_known_FE_rest: added {len(known_fe) - count_before} formulas")
        
        return known_fe
    
    def structure_to_graph(self, structure: Structure, sg_number: int = 1) -> Data:
        """将结构转换为 GCNN 图数据"""
        try:
            struct_dict = structure.as_dict()
            global_feat = self.extractor.extract_from_structure_dict(struct_dict, sg_number)
            graph = structure_to_graph(struct_dict, 0, global_feat)
            return graph
        except Exception as e:
            return None
    
    def structure_to_nequip_data(self, structure: Structure, sg_number: int = 1) -> Data:
        """将结构转换为 NequIP 图数据"""
        try:
            struct_dict = structure.as_dict()
            global_feat = self.extractor.extract_from_structure_dict(struct_dict, sg_number)
            graph = structure_to_nequip_graph(struct_dict, 0, global_feat, cutoff=5.0)
            return graph
        except Exception as e:
            return None
    
    def predict_batch_gcnn(self, graphs: List[Data]) -> np.ndarray:
        """GCNN 批量预测"""
        if not graphs:
            return np.array([])
        
        batch = Batch.from_data_list(graphs).to(self.device)
        
        with torch.no_grad():
            gcnn_logits = self.gcnn_model(batch)
            gcnn_probs = torch.sigmoid(gcnn_logits).cpu().numpy()
        
        return gcnn_probs
    
    def predict_batch_nequip(self, graphs: List[Data]) -> np.ndarray:
        """NequIP 批量预测"""
        if not graphs:
            return np.array([])
        
        batch = Batch.from_data_list(graphs).to(self.device)
        
        with torch.no_grad():
            nequip_logits = self.nequip_model(batch)
            nequip_probs = torch.sigmoid(nequip_logits).cpu().numpy()
        
        return nequip_probs
    
    def fetch_polar_materials(self, chunk_size: int = 1000) -> List[Dict]:
        """从 MP 获取所有极性材料"""
        print("\n" + "="*60)
        print("从 Materials Project 获取极性材料")
        print("="*60)
        
        all_materials = []
        
        with MPRester(api_key=API_KEY) as mpr:
            print("Connected to Materials Project")
            
            # 查询极性空间群的材料
            for sg in tqdm(POLAR_SPACE_GROUPS, desc="Fetching space groups"):
                try:
                    docs = mpr.materials.summary.search(
                        spacegroup_number=sg,
                        fields=["material_id", "formula_pretty", "structure", 
                               "symmetry", "energy_above_hull", "is_stable"]
                    )
                    
                    for doc in docs:
                        # 从 symmetry 字段提取空间群信息
                        sg_num = doc.symmetry.number if doc.symmetry else sg
                        sg_symbol = doc.symmetry.symbol if doc.symmetry else "Unknown"
                        
                        all_materials.append({
                            'material_id': str(doc.material_id),
                            'formula': doc.formula_pretty,
                            'structure': doc.structure,
                            'spacegroup_number': sg_num,
                            'spacegroup_symbol': sg_symbol,
                            'energy_above_hull': doc.energy_above_hull,
                            'is_stable': doc.is_stable
                        })
                except Exception as e:
                    print(f"Error fetching SG {sg}: {e}")
                    continue
                
                time.sleep(0.1)  # 避免 API 限制
        
        print(f"\nTotal polar materials: {len(all_materials)}")
        return all_materials
    
    def screen_materials(
        self, 
        materials: List[Dict],
        batch_size: int = 64,
        output_dir: str = None
    ) -> pd.DataFrame:
        """筛选材料"""
        print("\n" + "="*60)
        print("铁电材料筛选")
        print("="*60)
        
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        batch_gcnn_graphs = []
        batch_nequip_graphs = []
        batch_info = []
        
        stats = {
            'total': len(materials),
            'processed': 0,
            'errors': 0,
            'gcnn_positive': 0,
            'nequip_positive': 0,
            'consensus_positive': 0
        }
        
        start_time = time.time()
        
        for mat in tqdm(materials, desc="Processing structures"):
            structure = mat['structure']
            sg_number = mat['spacegroup_number']
            
            # 分别为两个模型构建图
            gcnn_graph = self.structure_to_graph(structure, sg_number)
            nequip_graph = self.structure_to_nequip_data(structure, sg_number)
            
            if gcnn_graph is None or nequip_graph is None:
                stats['errors'] += 1
                continue
            
            batch_gcnn_graphs.append(gcnn_graph)
            batch_nequip_graphs.append(nequip_graph)
            batch_info.append({
                'material_id': mat['material_id'],
                'formula': mat['formula'],
                'spacegroup_number': sg_number,
                'spacegroup_symbol': mat['spacegroup_symbol'],
                'num_atoms': len(structure),
                'energy_above_hull': mat.get('energy_above_hull', None),
                'is_stable': mat.get('is_stable', None)
            })
            
            # 批量预测
            if len(batch_gcnn_graphs) >= batch_size:
                gcnn_probs = self.predict_batch_gcnn(batch_gcnn_graphs)
                nequip_probs = self.predict_batch_nequip(batch_nequip_graphs)
                
                for info, gp, np_ in zip(batch_info, gcnn_probs, nequip_probs):
                    gcnn_fe = gp >= self.gcnn_threshold
                    nequip_fe = np_ >= self.nequip_threshold
                    consensus = gcnn_fe and nequip_fe
                    
                    info['gcnn_probability'] = float(gp)
                    info['nequip_probability'] = float(np_)
                    info['gcnn_ferroelectric'] = bool(gcnn_fe)
                    info['nequip_ferroelectric'] = bool(nequip_fe)
                    info['consensus_ferroelectric'] = consensus
                    info['avg_probability'] = (float(gp) + float(np_)) / 2
                    info['is_known_fe'] = info['formula'] in self.known_fe_formulas
                    
                    results.append(info)
                    stats['processed'] += 1
                    
                    if gcnn_fe:
                        stats['gcnn_positive'] += 1
                    if nequip_fe:
                        stats['nequip_positive'] += 1
                    if consensus:
                        stats['consensus_positive'] += 1
                
                batch_gcnn_graphs = []
                batch_nequip_graphs = []
                batch_info = []
        
        # 处理剩余的
        if batch_gcnn_graphs:
            gcnn_probs = self.predict_batch_gcnn(batch_gcnn_graphs)
            nequip_probs = self.predict_batch_nequip(batch_nequip_graphs)
            
            for info, gp, np_ in zip(batch_info, gcnn_probs, nequip_probs):
                gcnn_fe = gp >= self.gcnn_threshold
                nequip_fe = np_ >= self.nequip_threshold
                consensus = gcnn_fe and nequip_fe
                
                info['gcnn_probability'] = float(gp)
                info['nequip_probability'] = float(np_)
                info['gcnn_ferroelectric'] = bool(gcnn_fe)
                info['nequip_ferroelectric'] = bool(nequip_fe)
                info['consensus_ferroelectric'] = consensus
                info['avg_probability'] = (float(gp) + float(np_)) / 2
                info['is_known_fe'] = info['formula'] in self.known_fe_formulas
                
                results.append(info)
                stats['processed'] += 1
                
                if gcnn_fe:
                    stats['gcnn_positive'] += 1
                if nequip_fe:
                    stats['nequip_positive'] += 1
                if consensus:
                    stats['consensus_positive'] += 1
        
        elapsed = time.time() - start_time
        
        # 统计结果
        print(f"\n=== 筛选完成 ===")
        print(f"总材料数: {stats['total']:,}")
        print(f"成功处理: {stats['processed']:,}")
        print(f"错误: {stats['errors']}")
        print(f"耗时: {elapsed/60:.1f} 分钟")
        print(f"\n预测结果:")
        print(f"  GCNN 预测铁电: {stats['gcnn_positive']:,} ({100*stats['gcnn_positive']/stats['processed']:.2f}%)")
        print(f"  NequIP 预测铁电: {stats['nequip_positive']:,} ({100*stats['nequip_positive']/stats['processed']:.2f}%)")
        print(f"  双模型一致铁电: {stats['consensus_positive']:,} ({100*stats['consensus_positive']/stats['processed']:.2f}%)")
        
        # 创建 DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('avg_probability', ascending=False)
        
        return df
    
    def analyze_and_save(self, df: pd.DataFrame, output_dir: Path = None):
        """分析结果并保存两个版本"""
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)
        
        print("\n" + "="*60)
        print("结果分析与保存")
        print("="*60)
        
        # 1. 完整版本（包含所有预测为铁电的材料）
        fe_all = df[df['consensus_ferroelectric'] == True].copy()
        fe_all = fe_all.sort_values('avg_probability', ascending=False)
        fe_all.to_csv(output_dir / 'ferroelectric_all.csv', index=False)
        print(f"\n完整版本: {len(fe_all)} 个铁电候选")
        print(f"  保存到: ferroelectric_all.csv")
        
        # 按化学式去重
        fe_all_unique = fe_all.drop_duplicates(subset='formula', keep='first')
        fe_all_unique.to_csv(output_dir / 'ferroelectric_all_unique.csv', index=False)
        print(f"  唯一化学式: {len(fe_all_unique)} 个")
        
        # 2. 新发现版本（剔除训练数据）
        fe_new = fe_all[fe_all['is_known_fe'] == False].copy()
        fe_new.to_csv(output_dir / 'ferroelectric_new.csv', index=False)
        print(f"\n新发现版本: {len(fe_new)} 个铁电候选")
        print(f"  保存到: ferroelectric_new.csv")
        
        # 按化学式去重
        fe_new_unique = fe_new.drop_duplicates(subset='formula', keep='first')
        fe_new_unique.to_csv(output_dir / 'ferroelectric_new_unique.csv', index=False)
        print(f"  唯一化学式: {len(fe_new_unique)} 个")
        
        # 3. 分析训练集中未被找到的铁电材料
        # 已知铁电中被预测为铁电的
        known_found = fe_all[fe_all['is_known_fe'] == True]
        known_found_formulas = set(known_found['formula'])
        
        # 已知铁电中未被预测为铁电的
        known_not_found = self.known_fe_formulas - known_found_formulas
        
        # 检查未找到的是否在数据集中（可能空间群不是极性的）
        all_formulas_in_df = set(df['formula'])
        known_in_polar = self.known_fe_formulas & all_formulas_in_df
        known_not_in_polar = self.known_fe_formulas - all_formulas_in_df
        
        # 在极性材料中但未被预测为铁电的
        known_in_polar_not_predicted = known_in_polar - known_found_formulas
        
        print(f"\n=== 训练集铁电材料分析 ===")
        print(f"训练集铁电总数: {len(self.known_fe_formulas)}")
        print(f"  存在于极性材料中: {len(known_in_polar)}")
        print(f"  不在极性材料中 (非极性空间群): {len(known_not_in_polar)}")
        print(f"\n极性材料中的已知铁电:")
        print(f"  被成功识别: {len(known_found_formulas)}")
        print(f"  未被识别 (漏检): {len(known_in_polar_not_predicted)}")
        
        if len(known_in_polar) > 0:
            recall_known = len(known_found_formulas) / len(known_in_polar)
            print(f"  已知铁电召回率: {100*recall_known:.2f}%")
        
        # 保存分析报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_polar_materials': len(df),
            'total_ferroelectric_predicted': len(fe_all),
            'unique_ferroelectric_formulas': len(fe_all_unique),
            'new_ferroelectric_predicted': len(fe_new),
            'unique_new_ferroelectric_formulas': len(fe_new_unique),
            'training_set': {
                'total_known_fe': len(self.known_fe_formulas),
                'in_polar_materials': len(known_in_polar),
                'not_in_polar_materials': len(known_not_in_polar),
                'successfully_identified': len(known_found_formulas),
                'missed': len(known_in_polar_not_predicted),
                'recall_on_known': len(known_found_formulas) / len(known_in_polar) if known_in_polar else 0
            }
        }
        
        with open(output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # 保存未被识别的铁电材料列表
        missed_df = df[df['formula'].isin(known_in_polar_not_predicted)]
        if len(missed_df) > 0:
            missed_df = missed_df.sort_values('avg_probability', ascending=False)
            missed_df.to_csv(output_dir / 'known_fe_missed.csv', index=False)
            print(f"\n未被识别的已知铁电保存到: known_fe_missed.csv")
            
            print(f"\n--- 未被识别的已知铁电 (Top 20) ---")
            for _, row in missed_df.head(20).iterrows():
                print(f"  {row['formula']:20s} | SG: {row['spacegroup_number']:3d} | "
                      f"GCNN: {row['gcnn_probability']:.4f} | NequIP: {row['nequip_probability']:.4f}")
        
        # 保存不在极性空间群中的铁电
        with open(output_dir / 'known_fe_non_polar.txt', 'w') as f:
            f.write("# 训练集中不在极性空间群的铁电材料\n")
            f.write(f"# 总计: {len(known_not_in_polar)} 个\n\n")
            for formula in sorted(known_not_in_polar):
                f.write(f"{formula}\n")
        
        print(f"\n=== 分析完成 ===")
        print(f"所有结果保存在: {output_dir}")
        
        return report


def main():
    """主函数"""
    print("="*60)
    print("Materials Project 极性材料铁电筛选")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 输出目录
    output_dir = Path(__file__).parent
    
    # 初始化筛选器
    screener = MPPolarScreener()
    
    # 从 MP 获取极性材料
    materials = screener.fetch_polar_materials()
    
    if not materials:
        print("未获取到任何材料!")
        return
    
    # 保存原始数据（可选）
    print(f"\n保存原始极性材料信息...")
    raw_info = [{
        'material_id': m['material_id'],
        'formula': m['formula'],
        'spacegroup_number': m['spacegroup_number'],
        'spacegroup_symbol': m['spacegroup_symbol'],
        'num_atoms': len(m['structure']),
        'energy_above_hull': m.get('energy_above_hull'),
        'is_stable': m.get('is_stable')
    } for m in materials]
    pd.DataFrame(raw_info).to_csv(output_dir / 'mp_polar_materials_raw.csv', index=False)
    
    # 筛选
    df = screener.screen_materials(materials, batch_size=64, output_dir=output_dir)
    
    # 保存完整预测结果
    df.to_csv(output_dir / 'mp_polar_predictions_all.csv', index=False)
    
    # 分析并保存两个版本
    report = screener.analyze_and_save(df, output_dir)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
