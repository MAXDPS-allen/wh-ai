"""
铁电材料数据库筛选脚本
=======================
使用 GCNN 分类器对数据库中的结构进行铁电性预测

数据库: new_data/database/
结构格式: VASP (POSCAR)
"""

import os
import sys
import glob
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent))

from pymatgen.core import Structure
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader

from GCNN_v5 import GCNNClassifierV5, structure_to_graph
from feature_engineering import UnifiedFeatureExtractor, FEATURE_DIM


class DatabaseScreener:
    """数据库筛选器"""
    
    def __init__(self, model_path: str = None, threshold: float = 0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # 加载模型
        self.model = GCNNClassifierV5().to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.threshold = checkpoint.get('threshold', threshold)
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                self.threshold = checkpoint.get('threshold', threshold)
            else:
                self.model.load_state_dict(checkpoint)
                self.threshold = threshold
            print(f"Loaded model from {model_path}")
            print(f"Threshold: {self.threshold:.4f}")
        else:
            self.threshold = threshold
            print("Using untrained model (for testing)")
        
        self.model.eval()
        
        # 特征提取器
        self.extractor = UnifiedFeatureExtractor()
        
        # 统计
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'errors': 0,
            'ferroelectric': 0,
            'non_ferroelectric': 0
        }
    
    def load_vasp_structure(self, filepath: str) -> Structure:
        """加载 VASP 结构文件"""
        try:
            return Structure.from_file(filepath)
        except Exception as e:
            return None
    
    def structure_to_graph_data(self, structure: Structure) -> Data:
        """将结构转换为图数据"""
        try:
            # 获取空间群
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            try:
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                sg_number = sga.get_space_group_number()
            except:
                sg_number = 1
            
            # 提取全局特征
            struct_dict = structure.as_dict()
            global_feat = self.extractor.extract_from_structure_dict(struct_dict, sg_number)
            
            # 构建图
            graph = structure_to_graph(struct_dict, 0, global_feat)
            return graph
        except Exception as e:
            return None
    
    def predict_batch(self, graphs: list) -> np.ndarray:
        """批量预测"""
        if not graphs:
            return np.array([])
        
        batch = Batch.from_data_list(graphs).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return probs
    
    def screen_database(
        self, 
        database_dir: str,
        output_file: str = None,
        batch_size: int = 64,
        use_normalized: bool = True
    ):
        """
        筛选数据库
        
        Args:
            database_dir: 数据库目录
            output_file: 输出文件路径
            batch_size: 批处理大小
            use_normalized: 是否使用 normalized 文件
        """
        print("\n" + "="*60)
        print("铁电材料数据库筛选")
        print("="*60)
        
        # 查找所有结构文件
        pattern = "*_normalized.vasp" if use_normalized else "*.vasp"
        all_files = glob.glob(os.path.join(database_dir, "**", pattern), recursive=True)
        
        # 过滤掉非 normalized 文件（如果使用 normalized）
        if use_normalized:
            all_files = [f for f in all_files if "_normalized.vasp" in f]
        else:
            all_files = [f for f in all_files if "_normalized.vasp" not in f]
        
        self.stats['total_files'] = len(all_files)
        print(f"找到 {len(all_files):,} 个结构文件")
        
        if not all_files:
            print("没有找到结构文件!")
            return
        
        # 结果存储
        results = []
        
        # 批量处理
        batch_graphs = []
        batch_info = []  # (filepath, formula, spacegroup)
        
        start_time = time.time()
        
        for filepath in tqdm(all_files, desc="处理结构"):
            # 加载结构
            structure = self.load_vasp_structure(filepath)
            if structure is None:
                self.stats['errors'] += 1
                continue
            
            # 转换为图
            graph = self.structure_to_graph_data(structure)
            if graph is None:
                self.stats['errors'] += 1
                continue
            
            # 提取信息
            formula = structure.composition.reduced_formula
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                sg_number = sga.get_space_group_number()
                sg_symbol = sga.get_space_group_symbol()
            except:
                sg_number = 1
                sg_symbol = "P1"
            
            batch_graphs.append(graph)
            batch_info.append({
                'filepath': filepath,
                'formula': formula,
                'spacegroup_number': sg_number,
                'spacegroup_symbol': sg_symbol,
                'num_atoms': len(structure)
            })
            
            # 批量预测
            if len(batch_graphs) >= batch_size:
                probs = self.predict_batch(batch_graphs)
                
                for info, prob in zip(batch_info, probs):
                    is_fe = prob >= self.threshold
                    info['probability'] = float(prob)
                    info['is_ferroelectric'] = is_fe
                    results.append(info)
                    
                    if is_fe:
                        self.stats['ferroelectric'] += 1
                    else:
                        self.stats['non_ferroelectric'] += 1
                
                self.stats['processed'] += len(batch_graphs)
                batch_graphs = []
                batch_info = []
        
        # 处理剩余
        if batch_graphs:
            probs = self.predict_batch(batch_graphs)
            
            for info, prob in zip(batch_info, probs):
                is_fe = prob >= self.threshold
                info['probability'] = float(prob)
                info['is_ferroelectric'] = is_fe
                results.append(info)
                
                if is_fe:
                    self.stats['ferroelectric'] += 1
                else:
                    self.stats['non_ferroelectric'] += 1
            
            self.stats['processed'] += len(batch_graphs)
        
        elapsed = time.time() - start_time
        
        # 打印统计
        print("\n" + "="*60)
        print("筛选完成!")
        print("="*60)
        print(f"总文件数: {self.stats['total_files']:,}")
        print(f"成功处理: {self.stats['processed']:,}")
        print(f"处理错误: {self.stats['errors']:,}")
        print(f"预测为铁电: {self.stats['ferroelectric']:,} ({100*self.stats['ferroelectric']/max(1,self.stats['processed']):.2f}%)")
        print(f"预测为非铁电: {self.stats['non_ferroelectric']:,}")
        print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        print(f"平均速度: {self.stats['processed']/elapsed:.1f} 样本/秒")
        
        # 保存结果
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(Path(__file__).parent.parent / 'discovery_results' / f'database_screening_{timestamp}.csv')
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df = pd.DataFrame(results)
        df = df.sort_values('probability', ascending=False)
        df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
        
        # 保存铁电候选
        fe_df = df[df['is_ferroelectric'] == True]
        fe_output = output_file.replace('.csv', '_ferroelectric.csv')
        fe_df.to_csv(fe_output, index=False)
        print(f"铁电候选已保存到: {fe_output}")
        
        # 按化学式统计
        print("\n" + "="*60)
        print("Top 20 铁电候选材料:")
        print("="*60)
        for i, row in fe_df.head(20).iterrows():
            print(f"  {row['formula']:20s} | SG: {row['spacegroup_number']:3d} | Prob: {row['probability']:.4f}")
        
        return df


def main():
    # 配置
    database_dir = Path(__file__).parent.parent / 'new_data' / 'database'
    model_path = Path(__file__).parent.parent / 'model_gcnn_v5' / 'gcnn_v5_model0_best.pt'
    
    # 检查模型
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / 'model_gcnn_v5' / 'gcnn_v5_model0_final.pt'
    
    print(f"Database: {database_dir}")
    print(f"Model: {model_path}")
    
    # 创建筛选器
    screener = DatabaseScreener(
        model_path=str(model_path),
        threshold=0.5  # 会被模型检查点覆盖
    )
    
    # 开始筛选
    results = screener.screen_database(
        database_dir=str(database_dir),
        batch_size=64,
        use_normalized=True
    )


if __name__ == '__main__':
    main()
