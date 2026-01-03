"""
MP 原始数据筛选脚本
====================
只筛选标记为 1 和 5 的文件（MP 爬取的原始数据）
其他 2, 3, 4 是插值结果，不参与筛选

使用双模型 (GCNN + NequIP) 进行筛选
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

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'code_v5'))

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data, Batch

from GCNN_v5 import GCNNClassifierV5, structure_to_graph
from NequIP_Classifier import NequIPClassifier, NequIPConfig, structure_to_nequip_graph
from feature_engineering import UnifiedFeatureExtractor


class MPOriginalScreener:
    """MP 原始数据筛选器 (仅筛选端点 1 和 5)"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.extractor = UnifiedFeatureExtractor()
        
        # 加载 GCNN 模型
        print("\n加载 GCNN 模型...")
        self.gcnn_model = GCNNClassifierV5().to(self.device)
        gcnn_path = Path(__file__).parent.parent / 'model_gcnn_v5' / 'gcnn_v5_model0_best.pt'
        gcnn_ckpt = torch.load(gcnn_path, map_location=self.device, weights_only=False)
        self.gcnn_model.load_state_dict(gcnn_ckpt['model'])
        self.gcnn_model.eval()
        self.gcnn_threshold = gcnn_ckpt.get('threshold', 0.87)
        print(f"  GCNN 阈值: {self.gcnn_threshold:.4f}")
        
        # 加载 NequIP 模型
        print("\n加载 NequIP 模型...")
        self.nequip_config = NequIPConfig()
        self.nequip_model = NequIPClassifier(self.nequip_config).to(self.device)
        nequip_path = Path(__file__).parent.parent / 'model_nequip' / 'nequip_classifier_final.pt'
        nequip_ckpt = torch.load(nequip_path, map_location=self.device, weights_only=False)
        self.nequip_model.load_state_dict(nequip_ckpt['model_state_dict'])
        self.nequip_model.eval()
        self.nequip_threshold = nequip_ckpt.get('threshold', 0.94)
        print(f"  NequIP 阈值: {self.nequip_threshold:.4f}")
    
    def process_structure(self, filepath: str) -> dict:
        """处理单个结构文件"""
        try:
            structure = Structure.from_file(filepath)
            formula = structure.composition.reduced_formula
            
            # 空间群分析
            try:
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                sg_number = sga.get_space_group_number()
                sg_symbol = sga.get_space_group_symbol()
            except:
                sg_number = 1
                sg_symbol = "P1"
            
            # 提取特征
            struct_dict = structure.as_dict()
            global_feat = self.extractor.extract_from_structure_dict(struct_dict, sg_number)
            
            # 构建图
            gcnn_graph = structure_to_graph(struct_dict, 0, global_feat)
            nequip_graph = structure_to_nequip_graph(struct_dict, 0, global_feat, 5.0)
            
            if gcnn_graph is None or nequip_graph is None:
                return None
            
            # 预测
            gcnn_batch = Batch.from_data_list([gcnn_graph]).to(self.device)
            nequip_batch = Batch.from_data_list([nequip_graph]).to(self.device)
            
            with torch.no_grad():
                gcnn_prob = torch.sigmoid(self.gcnn_model(gcnn_batch)).item()
                nequip_prob = torch.sigmoid(self.nequip_model(nequip_batch)).item()
            
            # 判断端点
            if '1_normalized.vasp' in filepath or '/1.vasp' in filepath:
                endpoint = '1'
            elif '5_normalized.vasp' in filepath or '/5.vasp' in filepath:
                endpoint = '5'
            else:
                endpoint = 'unknown'
            
            return {
                'filepath': filepath,
                'formula': formula,
                'spacegroup_number': sg_number,
                'spacegroup_symbol': sg_symbol,
                'num_atoms': len(structure),
                'endpoint': endpoint,
                'gcnn_probability': gcnn_prob,
                'nequip_probability': nequip_prob,
                'gcnn_ferroelectric': gcnn_prob >= self.gcnn_threshold,
                'nequip_ferroelectric': nequip_prob >= self.nequip_threshold,
                'consensus_ferroelectric': (gcnn_prob >= self.gcnn_threshold) and (nequip_prob >= self.nequip_threshold),
                'avg_probability': (gcnn_prob + nequip_prob) / 2
            }
        except Exception as e:
            return None
    
    def screen(self, database_dir: str, use_normalized: bool = True):
        """筛选 MP 原始数据"""
        print("\n" + "="*60)
        print("MP 原始数据筛选 (仅端点 1 和 5)")
        print("="*60)
        
        # 只查找 1 和 5 文件
        if use_normalized:
            files_1 = glob.glob(os.path.join(database_dir, "**", "1_normalized.vasp"), recursive=True)
            files_5 = glob.glob(os.path.join(database_dir, "**", "5_normalized.vasp"), recursive=True)
        else:
            files_1 = glob.glob(os.path.join(database_dir, "**", "1.vasp"), recursive=True)
            files_5 = glob.glob(os.path.join(database_dir, "**", "5.vasp"), recursive=True)
        
        all_files = files_1 + files_5
        
        print(f"\nMP 原始数据文件:")
        print(f"  端点 1: {len(files_1):,}")
        print(f"  端点 5: {len(files_5):,}")
        print(f"  总计: {len(all_files):,}")
        
        # 筛选
        results = []
        errors = 0
        
        start_time = time.time()
        
        for filepath in tqdm(all_files, desc="筛选中"):
            result = self.process_structure(filepath)
            if result:
                results.append(result)
            else:
                errors += 1
        
        elapsed = time.time() - start_time
        
        # 转换为 DataFrame
        df = pd.DataFrame(results)
        
        # 统计
        print("\n" + "="*60)
        print("筛选结果统计")
        print("="*60)
        print(f"成功处理: {len(df):,}")
        print(f"处理错误: {errors}")
        print(f"耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        print(f"速度: {len(df)/elapsed:.1f} 样本/秒")
        
        print(f"\n--- 按模型统计 ---")
        print(f"GCNN 预测铁电: {df['gcnn_ferroelectric'].sum():,} ({100*df['gcnn_ferroelectric'].mean():.2f}%)")
        print(f"NequIP 预测铁电: {df['nequip_ferroelectric'].sum():,} ({100*df['nequip_ferroelectric'].mean():.2f}%)")
        print(f"双模型一致铁电: {df['consensus_ferroelectric'].sum():,} ({100*df['consensus_ferroelectric'].mean():.2f}%)")
        
        # 去重统计
        unique = df.drop_duplicates(subset=['formula'])
        unique_fe = unique[unique['consensus_ferroelectric']]
        
        print(f"\n--- 去重后统计 ---")
        print(f"唯一化学式: {len(unique):,}")
        print(f"唯一铁电候选: {len(unique_fe):,} ({100*len(unique_fe)/len(unique):.2f}%)")
        
        # 按端点统计
        print(f"\n--- 按端点统计 ---")
        for ep in ['1', '5']:
            ep_df = df[df['endpoint'] == ep]
            ep_fe = ep_df['consensus_ferroelectric'].sum()
            print(f"  端点 {ep}: {len(ep_df):,} 总, {ep_fe:,} 铁电 ({100*ep_fe/len(ep_df):.2f}%)")
        
        # 保存结果
        output_dir = Path(__file__).parent
        
        # 全部结果
        df.to_csv(output_dir / 'mp_original_all.csv', index=False)
        
        # 铁电候选
        fe_df = df[df['consensus_ferroelectric']].sort_values('avg_probability', ascending=False)
        fe_df.to_csv(output_dir / 'mp_original_ferroelectric.csv', index=False)
        
        # 去重铁电候选
        unique_fe_sorted = unique_fe.sort_values('avg_probability', ascending=False)
        unique_fe_sorted.to_csv(output_dir / 'mp_original_ferroelectric_unique.csv', index=False)
        
        print(f"\n结果保存到:")
        print(f"  {output_dir / 'mp_original_all.csv'}")
        print(f"  {output_dir / 'mp_original_ferroelectric.csv'}")
        print(f"  {output_dir / 'mp_original_ferroelectric_unique.csv'}")
        
        # Top 30
        print("\n" + "="*60)
        print("Top 30 双模型一致铁电候选 (去重)")
        print("="*60)
        for _, row in unique_fe_sorted.head(30).iterrows():
            print(f"  {row['formula']:20s} | SG: {row['spacegroup_number']:3d} | "
                  f"GCNN: {row['gcnn_probability']:.4f} | NequIP: {row['nequip_probability']:.4f}")
        
        return df


def main():
    database_dir = Path(__file__).parent.parent / 'new_data' / 'database'
    
    print("="*60)
    print("MP 原始数据铁电材料筛选")
    print("只筛选端点 1 和 5 (MP 爬取的原始结构)")
    print("排除插值结构 (2, 3, 4)")
    print("="*60)
    
    screener = MPOriginalScreener()
    results = screener.screen(str(database_dir), use_normalized=True)


if __name__ == '__main__':
    main()
