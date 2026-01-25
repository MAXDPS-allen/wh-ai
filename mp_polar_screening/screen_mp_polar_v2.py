"""
MP Polar Materials Screening v2
================================
使用新训练的 GCNN v6 + NequIP v2 模型重新筛选
直接从 MP API 获取极性材料并筛选
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'code_v5'))
sys.path.insert(0, str(ROOT_DIR / 'shared'))

from pymatgen.core import Structure
from torch_geometric.data import Data

# 导入特征工程
from feature_engineering import UnifiedFeatureExtractor, FEATURE_DIM, ELEMENT_DATABASE

# 导入训练脚本中的模型定义
from GCNN_v6 import GCNNClassifierV6
from NequIP_Classifier_v2 import NequIPClassifierV2

# MP API
from mp_api.client import MPRester

# 配置
API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"

# 极性空间群列表 (共68个)
POLAR_SPACE_GROUPS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    75, 76, 77, 78, 79, 80,
    99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110,
    143, 144, 145, 146,
    156, 157, 158, 159, 160, 161,
    168, 169, 170, 171, 172, 173,
    183, 184, 185, 186
]


def load_models(device):
    """加载GCNN v6和NequIP v2模型"""
    
    # 加载GCNN v6模型
    gcnn_models = []
    gcnn_dir = ROOT_DIR / 'model_gcnn_v6'
    
    print("Loading GCNN v6 models...")
    for i in range(3):
        model_path = gcnn_dir / f'gcnn_v6_model{i}_best.pt'
        if model_path.exists():
            model = GCNNClassifierV6().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            gcnn_models.append(model)
            print(f"  Loaded model {i}")
    
    # 加载GCNN v6阈值
    gcnn_config_path = gcnn_dir / 'training_config.json'
    if gcnn_config_path.exists():
        with open(gcnn_config_path) as f:
            gcnn_config = json.load(f)
        gcnn_threshold = gcnn_config.get('best_threshold', 0.01)
    else:
        gcnn_threshold = 0.01
    
    print(f"GCNN v6 threshold: {gcnn_threshold}")
    
    # 加载NequIP v2模型
    nequip_dir = ROOT_DIR / 'model_nequip_v2'
    nequip_path = nequip_dir / 'nequip_v2_best.pt'
    
    print("\nLoading NequIP v2 model...")
    nequip_model = NequIPClassifierV2().to(device)
    checkpoint = torch.load(nequip_path, map_location=device, weights_only=False)
    nequip_model.load_state_dict(checkpoint['model'])
    nequip_model.eval()
    print("  Loaded successfully")
    
    # 加载NequIP v2阈值
    nequip_config_path = nequip_dir / 'training_config.json'
    if nequip_config_path.exists():
        with open(nequip_config_path) as f:
            nequip_config = json.load(f)
        nequip_threshold = nequip_config.get('best_threshold', 0.01)
    else:
        nequip_threshold = 0.01
    
    print(f"NequIP v2 threshold: {nequip_threshold}")
    
    return gcnn_models, gcnn_threshold, nequip_model, nequip_threshold


def structure_to_gcnn_data(structure: Structure, global_features, device):
    """转换结构为GCNN输入格式"""
    try:
        node_features = []
        for site in structure:
            el = site.specie.symbol
            if el in ELEMENT_DATABASE:
                data = ELEMENT_DATABASE[el]
                feat = [
                    data[0] / 100.0, data[1] / 200.0, data[2] / 2.5, data[3] / 4.0,
                    data[4] / 15.0, data[5] / 8.0, data[6] / 3500.0, data[7] / 200.0,
                    site.frac_coords[0], site.frac_coords[1], site.frac_coords[2],
                    np.sin(2 * np.pi * site.frac_coords[0]),
                    np.sin(2 * np.pi * site.frac_coords[1]),
                    np.sin(2 * np.pi * site.frac_coords[2]),
                    np.cos(2 * np.pi * site.frac_coords[0]),
                    np.cos(2 * np.pi * site.frac_coords[1]),
                ]
            else:
                feat = [0.5] * 16
            node_features.append(feat)
        
        x = torch.tensor(node_features, dtype=torch.float, device=device)
        
        edge_index = []
        cutoff = 5.0
        for i, site_i in enumerate(structure):
            neighbors = structure.get_neighbors(site_i, cutoff)
            for neighbor in neighbors:
                j = neighbor.index
                if i != j:
                    edge_index.append([i, j])
        
        if not edge_index:
            for i in range(len(structure)):
                for j in range(len(structure)):
                    if i != j:
                        edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).T
        u = torch.tensor(global_features, dtype=torch.float, device=device).unsqueeze(0)
        batch = torch.zeros(len(node_features), dtype=torch.long, device=device)
        
        return Data(x=x, edge_index=edge_index, u=u, batch=batch)
    except:
        return None


def structure_to_nequip_data(structure: Structure, global_features, device, cutoff=5.0):
    """转换结构为NequIP输入格式"""
    try:
        atomic_numbers = [site.specie.Z for site in structure]
        x = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
        
        edge_index, edge_vec, edge_length = [], [], []
        
        for i, site_i in enumerate(structure):
            neighbors = structure.get_neighbors(site_i, cutoff)
            for neighbor in neighbors:
                j = neighbor.index
                if i != j:
                    edge_index.append([i, j])
                    vec = neighbor.coords - site_i.coords
                    edge_vec.append(vec)
                    edge_length.append(neighbor.nn_distance)
        
        if not edge_index:
            n = len(atomic_numbers)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index.append([i, j])
                        vec_frac = structure[j].frac_coords - structure[i].frac_coords
                        vec = structure.lattice.get_cartesian_coords(vec_frac)
                        dist = max(np.linalg.norm(vec), 1.0)
                        edge_vec.append(vec)
                        edge_length.append(dist)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).T
        edge_vec = torch.tensor(np.array(edge_vec), dtype=torch.float, device=device)
        edge_length = torch.tensor(edge_length, dtype=torch.float, device=device)
        u = torch.tensor(global_features, dtype=torch.float, device=device).unsqueeze(0)
        batch = torch.zeros(len(atomic_numbers), dtype=torch.long, device=device)
        
        return Data(x=x, edge_index=edge_index, edge_vec=edge_vec, edge_length=edge_length, u=u, batch=batch)
    except:
        return None


def fetch_mp_polar_materials():
    """从 MP API 获取所有极性材料"""
    print("Fetching polar materials from Materials Project...")
    
    all_materials = []
    
    with MPRester(API_KEY) as mpr:
        for sg in tqdm(POLAR_SPACE_GROUPS, desc="Fetching space groups"):
            try:
                docs = mpr.materials.summary.search(
                    spacegroup_number=sg,
                    fields=['material_id', 'formula_pretty', 'structure', 
                            'symmetry', 'energy_above_hull', 'is_stable']
                )
                
                for doc in docs:
                    mat = {
                        'material_id': str(doc.material_id),
                        'formula': doc.formula_pretty,
                        'structure': doc.structure,
                        'spacegroup_number': sg,
                        'spacegroup_symbol': doc.symmetry.symbol if doc.symmetry else '',
                        'energy_above_hull': doc.energy_above_hull,
                        'is_stable': doc.is_stable,
                    }
                    all_materials.append(mat)
                
                time.sleep(0.1)
            except Exception as e:
                print(f"  Error fetching SG {sg}: {e}")
                continue
    
    print(f"Total polar materials fetched: {len(all_materials)}")
    return all_materials


def screen_mp_polar_materials():
    """使用新模型筛选MP极性材料"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    gcnn_models, gcnn_thresh, nequip_model, nequip_thresh = load_models(device)
    
    # 从 MP 获取极性材料
    materials = fetch_mp_polar_materials()
    
    # 加载训练数据用于过滤
    data_dir = ROOT_DIR / 'new_data'
    
    # 加载正样本
    positive_formulas = set()
    for fname in ['dataset_original_ferroelectric.jsonl', 'dataset_known_FE_rest.jsonl']:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    item = json.loads(line)
                    formula = item.get('formula', item.get('pretty_formula', ''))
                    if formula:
                        positive_formulas.add(formula)
    
    # 加载负样本
    negative_formulas = set()
    neg_path = data_dir / 'dataset_nonFE_expanded.jsonl'
    if neg_path.exists():
        with open(neg_path) as f:
            for line in f:
                item = json.loads(line)
                formula = item.get('formula', item.get('pretty_formula', ''))
                if formula:
                    negative_formulas.add(formula)
    
    print(f"\nTraining positives: {len(positive_formulas)}")
    print(f"Training negatives: {len(negative_formulas)}")
    
    # 初始化特征提取器
    extractor = UnifiedFeatureExtractor()
    
    # 筛选结果
    all_predictions = []  # 所有预测
    fe_candidates = []  # 完整候选列表
    new_discoveries = []  # 排除训练数据的新发现
    
    print(f"\nTotal materials to screen: {len(materials)}")
    print("\nScreening with GCNN v6 + NequIP v2...")
    
    for mat in tqdm(materials, desc="Screening"):
        try:
            material_id = mat['material_id']
            formula = mat['formula']
            structure = mat['structure']
            sg = mat['spacegroup_number']
            
            # 提取特征 - 需要结构字典
            struct_dict = structure.as_dict()
            global_feat = extractor.extract_from_structure_dict(struct_dict, sg)
            
            # GCNN v6 预测
            gcnn_data = structure_to_gcnn_data(structure, global_feat, device)
            if gcnn_data is None:
                continue
            
            gcnn_probs = []
            with torch.no_grad():
                for model in gcnn_models:
                    logits = model(gcnn_data)
                    prob = torch.sigmoid(logits).item()
                    gcnn_probs.append(prob)
            gcnn_prob = np.mean(gcnn_probs)
            gcnn_pred = gcnn_prob >= gcnn_thresh
            
            # NequIP v2 预测
            nequip_data = structure_to_nequip_data(structure, global_feat, device)
            if nequip_data is None:
                continue
            
            with torch.no_grad():
                logits = nequip_model(nequip_data)
                nequip_prob = torch.sigmoid(logits).item()
            nequip_pred = nequip_prob >= nequip_thresh
            
            # 双模型一致预测
            consensus = gcnn_pred and nequip_pred
            
            # 检查是否为已知训练样本
            is_known_fe = formula in positive_formulas
            is_known_non_fe = formula in negative_formulas
            
            pred_record = {
                'material_id': material_id,
                'formula': formula,
                'spacegroup_number': sg,
                'spacegroup_symbol': mat.get('spacegroup_symbol', ''),
                'energy_above_hull': mat.get('energy_above_hull'),
                'is_stable': mat.get('is_stable'),
                'gcnn_v6_prob': float(gcnn_prob),
                'nequip_v2_prob': float(nequip_prob),
                'gcnn_ferroelectric': gcnn_pred,
                'nequip_ferroelectric': nequip_pred,
                'consensus_ferroelectric': consensus,
                'is_known_fe': is_known_fe,
                'is_known_non_fe': is_known_non_fe,
            }
            all_predictions.append(pred_record)
            
            if consensus:
                fe_candidates.append(pred_record)
                
                # 检查是否为新发现
                if not is_known_fe and not is_known_non_fe:
                    new_discoveries.append(pred_record)
        
        except Exception as e:
            continue
    
    # 保存结果
    output_dir = Path(__file__).parent
    
    # 所有预测结果
    all_df = pd.DataFrame(all_predictions)
    all_csv = output_dir / 'mp_polar_predictions_v2_all.csv'
    all_df.to_csv(all_csv, index=False)
    print(f"\nAll predictions saved to: {all_csv}")
    
    # 完整候选列表
    fe_df = pd.DataFrame(fe_candidates)
    fe_csv = output_dir / 'fe_candidates_v2_complete.csv'
    fe_df.to_csv(fe_csv, index=False)
    print(f"FE candidates (complete): {len(fe_candidates)}")
    print(f"Saved to: {fe_csv}")
    
    # 新发现列表
    new_df = pd.DataFrame(new_discoveries)
    new_csv = output_dir / 'fe_candidates_v2_new.csv'
    new_df.to_csv(new_csv, index=False)
    print(f"\nNew FE discoveries: {len(new_discoveries)}")
    print(f"Saved to: {new_csv}")
    
    # 分析训练集召回
    known_fe_in_polar = [p for p in all_predictions if p['is_known_fe']]
    known_fe_recalled = [p for p in known_fe_in_polar if p['consensus_ferroelectric']]
    known_fe_missed = [p for p in known_fe_in_polar if not p['consensus_ferroelectric']]
    
    print(f"\n=== Training Set Recall Analysis ===")
    print(f"Known FE in polar materials: {len(known_fe_in_polar)}")
    print(f"Correctly recalled: {len(known_fe_recalled)}")
    print(f"Missed: {len(known_fe_missed)}")
    if len(known_fe_in_polar) > 0:
        print(f"Recall rate: {len(known_fe_recalled)/len(known_fe_in_polar)*100:.2f}%")
    
    # 保存未召回的已知铁电
    if known_fe_missed:
        missed_df = pd.DataFrame(known_fe_missed)
        missed_csv = output_dir / 'known_fe_missed_v2.csv'
        missed_df.to_csv(missed_csv, index=False)
        print(f"Missed known FE saved to: {missed_csv}")
    
    # 保存筛选报告
    report = {
        'screening_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_materials_screened': len(materials),
        'gcnn_v6_threshold': gcnn_thresh,
        'nequip_v2_threshold': nequip_thresh,
        'total_fe_candidates': len(fe_candidates),
        'new_discoveries': len(new_discoveries),
        'training_positives': len(positive_formulas),
        'training_negatives': len(negative_formulas),
        'known_fe_in_polar': len(known_fe_in_polar),
        'known_fe_recalled': len(known_fe_recalled),
        'known_fe_missed': len(known_fe_missed),
        'recall_rate': len(known_fe_recalled)/len(known_fe_in_polar)*100 if len(known_fe_in_polar) > 0 else 0,
    }
    
    report_file = output_dir / 'screening_v2_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_file}")
    
    # 打印统计
    print("\n" + "="*60)
    print("SCREENING v2 SUMMARY")
    print("="*60)
    print(f"Total materials screened: {len(materials)}")
    print(f"FE candidates (complete): {len(fe_candidates)}")
    print(f"New discoveries: {len(new_discoveries)}")
    print(f"Known FE recall rate: {report['recall_rate']:.2f}%")
    print("="*60)


if __name__ == '__main__':
    screen_mp_polar_materials()
