"""
扩展负样本数据集
================
1. 清理训练数据中的标签冲突
2. 从MP获取更多极性非铁电材料
3. 生成新的训练数据集

极性非铁电材料的定义:
- 属于极性空间群 (1-10, 25-46, 75-80, 99-110, 143-146, 156-161, 168-173, 183-186)
- 不在已知铁电材料列表中
- 优先选择:
  - 已知的反铁电材料
  - 已知的压电但非铁电材料
  - 稳定的极性材料（能量低于hull）
"""

import json
from pathlib import Path
from mp_api.client import MPRester
from pymatgen.core import Structure, Composition
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# MP API密钥
MP_API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"

# 极性空间群
POLAR_SPACE_GROUPS = list(range(1, 11)) + list(range(25, 47)) + list(range(75, 81)) + \
                     list(range(99, 111)) + list(range(143, 147)) + list(range(156, 162)) + \
                     list(range(168, 174)) + list(range(183, 187))


def load_known_ferroelectrics():
    """加载已知铁电材料"""
    root = Path(__file__).parent
    known_fe = set()
    
    fe_files = [
        root / 'dataset_original_ferroelectric.jsonl',
        root / 'dataset_known_FE_rest.jsonl'
    ]
    
    for f in fe_files:
        if f.exists():
            with open(f, 'r') as fp:
                for line in fp:
                    try:
                        item = json.loads(line)
                        if 'formula' in item:
                            comp = Composition(item['formula'])
                            known_fe.add(comp.reduced_formula)
                        elif 'structure' in item:
                            s = Structure.from_dict(item['structure'])
                            known_fe.add(s.composition.reduced_formula)
                    except:
                        pass
    
    print(f"已知铁电材料: {len(known_fe)} 个")
    return known_fe


def load_current_negative_samples():
    """加载当前负样本"""
    root = Path(__file__).parent
    non_fe = set()
    structures = []
    
    non_fe_files = [
        root / 'dataset_nonFE.jsonl',
        root / 'dataset_polar_non_ferroelectric_final.jsonl'
    ]
    
    for f in non_fe_files:
        if f.exists():
            with open(f, 'r') as fp:
                for line in fp:
                    try:
                        item = json.loads(line)
                        formula = None
                        struct = None
                        
                        if 'formula' in item:
                            comp = Composition(item['formula'])
                            formula = comp.reduced_formula
                        if 'structure' in item:
                            struct = Structure.from_dict(item['structure'])
                            formula = struct.composition.reduced_formula
                        
                        if formula:
                            non_fe.add(formula)
                        if struct:
                            structures.append((formula, struct))
                    except:
                        pass
    
    print(f"当前负样本: {len(non_fe)} 个唯一化学式, {len(structures)} 个结构")
    return non_fe, structures


def fetch_polar_nonferroelectric_from_mp(known_fe, existing_non_fe, target_count=5000):
    """从MP获取极性非铁电材料"""
    print(f"\n从MP获取极性非铁电材料...")
    
    new_samples = []
    seen_formulas = set(existing_non_fe)
    
    with MPRester(MP_API_KEY) as mpr:
        # 分批查询每个空间群
        all_docs = []
        
        print("查询MP数据库（分批按空间群）...")
        for sg in tqdm(POLAR_SPACE_GROUPS, desc="查询空间群"):
            try:
                docs = mpr.materials.summary.search(
                    spacegroup_number=sg,
                    energy_above_hull=(0, 0.1),  # 相对稳定的材料
                    fields=["material_id", "formula_pretty", "structure", "symmetry", 
                            "energy_above_hull", "band_gap", "is_stable"]
                )
                all_docs.extend(docs)
            except Exception as e:
                print(f"\n  空间群 {sg} 查询失败: {e}")
                continue
        
        print(f"\n获取到 {len(all_docs)} 个极性材料")
        
        # 筛选非铁电材料
        for doc in tqdm(all_docs, desc="筛选非铁电材料"):
            try:
                formula = doc.formula_pretty
                comp = Composition(formula)
                reduced = comp.reduced_formula
                
                # 跳过已知铁电
                if reduced in known_fe:
                    continue
                
                # 跳过已有样本
                if reduced in seen_formulas:
                    continue
                
                # 获取结构
                struct = doc.structure
                if struct is None:
                    continue
                
                # 获取空间群
                sg = doc.symmetry.number if doc.symmetry else None
                
                # 添加样本
                sample = {
                    "material_id": str(doc.material_id),
                    "formula": reduced,
                    "spacegroup_number": sg,
                    "energy_above_hull": doc.energy_above_hull,
                    "band_gap": doc.band_gap,
                    "is_stable": doc.is_stable,
                    "structure": struct.as_dict()
                }
                
                new_samples.append(sample)
                seen_formulas.add(reduced)
                
                if len(new_samples) >= target_count:
                    break
                    
            except Exception as e:
                continue
    
    print(f"新获取的极性非铁电材料: {len(new_samples)} 个")
    return new_samples


def save_cleaned_dataset(known_fe, existing_structures, new_samples):
    """保存清理后的数据集"""
    root = Path(__file__).parent
    
    # 清理现有负样本（移除与正样本冲突的）
    cleaned_structures = []
    removed_count = 0
    
    for formula, struct in existing_structures:
        if formula not in known_fe:
            cleaned_structures.append((formula, struct))
        else:
            removed_count += 1
    
    print(f"\n从现有负样本中移除 {removed_count} 个与正样本冲突的结构")
    
    # 保存清理后的现有负样本
    cleaned_file = root / 'dataset_nonFE_cleaned.jsonl'
    with open(cleaned_file, 'w') as f:
        for formula, struct in cleaned_structures:
            item = {
                "formula": formula,
                "structure": struct.as_dict()
            }
            f.write(json.dumps(item) + '\n')
    print(f"保存清理后的负样本: {cleaned_file.name} ({len(cleaned_structures)} 条)")
    
    # 保存新获取的负样本
    new_file = root / 'dataset_nonFE_mp_polar.jsonl'
    with open(new_file, 'w') as f:
        for sample in new_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"保存MP极性非铁电: {new_file.name} ({len(new_samples)} 条)")
    
    # 合并为最终数据集
    combined_file = root / 'dataset_nonFE_expanded.jsonl'
    seen = set()
    count = 0
    
    with open(combined_file, 'w') as f:
        # 先写清理后的现有数据
        for formula, struct in cleaned_structures:
            if formula not in seen:
                item = {
                    "formula": formula,
                    "structure": struct.as_dict(),
                    "source": "original"
                }
                f.write(json.dumps(item) + '\n')
                seen.add(formula)
                count += 1
        
        # 再写新数据
        for sample in new_samples:
            if sample['formula'] not in seen:
                sample['source'] = 'mp_polar'
                f.write(json.dumps(sample) + '\n')
                seen.add(sample['formula'])
                count += 1
    
    print(f"保存合并数据集: {combined_file.name} ({count} 条唯一化学式)")
    
    return count


def main():
    print("="*60)
    print("扩展负样本数据集")
    print("="*60)
    
    # 加载已知铁电
    known_fe = load_known_ferroelectrics()
    
    # 加载当前负样本
    existing_non_fe, existing_structures = load_current_negative_samples()
    
    # 检查冲突
    overlap = known_fe & existing_non_fe
    print(f"\n当前冲突数量: {len(overlap)}")
    
    # 从MP获取更多极性非铁电
    new_samples = fetch_polar_nonferroelectric_from_mp(
        known_fe, 
        existing_non_fe,
        target_count=10000  # 目标获取10000个新样本
    )
    
    # 保存清理后的数据集
    total = save_cleaned_dataset(known_fe, existing_structures, new_samples)
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print(f"""
最终数据集统计:
- 正样本: {len(known_fe)} 个
- 负样本: {total} 个
- 正负比例: 1:{total/len(known_fe):.1f}

新数据集文件:
- dataset_nonFE_cleaned.jsonl (清理后的原有负样本)
- dataset_nonFE_mp_polar.jsonl (新增的MP极性非铁电)
- dataset_nonFE_expanded.jsonl (合并后的完整负样本)
""")


if __name__ == "__main__":
    main()
