"""
材料生成流水线
=============================================
GAN生成 → 逆向设计 → 合理性验证 → CSV保存

流程:
1. 使用GAN生成器生成铁电材料特征向量
2. 使用逆向设计模型将特征转换为材料描述
3. 使用合理性验证模块过滤不合理的材料
4. 保存为CSV格式

使用方法:
    python generate_materials_pipeline.py --n_samples 100 --output materials.csv --validate
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加共享模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import (
    FEATURE_DIM,
    IDX_TO_ELEMENT,
    MAX_ELEMENTS
)

# 导入验证模块
from material_validator import MaterialValidator, ValidityLevel


# ==========================================
# 1. 配置
# ==========================================
class PipelineConfig:
    # 路径
    BASE_DIR = Path(__file__).parent.parent
    GAN_MODEL_DIR = BASE_DIR / 'model_gan_v2'
    INVERSE_MODEL_DIR = BASE_DIR / 'invs_dgn_model_v2'
    OUTPUT_DIR = BASE_DIR / 'generated_materials'
    
    # 模型参数
    LATENT_DIM = 128
    HIDDEN_DIM = 256
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 模型加载
# ==========================================
def load_generator(model_path: Path, config: PipelineConfig):
    """加载GAN生成器"""
    from FE_GAN_v2 import Generator
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=FEATURE_DIM
    ).to(config.DEVICE)
    
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    return generator


def load_inverse_model(model_path: Path, config: PipelineConfig):
    """加载逆向设计模型"""
    # 导入逆向设计模型
    sys.path.insert(0, str(Path(__file__).parent.parent / 'gen'))
    from inverse_design_v6 import InverseDesignModel, Config as InverseConfig
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    model = InverseDesignModel().to(config.DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model


# ==========================================
# 3. 生成流水线
# ==========================================
class MaterialGenerationPipeline:
    """材料生成流水线"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self.generator = None
        self.inverse_model = None
    
    def load_models(self):
        """加载模型"""
        print("Loading models...")
        
        # 尝试加载GAN生成器
        gan_paths = [
            self.config.GAN_MODEL_DIR / 'gan_v2_best.pt',
            self.config.GAN_MODEL_DIR / 'gan_v2_final.pt',
        ]
        
        for path in gan_paths:
            if path.exists():
                self.generator = load_generator(path, self.config)
                print(f"  ✓ Loaded generator from {path.name}")
                break
        
        if self.generator is None:
            raise FileNotFoundError("No GAN model found. Please train the GAN first.")
        
        # 加载逆向设计模型
        inverse_paths = [
            self.config.INVERSE_MODEL_DIR / 'inverse_design_v6_best.pt',
            self.config.INVERSE_MODEL_DIR / 'inverse_design_v6_final.pt',
        ]
        
        for path in inverse_paths:
            if path.exists():
                self.inverse_model = load_inverse_model(path, self.config)
                print(f"  ✓ Loaded inverse model from {path.name}")
                break
        
        if self.inverse_model is None:
            raise FileNotFoundError("No inverse design model found. Please train it first.")
    
    def generate_features(self, n_samples: int, target_class: int = 1) -> np.ndarray:
        """使用GAN生成特征向量"""
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.LATENT_DIM, device=self.config.DEVICE)
            labels = torch.full((n_samples,), target_class, dtype=torch.long, device=self.config.DEVICE)
            features = self.generator(z, labels)
        
        return features.cpu().numpy()
    
    def features_to_materials(self, features: np.ndarray) -> list:
        """使用逆向设计模型将特征转换为材料描述"""
        materials = []
        
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.config.DEVICE)
            outputs = self.inverse_model(x)
        
        batch_size = x.size(0)
        
        for i in range(batch_size):
            # 解码晶格
            lattice = outputs['lattice'][i].cpu().numpy()
            vol_root = lattice[0] * 20.0
            ratio_ba = max(lattice[1], 0.1)
            ratio_ca = max(lattice[2], 0.1)
            alpha = lattice[3] * 180.0
            beta = lattice[4] * 180.0
            gamma = lattice[5] * 180.0
            
            a = vol_root / (ratio_ba * ratio_ca) ** (1/3)
            b = a * ratio_ba
            c = a * ratio_ca
            volume = vol_root ** 3
            
            # 解码元素
            element_logits = outputs['element_logits'][i]
            element_preds = element_logits.argmax(dim=1).cpu().numpy()
            fractions = outputs['fractions'][i].cpu().numpy()
            
            # 构建化学式
            elements = []
            formula_parts = []
            for j in range(MAX_ELEMENTS):
                if fractions[j] > 0.02:
                    el = IDX_TO_ELEMENT.get(int(element_preds[j]), 'X')
                    elements.append(el)
                    # 计算比例系数
                    coeff = fractions[j] * 10  # 归一化到合理范围
                    formula_parts.append(f"{el}{coeff:.2f}")
            
            formula = '-'.join(elements) if elements else 'Unknown'
            
            # 空间群
            sg = int(outputs['spacegroup'][i].item() * 230)
            sg = max(1, min(sg, 230))
            
            materials.append({
                'id': i + 1,
                'formula': formula,
                'elements': ','.join(elements),
                'n_elements': len(elements),
                'a': round(a, 4),
                'b': round(b, 4),
                'c': round(c, 4),
                'alpha': round(alpha, 2),
                'beta': round(beta, 2),
                'gamma': round(gamma, 2),
                'volume': round(volume, 2),
                'spacegroup': sg,
                **{f'element_{j+1}': elements[j] if j < len(elements) else '' for j in range(5)},
                **{f'fraction_{j+1}': round(fractions[j], 4) if j < len(fractions) else 0 for j in range(5)}
            })
        
        return materials
    
    def generate_and_save(self, n_samples: int, output_path: str = None, 
                          target_class: int = 1) -> pd.DataFrame:
        """生成材料并保存为CSV"""
        print(f"\n{'='*60}")
        print(f"Generating {n_samples} materials...")
        print(f"{'='*60}")
        
        # 生成特征
        print("\n1. Generating feature vectors with GAN...")
        features = self.generate_features(n_samples, target_class)
        print(f"   Generated {features.shape[0]} feature vectors (dim={features.shape[1]})")
        
        # 转换为材料描述
        print("\n2. Converting features to material descriptions...")
        materials = self.features_to_materials(features)
        print(f"   Converted to {len(materials)} material descriptions")
        
        # 创建DataFrame
        df = pd.DataFrame(materials)
        
        # 保存原始生成结果
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.config.OUTPUT_DIR / f'generated_materials_{timestamp}.csv'
        else:
            output_path = Path(output_path)
        
        df.to_csv(output_path, index=False)
        print(f"\n3. Saved raw materials to: {output_path}")
        
        # 统计信息
        print(f"\n{'='*60}")
        print("Generation Summary")
        print(f"{'='*60}")
        print(f"Total materials: {len(df)}")
        print(f"Unique formulas: {df['formula'].nunique()}")
        print(f"Elements used: {df['elements'].str.split(',').explode().nunique()}")
        print(f"Volume range: {df['volume'].min():.1f} - {df['volume'].max():.1f} Å³")
        print(f"Spacegroup range: {df['spacegroup'].min()} - {df['spacegroup'].max()}")
        
        # 最常见元素
        all_elements = df['elements'].str.split(',').explode()
        top_elements = all_elements.value_counts().head(10)
        print(f"\nTop 10 elements:")
        for el, count in top_elements.items():
            print(f"  {el}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def validate_materials(self, df: pd.DataFrame, 
                           min_level: str = 'GOOD') -> pd.DataFrame:
        """验证材料合理性"""
        print(f"\n{'='*60}")
        print("Validating Material Validity...")
        print(f"{'='*60}")
        
        validator = MaterialValidator()
        results_df = validator.validate_dataframe(df)
        
        # 统计各等级
        print("\nValidity Distribution:")
        for level in ValidityLevel:
            count = (results_df['validity_level'] == level.name).sum()
            pct = count / len(results_df) * 100
            print(f"  {level.name}: {count} ({pct:.1f}%)")
        
        # 合并结果
        merged_df = pd.concat([df, results_df.drop(['id', 'formula'], axis=1)], axis=1)
        
        # 过滤
        level_thresholds = {
            'EXCELLENT': 80,
            'GOOD': 65,
            'ACCEPTABLE': 50,
            'POOR': 35,
            'INVALID': 0
        }
        threshold = level_thresholds.get(min_level, 50)
        valid_df = merged_df[merged_df['score'] >= threshold].copy()
        
        print(f"\nFiltered to {len(valid_df)} materials (>= {min_level})")
        
        return valid_df
    
    def run(self, n_samples: int = 100, output_path: str = None, 
            validate: bool = True, min_level: str = 'GOOD'):
        """运行完整流水线"""
        self.load_models()
        df = self.generate_and_save(n_samples, output_path)
        
        if validate:
            valid_df = self.validate_materials(df, min_level)
            
            # 保存验证后的结果
            if output_path:
                validated_path = Path(output_path).with_suffix('.validated.csv')
            else:
                validated_path = self.config.OUTPUT_DIR / 'validated_materials.csv'
            
            valid_df.to_csv(validated_path, index=False)
            print(f"\n4. Saved validated materials to: {validated_path}")
            
            # 显示最佳材料
            print(f"\n{'='*60}")
            print("Top 5 Validated Materials:")
            print(f"{'='*60}")
            
            top_materials = valid_df.nlargest(5, 'score')
            for _, row in top_materials.iterrows():
                print(f"\n{row['formula']}:")
                print(f"  Score: {row['score']:.1f} ({row['validity_level']})")
                print(f"  Elements: {row['elements']}")
                print(f"  Lattice: a={row['a']:.2f}, b={row['b']:.2f}, c={row['c']:.2f}")
                print(f"  Spacegroup: {row['spacegroup']}")
            
            return valid_df
        
        return df


# ==========================================
# 4. 快速生成函数
# ==========================================
def generate_materials_quick(n_samples: int = 100, validate: bool = True) -> pd.DataFrame:
    """快速生成材料 (用于其他脚本调用)"""
    pipeline = MaterialGenerationPipeline()
    return pipeline.run(n_samples, validate=validate)


# ==========================================
# 5. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Generate ferroelectric materials')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of materials to generate')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate materials')
    parser.add_argument('--no-validate', dest='validate', action='store_false', help='Skip validation')
    parser.add_argument('--min-level', type=str, default='GOOD', 
                        choices=['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR'],
                        help='Minimum validity level for filtering')
    args = parser.parse_args()
    
    pipeline = MaterialGenerationPipeline()
    pipeline.run(args.n_samples, args.output, args.validate, args.min_level)


if __name__ == '__main__':
    main()
