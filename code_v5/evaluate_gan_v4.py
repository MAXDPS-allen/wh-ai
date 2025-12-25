"""
GAN v4 生成质量评估
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from feature_engineering import UnifiedFeatureExtractor, FEATURE_DIM, FEATURE_NAMES

DATA_DIR = Path(__file__).parent.parent / 'new_data'
MODEL_DIR = Path(__file__).parent.parent / 'model_gan_v4'
REPORT_DIR = Path(__file__).parent.parent / 'reports_gan_v4'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_real_data():
    extractor = UnifiedFeatureExtractor()
    fe_features = []
    nonfe_features = []
    
    fe_files = [
        DATA_DIR / 'dataset_original_ferroelectric.jsonl',
        DATA_DIR / 'dataset_known_FE_rest.jsonl',
    ]
    nonfe_files = [
        DATA_DIR / 'dataset_nonFE.jsonl',
        DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl',
    ]
    
    for fpath in fe_files:
        if fpath.exists():
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        struct = item.get('structure')
                        if struct:
                            sg = item.get('spacegroup_number', None)
                            feat = extractor.extract_from_structure_dict(struct, sg)
                            if np.sum(np.abs(feat)) > 0:
                                fe_features.append(feat)
                    except:
                        continue
    
    for fpath in nonfe_files:
        if fpath.exists():
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        struct = item.get('structure')
                        if struct:
                            sg = item.get('spacegroup_number', None)
                            feat = extractor.extract_from_structure_dict(struct, sg)
                            if np.sum(np.abs(feat)) > 0:
                                nonfe_features.append(feat)
                    except:
                        continue
    
    return np.array(fe_features, dtype=np.float32), np.array(nonfe_features, dtype=np.float32)


def load_generator():
    class Generator(nn.Module):
        def __init__(self, latent_dim=128, hidden_dim=768, output_dim=64):
            super().__init__()
            self.cond_embed = nn.Embedding(2, 64)
            self.net = nn.Sequential(
                nn.Linear(latent_dim + 64, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Sigmoid()
            )
        
        def forward(self, z, labels):
            cond = self.cond_embed(labels)
            x = torch.cat([z, cond], dim=1)
            return self.net(x)
    
    model_path = MODEL_DIR / 'generator_v4_best.pt'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    G = Generator(128, 768, 64).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    G.eval()
    return G


def generate_samples(G, n_samples, label=1):
    with torch.no_grad():
        z = torch.randn(n_samples, 128, device=DEVICE)
        labels = torch.full((n_samples,), label, dtype=torch.long, device=DEVICE)
        samples = G(z, labels).cpu().numpy()
    return samples


def evaluate_distribution(real, generated, name="FE"):
    print(f"\n{'='*60}")
    print(f"分布评估: {name}")
    print(f"{'='*60}")
    
    n_features = min(20, real.shape[1])
    
    ks_stats = []
    ks_passes = 0
    for i in range(n_features):
        stat, pval = stats.ks_2samp(real[:, i], generated[:, i])
        ks_stats.append(stat)
        if pval > 0.05:
            ks_passes += 1
    
    print(f"\nKS检验 (前{n_features}个特征):")
    print(f"  平均KS统计量: {np.mean(ks_stats):.4f}")
    print(f"  通过测试的特征数: {ks_passes}/{n_features} ({100*ks_passes/n_features:.1f}%)")
    
    real_mean = real.mean(0)
    gen_mean = generated.mean(0)
    real_std = real.std(0)
    gen_std = generated.std(0)
    
    mean_diff = np.abs(real_mean - gen_mean).mean()
    std_diff = np.abs(real_std - gen_std).mean()
    corr = np.corrcoef(real_mean, gen_mean)[0, 1]
    
    print(f"\n统计量匹配:")
    print(f"  均值差异: {mean_diff:.4f}")
    print(f"  标准差差异: {std_diff:.4f}")
    print(f"  均值相关系数: {corr:.4f}")
    
    return {
        'ks_mean': np.mean(ks_stats),
        'ks_passes': ks_passes,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'correlation': corr
    }


def tsne_visualization(real_fe, real_nonfe, gen_fe, gen_nonfe, save_path):
    print("\n生成t-SNE可视化...")
    
    n_samples = 300
    data = np.vstack([
        real_fe[:n_samples],
        real_nonfe[:n_samples],
        gen_fe[:n_samples],
        gen_nonfe[:n_samples]
    ])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(data)
    
    real_fe_center = embedded[:n_samples].mean(0)
    real_nonfe_center = embedded[n_samples:2*n_samples].mean(0)
    gen_fe_center = embedded[2*n_samples:3*n_samples].mean(0)
    gen_nonfe_center = embedded[3*n_samples:].mean(0)
    
    dist_fe = np.linalg.norm(real_fe_center - gen_fe_center)
    dist_nonfe = np.linalg.norm(real_nonfe_center - gen_nonfe_center)
    
    print(f"\nt-SNE类中心距离:")
    print(f"  Real FE vs Gen FE: {dist_fe:.4f}")
    print(f"  Real Non-FE vs Gen Non-FE: {dist_nonfe:.4f}")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['blue', 'red', 'cyan', 'orange']
    labels_list = ['Real FE', 'Real Non-FE', 'Gen FE', 'Gen Non-FE']
    
    for i, color in enumerate(colors):
        start_idx = i * n_samples
        end_idx = start_idx + n_samples
        ax.scatter(embedded[start_idx:end_idx, 0], embedded[start_idx:end_idx, 1], 
                   c=color, alpha=0.5, s=30, label=labels_list[i])
    
    ax.scatter(*real_fe_center, c='blue', s=200, marker='*', edgecolor='black')
    ax.scatter(*real_nonfe_center, c='red', s=200, marker='*', edgecolor='black')
    ax.scatter(*gen_fe_center, c='cyan', s=200, marker='X', edgecolor='black')
    ax.scatter(*gen_nonfe_center, c='orange', s=200, marker='X', edgecolor='black')
    
    ax.set_title(f't-SNE: Real vs Generated (GAN v4)\n'
                 f'FE Distance: {dist_fe:.2f}, Non-FE Distance: {dist_nonfe:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ t-SNE可视化已保存: {save_path}")
    
    return dist_fe, dist_nonfe


def main():
    print("="*60)
    print("GAN v4 生成质量评估")
    print("="*60)
    
    REPORT_DIR.mkdir(exist_ok=True)
    
    print("\n加载真实数据...")
    real_fe, real_nonfe = load_real_data()
    print(f"  FE样本: {len(real_fe)}")
    print(f"  Non-FE样本: {len(real_nonfe)}")
    
    print("\n加载生成器...")
    G = load_generator()
    if G is None:
        return
    
    n_gen = 1000
    print(f"\n生成{n_gen}个样本...")
    gen_fe = generate_samples(G, n_gen, label=1)
    gen_nonfe = generate_samples(G, n_gen, label=0)
    
    fe_metrics = evaluate_distribution(real_fe, gen_fe, "铁电材料(FE)")
    nonfe_metrics = evaluate_distribution(real_nonfe, gen_nonfe, "非铁电材料(Non-FE)")
    
    dist_fe, dist_nonfe = tsne_visualization(
        real_fe, real_nonfe, gen_fe, gen_nonfe,
        REPORT_DIR / 'gan_v4_tsne.png'
    )
    
    print("\n" + "="*60)
    print("评估总结")
    print("="*60)
    print(f"\n铁电材料(FE)生成质量:")
    print(f"  - KS检验通过率: {fe_metrics['ks_passes']}/20 ({100*fe_metrics['ks_passes']/20:.1f}%)")
    print(f"  - 均值相关系数: {fe_metrics['correlation']:.4f}")
    print(f"  - t-SNE中心距离: {dist_fe:.4f}")
    
    print(f"\n非铁电材料(Non-FE)生成质量:")
    print(f"  - KS检验通过率: {nonfe_metrics['ks_passes']}/20 ({100*nonfe_metrics['ks_passes']/20:.1f}%)")
    print(f"  - 均值相关系数: {nonfe_metrics['correlation']:.4f}")
    print(f"  - t-SNE中心距离: {dist_nonfe:.4f}")
    
    print("\n" + "-"*60)
    print("与之前版本对比:")
    print("-"*60)
    print(f"  GAN v2 FE t-SNE距离: 28.57")
    print(f"  GAN v3 FE t-SNE距离: 4.44")
    print(f"  GAN v4 FE t-SNE距离: {dist_fe:.2f}")


if __name__ == '__main__':
    main()
