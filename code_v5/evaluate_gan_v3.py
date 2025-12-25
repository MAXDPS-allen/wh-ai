"""
GAN v3 生成质量评估
==================
评估生成样本与真实样本的分布匹配程度
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import UnifiedFeatureExtractor, FEATURE_DIM, FEATURE_NAMES

# 配置
DATA_DIR = Path(__file__).parent.parent / 'new_data'
MODEL_DIR = Path(__file__).parent.parent / 'model_gan_v3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_real_data():
    """加载真实数据"""
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
    """加载GAN生成器"""
    import torch.nn as nn
    
    class Generator(nn.Module):
        def __init__(self, latent_dim=128, hidden_dim=512, output_dim=64):
            super().__init__()
            self.cond_embed = nn.Embedding(2, 32)
            self.net = nn.Sequential(
                nn.Linear(latent_dim + 32, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Sigmoid()
            )
        
        def forward(self, z, labels):
            cond = self.cond_embed(labels)
            x = torch.cat([z, cond], dim=1)
            return self.net(x)
    
    model_path = MODEL_DIR / 'generator_v3_best.pt'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    G = Generator(128, 512, 64).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()
    return G


def generate_samples(G, n_samples, label=1):
    """生成样本"""
    with torch.no_grad():
        z = torch.randn(n_samples, 128, device=DEVICE)
        labels = torch.full((n_samples,), label, dtype=torch.long, device=DEVICE)
        samples = G(z, labels).cpu().numpy()
    return samples


def evaluate_distribution(real, generated, name="FE"):
    """评估分布匹配程度"""
    print(f"\n{'='*60}")
    print(f"分布评估: {name}")
    print(f"{'='*60}")
    
    n_features = min(20, real.shape[1])
    
    # 1. KS检验
    ks_stats = []
    ks_passes = 0
    for i in range(n_features):
        stat, pval = stats.ks_2samp(real[:, i], generated[:, i])
        ks_stats.append(stat)
        if pval > 0.05:  # 分布相似
            ks_passes += 1
    
    print(f"\nKS检验 (前{n_features}个特征):")
    print(f"  平均KS统计量: {np.mean(ks_stats):.4f}")
    print(f"  通过测试的特征数: {ks_passes}/{n_features} ({100*ks_passes/n_features:.1f}%)")
    
    # 2. 均值和方差匹配
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
    
    # 3. 逐特征分析
    print(f"\n前10个特征详细对比:")
    print(f"{'特征':>8} {'真实均值':>10} {'生成均值':>10} {'差异':>10} {'真实std':>10} {'生成std':>10}")
    for i in range(min(10, n_features)):
        fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"F{i}"
        print(f"{fname[:8]:>8} {real_mean[i]:>10.4f} {gen_mean[i]:>10.4f} "
              f"{abs(real_mean[i]-gen_mean[i]):>10.4f} {real_std[i]:>10.4f} {gen_std[i]:>10.4f}")
    
    return {
        'ks_mean': np.mean(ks_stats),
        'ks_passes': ks_passes,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'correlation': corr
    }


def tsne_visualization(real_fe, real_nonfe, gen_fe, gen_nonfe, save_path):
    """t-SNE可视化"""
    print("\n生成t-SNE可视化...")
    
    n_samples = 300
    data = np.vstack([
        real_fe[:n_samples],
        real_nonfe[:n_samples],
        gen_fe[:n_samples],
        gen_nonfe[:n_samples]
    ])
    
    labels = (['Real FE'] * min(n_samples, len(real_fe)) + 
              ['Real Non-FE'] * min(n_samples, len(real_nonfe)) + 
              ['Gen FE'] * min(n_samples, len(gen_fe)) + 
              ['Gen Non-FE'] * min(n_samples, len(gen_nonfe)))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(data)
    
    # 计算类中心距离
    real_fe_center = embedded[:n_samples].mean(0)
    real_nonfe_center = embedded[n_samples:2*n_samples].mean(0)
    gen_fe_center = embedded[2*n_samples:3*n_samples].mean(0)
    gen_nonfe_center = embedded[3*n_samples:].mean(0)
    
    dist_fe = np.linalg.norm(real_fe_center - gen_fe_center)
    dist_nonfe = np.linalg.norm(real_nonfe_center - gen_nonfe_center)
    
    print(f"\nt-SNE类中心距离:")
    print(f"  Real FE vs Gen FE: {dist_fe:.4f}")
    print(f"  Real Non-FE vs Gen Non-FE: {dist_nonfe:.4f}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {'Real FE': 'blue', 'Real Non-FE': 'red', 
              'Gen FE': 'cyan', 'Gen Non-FE': 'orange'}
    
    for i, (x, y) in enumerate(embedded):
        ax.scatter(x, y, c=colors[labels[i]], alpha=0.5, s=30)
    
    # 绘制中心点
    ax.scatter(*real_fe_center, c='blue', s=200, marker='*', edgecolor='black', label='Real FE Center')
    ax.scatter(*real_nonfe_center, c='red', s=200, marker='*', edgecolor='black', label='Real Non-FE Center')
    ax.scatter(*gen_fe_center, c='cyan', s=200, marker='X', edgecolor='black', label='Gen FE Center')
    ax.scatter(*gen_nonfe_center, c='orange', s=200, marker='X', edgecolor='black', label='Gen Non-FE Center')
    
    ax.set_title(f't-SNE: Real vs Generated Distributions\n'
                 f'FE Center Distance: {dist_fe:.2f}, Non-FE Distance: {dist_nonfe:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ t-SNE可视化已保存: {save_path}")
    
    return dist_fe, dist_nonfe


def feature_distribution_plots(real, generated, name, save_path):
    """特征分布对比图"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i, ax in enumerate(axes.flat):
        if i >= 20:
            break
        ax.hist(real[:, i], bins=30, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(generated[:, i], bins=30, alpha=0.5, label='Generated', density=True, color='orange')
        fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"Feature {i}"
        ax.set_title(fname[:15], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} Feature Distributions: Real vs Generated', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ 特征分布图已保存: {save_path}")


def main():
    print("="*60)
    print("GAN v3 生成质量评估")
    print("="*60)
    
    # 加载数据
    print("\n加载真实数据...")
    real_fe, real_nonfe = load_real_data()
    print(f"  FE样本: {len(real_fe)}")
    print(f"  Non-FE样本: {len(real_nonfe)}")
    
    # 加载生成器
    print("\n加载生成器...")
    G = load_generator()
    if G is None:
        print("无法加载生成器!")
        return
    
    # 生成样本
    n_gen = 1000
    print(f"\n生成{n_gen}个样本...")
    gen_fe = generate_samples(G, n_gen, label=1)
    gen_nonfe = generate_samples(G, n_gen, label=0)
    
    # 评估FE分布
    fe_metrics = evaluate_distribution(real_fe, gen_fe, "铁电材料(FE)")
    
    # 评估Non-FE分布
    nonfe_metrics = evaluate_distribution(real_nonfe, gen_nonfe, "非铁电材料(Non-FE)")
    
    # t-SNE可视化
    report_dir = MODEL_DIR.parent / 'reports_gan_v3'
    report_dir.mkdir(exist_ok=True)
    
    dist_fe, dist_nonfe = tsne_visualization(
        real_fe, real_nonfe, gen_fe, gen_nonfe,
        report_dir / 'gan_v3_tsne.png'
    )
    
    # 特征分布图
    feature_distribution_plots(real_fe, gen_fe, "FE", report_dir / 'gan_v3_fe_distributions.png')
    feature_distribution_plots(real_nonfe, gen_nonfe, "Non-FE", report_dir / 'gan_v3_nonfe_distributions.png')
    
    # 总结
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
    
    # 与GAN v2对比
    print("\n" + "-"*60)
    print("与GAN v2对比:")
    print("-"*60)
    print(f"  GAN v2 FE t-SNE距离: 28.57 -> GAN v3: {dist_fe:.2f}")
    print(f"  GAN v2 KS通过率: 0% -> GAN v3 FE: {100*fe_metrics['ks_passes']/20:.1f}%")
    
    improvement = (28.57 - dist_fe) / 28.57 * 100
    print(f"\n  ✓ t-SNE距离改进: {improvement:.1f}%")


if __name__ == '__main__':
    main()
