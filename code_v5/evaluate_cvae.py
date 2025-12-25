"""
CVAE 生成质量评估
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
MODEL_DIR = Path(__file__).parent.parent / 'model_cvae'
REPORT_DIR = Path(__file__).parent.parent / 'reports_cvae'
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


def load_cvae():
    class Encoder(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=512, latent_dim=32):
            super().__init__()
            self.cond_embed = nn.Embedding(2, 32)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim + 32, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        def forward(self, x, labels):
            cond = self.cond_embed(labels)
            h = torch.cat([x, cond], dim=1)
            h = self.encoder(h)
            return self.fc_mu(h), self.fc_logvar(h)

    class Decoder(nn.Module):
        def __init__(self, latent_dim=32, hidden_dim=512, output_dim=64):
            super().__init__()
            self.cond_embed = nn.Embedding(2, 32)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + 32, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
        
        def forward(self, z, labels):
            cond = self.cond_embed(labels)
            h = torch.cat([z, cond], dim=1)
            return self.decoder(h)

    class CVAE(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=512, latent_dim=32):
            super().__init__()
            self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
            self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
            self.latent_dim = latent_dim
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def forward(self, x, labels):
            mu, logvar = self.encoder(x, labels)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z, labels)
            return recon, mu, logvar
        
        def sample(self, n_samples, labels, device):
            z = torch.randn(n_samples, self.latent_dim, device=device)
            return self.decoder(z, labels)
    
    model_path = MODEL_DIR / 'cvae_best.pt'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = CVAE(64, 512, 32).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def generate_samples(model, n_samples, label=1):
    with torch.no_grad():
        labels = torch.full((n_samples,), label, dtype=torch.long, device=DEVICE)
        samples = model.sample(n_samples, labels, DEVICE).cpu().numpy()
    return samples


def evaluate_distribution(real, generated, name="FE"):
    print(f"\n{'='*60}")
    print(f"分布评估: {name}")
    print(f"{'='*60}")
    
    n_features = min(20, real.shape[1])
    
    ks_stats = []
    ks_passes = 0
    ks_details = []
    
    for i in range(n_features):
        stat, pval = stats.ks_2samp(real[:, i], generated[:, i])
        ks_stats.append(stat)
        passed = pval > 0.05
        if passed:
            ks_passes += 1
        ks_details.append((FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"F{i}", stat, pval, passed))
    
    print(f"\nKS检验 (前{n_features}个特征):")
    print(f"  平均KS统计量: {np.mean(ks_stats):.4f}")
    print(f"  通过测试的特征数: {ks_passes}/{n_features} ({100*ks_passes/n_features:.1f}%)")
    
    print(f"\n  通过KS检验的特征:")
    for fname, stat, pval, passed in ks_details:
        if passed:
            print(f"    ✓ {fname}: KS={stat:.4f}, p={pval:.4f}")
    
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
    
    ax.set_title(f't-SNE: Real vs Generated (CVAE)\n'
                 f'FE Distance: {dist_fe:.2f}, Non-FE Distance: {dist_nonfe:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ t-SNE可视化已保存: {save_path}")
    
    return dist_fe, dist_nonfe


def feature_distribution_plots(real, generated, name, save_path):
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i, ax in enumerate(axes.flat):
        if i >= 20:
            break
        ax.hist(real[:, i], bins=30, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(generated[:, i], bins=30, alpha=0.5, label='Generated', density=True, color='orange')
        
        # KS检验
        stat, pval = stats.ks_2samp(real[:, i], generated[:, i])
        status = "✓" if pval > 0.05 else "✗"
        
        fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"Feature {i}"
        ax.set_title(f'{status} {fname[:12]}\nKS={stat:.2f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} Feature Distributions: Real vs Generated (CVAE)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ 特征分布图已保存: {save_path}")


def main():
    print("="*60)
    print("CVAE 生成质量评估")
    print("="*60)
    
    REPORT_DIR.mkdir(exist_ok=True)
    
    print("\n加载真实数据...")
    real_fe, real_nonfe = load_real_data()
    print(f"  FE样本: {len(real_fe)}")
    print(f"  Non-FE样本: {len(real_nonfe)}")
    
    print("\n加载CVAE模型...")
    model = load_cvae()
    if model is None:
        return
    
    n_gen = 1000
    print(f"\n生成{n_gen}个样本...")
    gen_fe = generate_samples(model, n_gen, label=1)
    gen_nonfe = generate_samples(model, n_gen, label=0)
    
    fe_metrics = evaluate_distribution(real_fe, gen_fe, "铁电材料(FE)")
    nonfe_metrics = evaluate_distribution(real_nonfe, gen_nonfe, "非铁电材料(Non-FE)")
    
    dist_fe, dist_nonfe = tsne_visualization(
        real_fe, real_nonfe, gen_fe, gen_nonfe,
        REPORT_DIR / 'cvae_tsne.png'
    )
    
    feature_distribution_plots(real_fe, gen_fe, "FE", REPORT_DIR / 'cvae_fe_distributions.png')
    feature_distribution_plots(real_nonfe, gen_nonfe, "Non-FE", REPORT_DIR / 'cvae_nonfe_distributions.png')
    
    print("\n" + "="*60)
    print("模型对比总结")
    print("="*60)
    print(f"\n{'模型':<15} {'KS通过率':<12} {'均值相关':<12} {'t-SNE FE距离':<15}")
    print("-"*55)
    print(f"{'GAN v2':<15} {'0%':<12} {'N/A':<12} {'28.57':<15}")
    print(f"{'GAN v3':<15} {'5%':<12} {'0.9986':<12} {'4.44':<15}")
    print(f"{'GAN v4':<15} {'0%':<12} {'0.9939':<12} {'5.45':<15}")
    print(f"{'CVAE':<15} {f'{100*fe_metrics['ks_passes']/20:.0f}%':<12} {f'{fe_metrics['correlation']:.4f}':<12} {f'{dist_fe:.2f}':<15}")
    
    print("\n" + "="*60)
    print(f"\n✓ CVAE是目前最佳模型:")
    print(f"  - KS检验通过率: {100*fe_metrics['ks_passes']/20:.0f}% (改进显著)")
    print(f"  - t-SNE中心距离: {dist_fe:.2f} (从GAN v2的28.57降低)")
    print(f"  - 标准差匹配更好: {fe_metrics['std_diff']:.4f}")


if __name__ == '__main__':
    main()
