import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# [修改] 新版 PyG 推荐从 loader 导入 DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
# [新增] 引入混合精度训练模块
from torch.cuda.amp import autocast, GradScaler
import time

# ==========================================
# 0. 显卡环境配置
# ==========================================
# 检查是否可用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    # [优化] 开启 CuDNN benchmark，针对输入尺寸固定的网络加速明显
    torch.backends.cudnn.benchmark = True

# ==========================================
# 1. 数据集定义
# ==========================================
class FerroelectricDataset(Dataset):
    def __init__(self, data_list):
        super(FerroelectricDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def json_to_graph(item, label, cutoff=6.0):
    """
    将单条 JSON 数据转换为 PyG 的 Data 对象
    注意：这部分是在 CPU 上运行的预处理
    新增：空间群、晶系、是否极性等结构特征
    """
    try:
        s_dict = item['structure']
        struct = Structure.from_dict(s_dict)
        
        atomic_numbers = [site.specie.number - 1 for site in struct]
        x = torch.tensor(atomic_numbers, dtype=torch.long)
        
        all_neighbors = struct.get_all_neighbors(r=cutoff, include_index=True)
        
        edge_indices = []
        edge_attrs = []
        
        for i, neighbors in enumerate(all_neighbors):
            for n_node in neighbors:
                j = n_node[2]
                dist = n_node[1]
                if i != j:
                    edge_indices.append([i, j])
                    edge_attrs.append([1.0 / (dist + 0.1)]) 
        
        if len(edge_indices) == 0:
            return None
            
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.float)
        
        # ===== 新增：提取结构特征 =====
        try:
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            spacegroup_number = sga.get_space_group_number()  # 1-230
            crystal_system = sga.get_crystal_system()  # 晶系
            is_polar = int(sga.get_point_group_symbol() in 
                          ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm'])  # 极性点群
        except:
            spacegroup_number = 1
            crystal_system = 'triclinic'
            is_polar = 0
        
        # 晶系编码 (7种晶系)
        crystal_system_map = {
            'triclinic': 0, 'monoclinic': 1, 'orthorhombic': 2,
            'tetragonal': 3, 'trigonal': 4, 'hexagonal': 5, 'cubic': 6
        }
        crystal_system_id = crystal_system_map.get(crystal_system, 0)
        
        # 晶格参数特征
        lattice = struct.lattice
        volume = lattice.volume
        density = struct.density
        num_sites = len(struct)
        avg_volume_per_atom = volume / num_sites
        
        # 晶格参数比值 (归一化)
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        
        # 构建全局特征向量 (10维)
        global_features = torch.tensor([
            spacegroup_number / 230.0,        # 归一化空间群号
            crystal_system_id / 6.0,          # 归一化晶系
            float(is_polar),                  # 是否极性点群
            min(volume, 5000) / 5000.0,       # 归一化体积 (截断)
            min(density, 10) / 10.0,          # 归一化密度
            min(num_sites, 200) / 200.0,      # 归一化原子数
            min(avg_volume_per_atom, 50) / 50.0,  # 平均原子体积
            (alpha - 60) / 60.0,              # 归一化角度偏离
            (beta - 60) / 60.0,
            (gamma - 60) / 60.0
        ], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    global_feat=global_features, spacegroup=spacegroup_number)
        
    except Exception as e:
        return None

def analyze_data_distribution(dataset_list):
    """分析数据集中的空间群、晶系等分布"""
    print("\n========== 数据分布分析 ==========")
    
    fe_spacegroups = []
    non_fe_spacegroups = []
    fe_polar = 0
    non_fe_polar = 0
    
    for data in dataset_list:
        label = data.y.item()
        sg = data.spacegroup
        is_polar = data.global_feat[2].item()  # 第3个特征是极性
        
        if label == 1:
            fe_spacegroups.append(sg)
            if is_polar > 0.5:
                fe_polar += 1
        else:
            non_fe_spacegroups.append(sg)
            if is_polar > 0.5:
                non_fe_polar += 1
    
    print(f"铁电材料 (FE): {len(fe_spacegroups)} 个")
    print(f"  - 极性点群: {fe_polar} ({fe_polar/len(fe_spacegroups)*100:.1f}%)")
    print(f"  - 常见空间群: {sorted(set(fe_spacegroups), key=fe_spacegroups.count, reverse=True)[:10]}")
    
    print(f"\n非铁电材料 (Non-FE): {len(non_fe_spacegroups)} 个")
    print(f"  - 极性点群: {non_fe_polar} ({non_fe_polar/len(non_fe_spacegroups)*100:.1f}%)")
    print(f"  - 常见空间群: {sorted(set(non_fe_spacegroups), key=non_fe_spacegroups.count, reverse=True)[:10]}")
    
    # 检查是否存在完美分离
    fe_sg_set = set(fe_spacegroups)
    non_fe_sg_set = set(non_fe_spacegroups)
    overlap = fe_sg_set & non_fe_sg_set
    print(f"\n空间群重叠: {len(overlap)} 个空间群同时出现在两类中")
    if len(overlap) < 10:
        print(f"  重叠空间群: {sorted(overlap)}")
    
    return fe_spacegroups, non_fe_spacegroups

def load_data(pos_files, neg_files):
    dataset_list = []
    print("Loading data (CPU Preprocessing)...")
    
    # 简单的计数器查看进度
    count = 0
    t0 = time.time()
    
    for f_name in pos_files:
        with open(f_name, 'r') as f:
            for line in f:
                item = json.loads(line)
                graph = json_to_graph(item, 1)
                if graph: 
                    dataset_list.append(graph)
                    count += 1
                    if count % 1000 == 0: print(f"Processed {count} graphs...", end='\r')

    # 加载多个负样本文件
    for f_name in neg_files:
        print(f"\nLoading negative samples from: {f_name}")
        with open(f_name, 'r') as f:
            for line in f:
                item = json.loads(line)
                graph = json_to_graph(item, 0)
                if graph: 
                    dataset_list.append(graph)
                    count += 1
                    if count % 1000 == 0: print(f"Processed {count} graphs...", end='\r')
            
    print(f"\nData loaded in {time.time()-t0:.2f}s")
    return dataset_list

# ==========================================
# 2. GNN 模型定义
# ==========================================
class CrystalGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=64, global_feat_dim=10):
        super(CrystalGNN, self).__init__()
        self.node_embedding = nn.Embedding(100, embedding_dim)
        
        # 多头注意力 GAT 层
        self.conv1 = GATConv(embedding_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 全局特征处理网络
        self.global_feat_net = nn.Sequential(
            nn.Linear(global_feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64)
        )
        
        # 融合后的分类器 (图嵌入 + 全局特征)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        graph_embedding = self.get_graph_embedding(data)
        
        # 处理全局特征 (PyG会将batch中的global_feat拼接，需要reshape)
        global_feat = data.global_feat.view(-1, 10)  # [batch_size, 10]
        global_embedding = self.global_feat_net(global_feat)
        
        # 融合图嵌入和全局特征
        combined = torch.cat([graph_embedding, global_embedding], dim=1)
        out = self.classifier(combined)
        return out  # 不应用 sigmoid，让 BCEWithLogitsLoss 处理

    def get_graph_embedding(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_embedding(x) 
        
        # 带残差连接和批归一化的 GAT 层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x) + x_res  # 残差连接
        
        x_res = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x) + x_res  # 残差连接
        
        x_graph = global_mean_pool(x, data.batch)
        return x_graph
    
    def get_embedding(self, data):
        """返回完整的融合嵌入向量"""
        graph_embedding = self.get_graph_embedding(data)
        global_feat = data.global_feat.view(-1, 10)
        global_embedding = self.global_feat_net(global_feat)
        return torch.cat([graph_embedding, global_embedding], dim=1)

# ==========================================
# 3. 主程序流程
# ==========================================

# 请确保文件路径正确
pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
# 负样本包含两部分：非极性非铁电 + 极性非铁电（更有挑战性）
neg_files = ['new_data/dataset_nonFE.jsonl', 'new_data/dataset_polar_non_ferroelectric_final.jsonl']

# 1. 加载数据
full_data_list = load_data(pos_files, neg_files)
print(f"Total Graphs Created: {len(full_data_list)}")

if len(full_data_list) == 0:
    print("Error: No data loaded. Check file paths.")
    exit()

# 1.5 数据分布分析
fe_sgs, non_fe_sgs = analyze_data_distribution(full_data_list)

# 2. 划分训练/测试集 (使用分层采样)
labels = [data.y.item() for data in full_data_list]
train_data, test_data = train_test_split(full_data_list, test_size=0.2, random_state=42, stratify=labels)

# 3. DataLoader [优化点]
# pin_memory=True: 加速数据从 CPU 内存复制到 GPU 显存
# num_workers: Linux 下可以设置为 4 或 8 来并行加载数据（虽然这里数据已经在内存里了，影响不大，但如果是读取磁盘则必须设）
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

# 4. 初始化模型
model = CrystalGNN(hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 计算权重：负样本数 / 正样本数 ≈ 984 / 149 ≈ 6.6
# 这会让模型由于"漏掉一个铁电"而受到的惩罚是"漏掉一个非铁电"的 6.6 倍
pos_weight = torch.tensor([6.0]).to(device) 
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# [新增] 初始化混合精度 Scaler
scaler = GradScaler()

# 5. 训练循环
print(f"Start Training on {device} with Mixed Precision...")
print(f"Model device: {next(model.parameters()).device}")
if device.type == 'cuda':
    print(f"GPU Memory before training: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
model.train()

for epoch in range(30):
    total_loss = 0
    start_time = time.time()
    
    for batch in train_loader:
        # 非阻塞传输数据到 GPU
        batch = batch.to(device, non_blocking=True)
        
        # 在第一个 epoch 的第一个 batch 验证数据确实在 GPU 上
        if epoch == 0 and total_loss == 0:
            print(f"Batch data device: {batch.x.device}")
            if device.type == 'cuda':
                print(f"GPU Memory after loading batch: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        
        optimizer.zero_grad()
        
        # [核心优化] 开启自动混合精度上下文
        with autocast():
            output = model(batch)
            loss = criterion(output.view(-1), batch.y)
        
        # [核心优化] 使用 scaler 进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    epoch_time = time.time() - start_time
    
    if (epoch+1) % 5 == 0:
        info = f"Epoch {epoch+1:02d} | Loss: {total_loss / len(train_loader):.4f} | Time: {epoch_time:.2f}s"
        if device.type == 'cuda':
            info += f" | GPU Mem: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB"
        print(info)

# 6. 详细评估与分析
model.eval()
y_true, y_pred, y_probs = [], [], []
test_spacegroups = []
test_global_feats = []

print("\nEvaluating...")
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device, non_blocking=True)
        
        with autocast():
            logits = model(batch).view(-1)
            prob = torch.sigmoid(logits)
            
        pred = (prob > 0.5).float()
        
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(prob.cpu().numpy())
        
        # 收集空间群和全局特征用于分析
        for i in range(len(batch.y)):
            test_spacegroups.append(batch.spacegroup[i] if hasattr(batch, 'spacegroup') else 0)
            test_global_feats.append(batch.global_feat.view(-1, 10)[i].cpu().numpy())

print("\n" + "="*60)
print("====== GNN 分类详细报告 ======")
print("="*60)

# 基础分类报告
print("\n【分类报告】")
print(classification_report(y_true, y_pred, target_names=['Non-FE', 'FE']))

try:
    auc = roc_auc_score(y_true, y_probs)
    print(f"ROC-AUC: {auc:.4f}")
except:
    print("ROC-AUC Error")
    auc = 0

# 混淆矩阵
print("\n【混淆矩阵】")
cm = confusion_matrix(y_true, y_pred)
print(f"              Predicted")
print(f"              Non-FE    FE")
print(f"Actual Non-FE   {cm[0][0]:4d}   {cm[0][1]:4d}")
print(f"       FE       {cm[1][0]:4d}   {cm[1][1]:4d}")

# 错误案例分析
print("\n【错误案例分析】")
y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
y_probs_np = np.array(y_probs)
test_global_feats_np = np.array(test_global_feats)

false_positives = np.where((y_true_np == 0) & (y_pred_np == 1))[0]
false_negatives = np.where((y_true_np == 1) & (y_pred_np == 0))[0]

print(f"假阳性 (预测为FE实际为Non-FE): {len(false_positives)} 个")
if len(false_positives) > 0 and len(false_positives) <= 10:
    for idx in false_positives[:10]:
        sg = int(test_global_feats_np[idx][0] * 230)
        is_polar = test_global_feats_np[idx][2]
        prob = y_probs_np[idx]
        print(f"  样本 {idx}: 空间群={sg}, 极性={is_polar:.2f}, 预测概率={prob:.4f}")

print(f"\n假阴性 (预测为Non-FE实际为FE): {len(false_negatives)} 个")
if len(false_negatives) > 0 and len(false_negatives) <= 10:
    for idx in false_negatives[:10]:
        sg = int(test_global_feats_np[idx][0] * 230)
        is_polar = test_global_feats_np[idx][2]
        prob = y_probs_np[idx]
        print(f"  样本 {idx}: 空间群={sg}, 极性={is_polar:.2f}, 预测概率={prob:.4f}")

# 预测概率分布分析
print("\n【预测概率分布】")
fe_probs = y_probs_np[y_true_np == 1]
non_fe_probs = y_probs_np[y_true_np == 0]

print(f"FE 样本预测概率: min={fe_probs.min():.4f}, mean={fe_probs.mean():.4f}, max={fe_probs.max():.4f}")
print(f"Non-FE 样本预测概率: min={non_fe_probs.min():.4f}, mean={non_fe_probs.mean():.4f}, max={non_fe_probs.max():.4f}")

# 全局特征统计
print("\n【测试集全局特征统计】")
fe_feats = test_global_feats_np[y_true_np == 1]
non_fe_feats = test_global_feats_np[y_true_np == 0]

feature_names = ['空间群', '晶系', '极性', '体积', '密度', '原子数', '平均原子体积', 'α角偏离', 'β角偏离', 'γ角偏离']
print(f"\n{'特征':<12} {'FE均值':<10} {'Non-FE均值':<12} {'差异':<10}")
print("-" * 50)
for i, name in enumerate(feature_names):
    fe_mean = fe_feats[:, i].mean()
    non_fe_mean = non_fe_feats[:, i].mean()
    diff = abs(fe_mean - non_fe_mean)
    print(f"{name:<12} {fe_mean:>8.4f}   {non_fe_mean:>10.4f}   {diff:>8.4f}")

# 检查是否存在完美线性分离
print("\n【数据泄露检测】")
print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
print(f"训练准确率: {(1 - total_loss / len(train_loader) / criterion.pos_weight.item()):.4f} (近似)")
print(f"测试准确率: {(y_true_np == y_pred_np).mean():.4f}")

if (y_true_np == y_pred_np).mean() == 1.0:
    print("⚠️  警告: 测试集准确率 100%，可能存在以下问题:")
    print("   1. 数据泄露 (训练集和测试集信息重叠)")
    print("   2. 某个特征与标签完美相关 (如极性点群)")
    print("   3. 数据集过于简单")
    
    # 检查极性特征的分离度
    fe_polar_ratio = (fe_feats[:, 2] > 0.5).mean()
    non_fe_polar_ratio = (non_fe_feats[:, 2] > 0.5).mean()
    print(f"\n   极性点群分布: FE={fe_polar_ratio:.2%}, Non-FE={non_fe_polar_ratio:.2%}")
    if fe_polar_ratio > 0.95 and non_fe_polar_ratio < 0.05:
        print("   ⚠️  极性特征几乎完美分离两类！这可能是主要原因。")

# 7. 提取特征示例
print("\n【特征提取】")
sample_batch = next(iter(test_loader)).to(device)
with torch.no_grad():
    latent_vectors = model.get_embedding(sample_batch)
print(f"提取的嵌入向量维度: {latent_vectors.shape}")
print(f"嵌入向量统计: mean={latent_vectors.mean():.4f}, std={latent_vectors.std():.4f}")

print("\n" + "="*60)