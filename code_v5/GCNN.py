import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ==========================================
# 1. 数据集定义：将晶体结构转为图 (Graph)
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
    使用 pymatgen 处理周期性边界条件下的邻居查找
    """
    try:
        # 1. 重建 Pymatgen Structure 对象
        s_dict = item['structure']
        struct = Structure.from_dict(s_dict)
        
        # 2. 获取原子节点特征 (这里只用原子序数，GNN会自动学习其化学性质)
        # 减1是因为 Embedding 索引从0开始，而原子序数从1开始
        atomic_numbers = [site.specie.number - 1 for site in struct]
        x = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # 3. 构建边 (基于距离 cutoff)
        # get_all_neighbors 返回: [neighbor, distance, index, image]
        all_neighbors = struct.get_all_neighbors(r=cutoff, include_index=True)
        
        edge_indices = []
        edge_attrs = []
        
        for i, neighbors in enumerate(all_neighbors):
            for n_node in neighbors:
                # n_node[2] 是邻居的索引, n_node[1] 是距离
                j = n_node[2]
                dist = n_node[1]
                if i != j: # 避免自环
                    edge_indices.append([i, j])
                    # 边特征可以是距离的倒数 (越近权重越大)
                    edge_attrs.append([1.0 / (dist + 0.1)]) 
        
        if len(edge_indices) == 0:
            return None # 也就是孤立原子，这在晶体中很少见，过滤掉
            
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        y = torch.tensor([label], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
    except Exception as e:
        # print(f"Error parsing structure: {e}")
        return None

def load_data(pos_files, neg_file):
    dataset_list = []
    
    print("Processing Positive Samples...")
    for f_name in pos_files:
        with open(f_name, 'r') as f:
            for line in f:
                item = json.loads(line)
                graph = json_to_graph(item, 1)
                if graph: dataset_list.append(graph)

    print("Processing Negative Samples...")
    with open(neg_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            graph = json_to_graph(item, 0)
            if graph: dataset_list.append(graph)
            
    return dataset_list

# ==========================================
# 2. GNN 模型定义 (带特征提取接口)
# ==========================================
class CrystalGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=64):
        super(CrystalGNN, self).__init__()
        
        # 1. 原子 Embedding 层
        # 假设最大原子序数是 100，将原子序数映射为 dense vector
        self.node_embedding = nn.Embedding(100, embedding_dim)
        
        # 2. 图卷积层 (这里使用 GAT - Graph Attention Network，比 GCN 更强)
        self.conv1 = GATConv(embedding_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        
        # 3. 读出层 (Global Pooling)
        # 将所有原子的特征聚合成一个图级特征
        
        # 4. 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # 获取图级特征
        graph_embedding = self.get_embedding(data)
        
        # 分类
        out = self.classifier(graph_embedding)
        return torch.sigmoid(out)

    def get_embedding(self, data):
        """
        专门用于给 GAN 提取高维特征的接口
        返回: [Batch_Size, hidden_dim] 的向量
        """
        x, edge_index = data.x, data.edge_index
        
        # Embedding: Atom ID -> Vector
        x = self.node_embedding(x) 
        
        # Message Passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # Global Pooling (把原子特征聚合成晶体特征)
        # batch 向量指示了哪些节点属于哪个晶体
        x_graph = global_mean_pool(x, data.batch)
        
        return x_graph

# ==========================================
# 3. 主程序流程
# ==========================================

# 文件路径 (请确保路径正确)
pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
neg_file = 'new_data/dataset_nonFE.jsonl'

# 1. 加载并转换数据
full_data_list = load_data(pos_files, neg_file)
print(f"Total Graphs Created: {len(full_data_list)}")

# 2. 划分训练/测试集
train_data, test_data = train_test_split(full_data_list, test_size=0.2, random_state=42)

# 3. DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 4. 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrystalGNN(hidden_dim=128).to(device) # 这里 hidden_dim=128 就是你想要的高维特征维度
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 5. 训练
print(f"Start Training on {device}...")
model.train()
for epoch in range(30): # 训练 30 个 Epoch
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output.view(-1), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# 6. 评估
model.eval()
y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        prob = model(batch).view(-1)
        pred = (prob > 0.5).float()
        
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(prob.cpu().numpy())

print("\n====== GNN Classification Report ======")
print(classification_report(y_true, y_pred, target_names=['Non-FE', 'FE']))
print(f"ROC-AUC: {roc_auc_score(y_true, y_probs):.4f}")
    
# ==========================================
# 7. 如何为 GAN 提取特征 (示例)
# ==========================================
print("\n====== Feature Extraction Example for GAN ======")
sample_batch = next(iter(test_loader)).to(device)
# 获取高维特征向量 (Embedding)
latent_vectors = model.get_embedding(sample_batch)
print(f"Input Graphs Batch: {sample_batch.num_graphs}")
print(f"Extracted Feature Shape: {latent_vectors.shape}") 
print("Each crystal is now represented by a 128-dimensional dense vector.")
# 你可以将 latent_vectors.cpu().numpy() 保存下来，或者直接接入 GAN 的判别器/生成器损失函数中