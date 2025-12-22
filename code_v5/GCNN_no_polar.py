"""
移除极性特征的版本 - 测试模型真实判别力
"""
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

class FerroelectricDataset(Dataset):
    def __init__(self, data_list):
        super(FerroelectricDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def json_to_graph(item, label, cutoff=6.0):
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
        
        # ===== 移除极性特征，只保留其他结构特征 =====
        try:
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            spacegroup_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
        except:
            spacegroup_number = 1
            crystal_system = 'triclinic'
        
        crystal_system_map = {
            'triclinic': 0, 'monoclinic': 1, 'orthorhombic': 2,
            'tetragonal': 3, 'trigonal': 4, 'hexagonal': 5, 'cubic': 6
        }
        crystal_system_id = crystal_system_map.get(crystal_system, 0)
        
        lattice = struct.lattice
        volume = lattice.volume
        density = struct.density
        num_sites = len(struct)
        avg_volume_per_atom = volume / num_sites
        
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        
        # 9维全局特征 (移除了极性判断)
        global_features = torch.tensor([
            spacegroup_number / 230.0,
            crystal_system_id / 6.0,
            # 移除极性特征
            min(volume, 5000) / 5000.0,
            min(density, 10) / 10.0,
            min(num_sites, 200) / 200.0,
            min(avg_volume_per_atom, 50) / 50.0,
            (alpha - 60) / 60.0,
            (beta - 60) / 60.0,
            (gamma - 60) / 60.0
        ], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    global_feat=global_features, spacegroup=spacegroup_number)
        
    except Exception as e:
        return None

def load_data(pos_files, neg_file):
    dataset_list = []
    print("Loading data...")
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

    with open(neg_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            graph = json_to_graph(item, 0)
            if graph: 
                dataset_list.append(graph)
                count += 1
            
    print(f"Data loaded in {time.time()-t0:.2f}s, Total: {len(dataset_list)}")
    return dataset_list

class CrystalGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=64, global_feat_dim=9):
        super(CrystalGNN, self).__init__()
        self.node_embedding = nn.Embedding(100, embedding_dim)
        
        self.conv1 = GATConv(embedding_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.global_feat_net = nn.Sequential(
            nn.Linear(global_feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64)
        )
        
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
        global_feat = data.global_feat.view(-1, 9)  # 9维特征
        global_embedding = self.global_feat_net(global_feat)
        combined = torch.cat([graph_embedding, global_embedding], dim=1)
        out = self.classifier(combined)
        return out

    def get_graph_embedding(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_embedding(x) 
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x) + x_res
        
        x_res = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x) + x_res
        
        x_graph = global_mean_pool(x, data.batch)
        return x_graph

# 主程序
pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
neg_file = 'new_data/dataset_nonFE.jsonl'

full_data_list = load_data(pos_files, neg_file)
labels = [data.y.item() for data in full_data_list]
train_data, test_data = train_test_split(full_data_list, test_size=0.2, random_state=42, stratify=labels)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

model = CrystalGNN(hidden_dim=128, global_feat_dim=9).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
pos_weight = torch.tensor([6.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("\n训练开始 (无极性特征版本)...")
model.train()
for epoch in range(30):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output.view(-1), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

# 评估
model.eval()
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device, non_blocking=True)
        logits = model(batch).view(-1)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(prob.cpu().numpy())

print("\n" + "="*60)
print("====== 无极性特征模型评估 ======")
print("="*60)
print(classification_report(y_true, y_pred, target_names=['Non-FE', 'FE']))
print(f"ROC-AUC: {roc_auc_score(y_true, y_probs):.4f}")

cm = confusion_matrix(y_true, y_pred)
print("\n混淆矩阵:")
print(f"              Predicted")
print(f"              Non-FE    FE")
print(f"Actual Non-FE   {cm[0][0]:4d}   {cm[0][1]:4d}")
print(f"       FE       {cm[1][0]:4d}   {cm[1][1]:4d}")
print("\n这个结果才能反映模型的真实判别力！")
