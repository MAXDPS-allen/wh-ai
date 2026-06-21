#!/usr/bin/env python3
"""
几何感知图神经网络回归模型 (生产级)
=====================================================================
CGCNN 风格的晶体图卷积网络, 直接从原子结构 (节点=原子, 边=近邻+距离)
预测铁电关键物性。基线 (train.py) 证明组分/晶格统计无法捕捉自发极化与
切换能垒 —— 这些量由原子位移/几何决定, 必须用几何感知模型。

设计:
  - 节点嵌入: 原子序数 embedding
  - 边特征: 距离高斯展开 (RBF), 旋转/平移不变 (Ps 模长、能垒、带隙均为标量不变量)
  - 多任务输出头: Ps / dw_depth / path_barrier / gap_polar + switchable 分类
  - 不确定性: 每个回归头额外预测 log-variance (异方差), 用于主动学习排序

依赖: torch (在 GPU 节点 g1-g9 上跑)。纯 PyTorch, 无 torch_geometric 依赖。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):
    """边距离的高斯径向基展开。"""
    def __init__(self, n_rbf: int = 64, cutoff: float = 6.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, cutoff, n_rbf))
        self.gamma = 1.0 / (self.centers[1] - self.centers[0]) ** 2

    def forward(self, d):  # d: (E,)
        return torch.exp(-self.gamma * (d[:, None] - self.centers[None, :]) ** 2)


class CGConv(nn.Module):
    """CGCNN 卷积层: 门控聚合近邻信息。"""
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.lin_f = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.lin_s = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        z = torch.cat([x[src], x[dst], edge_attr], dim=1)
        msg = torch.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        return x + self.bn(agg)


class FerroPropertyGNN(nn.Module):
    REG_TARGETS = ["Ps", "dw_depth", "path_barrier", "gap_polar"]

    def __init__(self, node_dim=128, edge_dim=64, n_conv=4, max_z=100, cutoff=6.0):
        super().__init__()
        self.embed = nn.Embedding(max_z + 1, node_dim)
        self.rbf = GaussianRBF(edge_dim, cutoff)
        self.convs = nn.ModuleList([CGConv(node_dim, edge_dim) for _ in range(n_conv)])
        self.pool_mlp = nn.Sequential(nn.Linear(node_dim, node_dim), nn.Softplus())
        # 每个回归目标: 预测 (mean, log_var)
        self.reg_heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(node_dim, 64), nn.Softplus(), nn.Linear(64, 2))
            for t in self.REG_TARGETS
        })
        self.cls_head = nn.Sequential(nn.Linear(node_dim, 64), nn.Softplus(), nn.Linear(64, 1))

    def forward(self, z, edge_index, edge_len, batch, n_graphs):
        x = self.embed(z)
        edge_attr = self.rbf(edge_len)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = self.pool_mlp(x)
        # 按图平均池化
        g = torch.zeros(n_graphs, x.size(1), device=x.device).index_add_(0, batch, x)
        counts = torch.zeros(n_graphs, 1, device=x.device).index_add_(
            0, batch, torch.ones(x.size(0), 1, device=x.device))
        g = g / counts.clamp(min=1)
        out = {}
        for t in self.REG_TARGETS:
            h = self.reg_heads[t](g)
            out[t] = (h[:, 0], h[:, 1])     # (mean, log_var)
        out["is_switchable"] = self.cls_head(g).squeeze(-1)
        return out


def heteroscedastic_loss(mean, log_var, target):
    """异方差回归损失 (含不确定性): 0.5*exp(-s)*(y-μ)² + 0.5*s。"""
    return (0.5 * torch.exp(-log_var) * (target - mean) ** 2 + 0.5 * log_var).mean()


if __name__ == "__main__":
    # 形状自检 (CPU)
    m = FerroPropertyGNN(node_dim=32, edge_dim=16, n_conv=2)
    z = torch.randint(1, 50, (40,))
    ei = torch.randint(0, 40, (2, 200))
    el = torch.rand(200) * 6
    batch = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    out = m(z, ei, el, batch, 2)
    print("forward ok; outputs:", {k: (tuple(v[0].shape) if isinstance(v, tuple) else tuple(v.shape)) for k, v in out.items()})
