#!/usr/bin/env python3
"""
E(3)-等变网络: 从高对称母相预测 极性失稳本征矢 + 自发极化 + 失稳判据
=====================================================================
核心方法学创新。输入高对称非极性晶体, 输出:
  - mode  : 每原子位移矢量场 (l=1o, 等变) —— 极性软模本征矢
  - Ps    : 全局极化矢量 (l=1o, 等变) —— 假想极性相自发极化
  - logit : 失稳分类 (l=0e, 标量) —— 是否为潜在铁电
  - amp   : 双势阱深度/失稳强度 (l=0e, 标量)
所有输出在 E(3) (旋转+反演+平移) 下正确变换: mode/Ps 为极性矢量 (1o, 反演变号)。

依赖: torch, e3nn
"""
from __future__ import annotations
import torch
from e3nn import o3
from e3nn.nn import Gate, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
from e3nn.io import CartesianTensor


def scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size,) + src.shape[1:])
    return out.index_add_(0, index, src)


class Convolution(torch.nn.Module):
    """NequIP/TFN 风格等变卷积: node ⊗ Y(r_ij), 权重由 |r_ij| 的径向 MLP 给出。"""
    def __init__(self, irreps_in, irreps_sh, irreps_out, num_basis=8, radial_hidden=64):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False)
        self.radial = FullyConnectedNet(
            [num_basis, radial_hidden, self.tp.weight_numel], torch.nn.functional.silu)
        self.sc = o3.Linear(self.irreps_in, self.irreps_out)   # 自连接

    def forward(self, x, edge_src, edge_dst, edge_sh, edge_len_emb, avg_deg=12.0):
        w = self.radial(edge_len_emb)
        msg = self.tp(x[edge_src], edge_sh, w)
        agg = scatter_sum(msg, edge_dst, x.shape[0]) / (avg_deg ** 0.5)
        return self.sc(x) + agg


class LatentFEModel(torch.nn.Module):
    def __init__(self, max_z=100, lmax=2, num_layers=4, num_basis=8, cutoff=6.0,
                 hidden_scalars=64, hidden_vectors=16, hidden_l2=8):
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis
        self.embed = torch.nn.Embedding(max_z + 1, hidden_scalars)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        irreps_hidden = o3.Irreps(
            f"{hidden_scalars}x0e + {hidden_vectors}x1o + {hidden_l2}x2e")
        irreps_node = o3.Irreps(f"{hidden_scalars}x0e")

        # 逐层卷积 + 门控非线性
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        irreps_in = irreps_node
        for _ in range(num_layers):
            # Gate: 标量做激活, 高阶用标量门控
            irreps_scalars = o3.Irreps(f"{hidden_scalars}x0e")
            irreps_gated = o3.Irreps(f"{hidden_vectors}x1o + {hidden_l2}x2e")
            irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e")
            gate = Gate(irreps_scalars, [torch.nn.functional.silu],
                        irreps_gates, [torch.sigmoid], irreps_gated)
            conv = Convolution(irreps_in, self.irreps_sh, gate.irreps_in, num_basis)
            self.layers.append(conv)
            self.gates.append(gate)
            irreps_in = gate.irreps_out
        self.irreps_feat = irreps_in

        # 输出头
        self.head_mode = o3.Linear(self.irreps_feat, o3.Irreps("1x1o"))      # 每原子位移矢量
        self.head_ps = o3.Linear(self.irreps_feat, o3.Irreps("1x1o"))        # 池化后极化矢量
        self.head_scalar = o3.Linear(self.irreps_feat, o3.Irreps("2x0e"))    # logit + amp
        # Born 有效电荷 Z* (对称部分): 0e + 2e (rank-2 等变张量)
        self.ct = CartesianTensor("ij=ji")                                   # 1x0e+1x2e
        self.head_zstar = o3.Linear(self.irreps_feat, o3.Irreps(self.ct))

    def forward(self, z, pos, edge_src, edge_dst, edge_vec, batch, n_graphs):
        edge_len = edge_vec.norm(dim=1)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True,
                                         normalization="component")
        edge_len_emb = soft_one_hot_linspace(
            edge_len, 0.0, self.cutoff, self.num_basis,
            basis="smooth_finite", cutoff=True).mul(self.num_basis ** 0.5)

        x = self.embed(z)
        for conv, gate in zip(self.layers, self.gates):
            x = gate(conv(x, edge_src, edge_dst, edge_sh, edge_len_emb))

        mode = self.head_mode(x)                          # (N,3) 每原子位移
        # Ps: 池化每原子等变特征 -> 全局矢量
        node_ps = self.head_ps(x)                         # (N,3)
        ps = scatter_sum(node_ps, batch, n_graphs)        # (G,3)
        scal = self.head_scalar(x)                        # (N,2)
        glob = scatter_sum(scal, batch, n_graphs)         # (G,2)
        counts = scatter_sum(torch.ones_like(batch, dtype=x.dtype), batch, n_graphs).clamp(min=1)
        glob = glob / counts[:, None]
        logit, amp = glob[:, 0], glob[:, 1]
        zstar = self.head_zstar(x)                        # (N, 6) irrep coeffs (0e+2e)
        return {"mode": mode, "Ps": ps, "logit": logit, "amp": amp,
                "zstar": zstar, "feat": x}


if __name__ == "__main__":
    # 等变性自检: 旋转输入 -> mode/Ps 同步旋转
    torch.manual_seed(0)
    m = LatentFEModel(num_layers=2, hidden_scalars=16, hidden_vectors=8, hidden_l2=4)
    N = 12
    z = torch.randint(1, 30, (N,))
    pos = torch.randn(N, 3)
    # 简单全连接图
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i); dst.append(j)
    es, ed = torch.tensor(src), torch.tensor(dst)
    evec = pos[ed] - pos[es]
    batch = torch.zeros(N, dtype=torch.long)
    out = m(z, pos, es, ed, evec, batch, 1)

    R = o3.rand_matrix()
    evec_r = evec @ R.T
    out_r = m(z, pos @ R.T, es, ed, evec_r, batch, 1)
    err_mode = (out_r["mode"] - out["mode"] @ R.T).abs().max().item()
    err_ps = (out_r["Ps"] - out["Ps"] @ R.T).abs().max().item()
    err_logit = (out_r["logit"] - out["logit"]).abs().max().item()
    print(f"equivariance check: mode_err={err_mode:.2e}  Ps_err={err_ps:.2e}  logit_inv_err={err_logit:.2e}")
    print("PASS" if max(err_mode, err_ps, err_logit) < 1e-4 else "FAIL")
