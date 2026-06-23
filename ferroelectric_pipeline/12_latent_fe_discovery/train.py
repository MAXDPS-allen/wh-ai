#!/usr/bin/env python3
"""
训练 LatentFEModel: 从高对称母相预测 极性失稳本征矢 + 自发极化
=====================================================================
关键: 极性软模/极化有任意整体**符号** (±P 两变体等价), 损失须对符号翻转不变
(u→-u, Ps→-Ps 同步)。本脚本用"按预测对齐靶符号"实现 sign-invariant 多任务损失。

评估指标 (新颖能力的度量):
  - mode 方向余弦 |cos<pred,true>|  —— 能否预测极性畸变方向 (本征矢)
  - Ps 矢量余弦 + |Ps| 的 R²/MAE
对照基线: 各向同性"平均模式"预测 (证明等变几何预测的价值)。

用法: conda activate fe_dft && python train.py --epochs 300 --device cpu
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from model import LatentFEModel

HERE = Path(__file__).parent


def build_graph(structure, cutoff=6.0, max_neigh=16):
    z = np.array([s.specie.Z for s in structure], dtype=np.int64)
    pos = structure.cart_coords.astype(np.float32)
    src, dst, vec = [], [], []
    for i, neighbors in enumerate(structure.get_all_neighbors(cutoff)):
        neighbors = sorted(neighbors, key=lambda n: n.nn_distance)[:max_neigh]
        for n in neighbors:
            src.append(i); dst.append(n.index); vec.append(n.coords - structure[i].coords)
    return (z, pos, np.array(src, np.int64), np.array(dst, np.int64),
            np.array(vec, np.float32))


class FEData(Dataset):
    def __init__(self, ds_dir: Path, cache=True):
        from pymatgen.core import Structure
        rows = list(csv.DictReader(open(ds_dir / "labels.csv")))
        modes = np.load(ds_dir / "modes.npz")
        cpath = ds_dir / "graph_cache.pt"
        if cache and cpath.exists():
            self.items = torch.load(cpath, weights_only=False)
            return
        self.items = []
        for r in rows:
            if int(r["label"]) != 1:        # 目前只用正样本做回归
                continue
            st = Structure.from_dict(json.load(open(ds_dir / r["parent_file"])))
            z, pos, src, dst, vec = build_graph(st)
            if len(src) == 0:
                continue
            self.items.append({
                "z": torch.tensor(z), "pos": torch.tensor(pos),
                "src": torch.tensor(src), "dst": torch.tensor(dst),
                "vec": torch.tensor(vec),
                "mode": torch.tensor(modes[r["cid"]]),                 # (N,3)
                "Ps": torch.tensor([float(r["Ps_x"]), float(r["Ps_y"]), float(r["Ps_z"])]),
                "dw": torch.tensor(float(r["dw_depth_meV"])),
                "cid": r["cid"],
            })
        if cache:
            torch.save(self.items, cpath)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    z, pos, src, dst, vec, mode, Ps, dw, bidx = [], [], [], [], [], [], [], [], []
    off = 0
    for g, item in enumerate(batch):
        n = item["z"].shape[0]
        z.append(item["z"]); pos.append(item["pos"])
        src.append(item["src"] + off); dst.append(item["dst"] + off)
        vec.append(item["vec"]); mode.append(item["mode"])
        Ps.append(item["Ps"]); dw.append(item["dw"])
        bidx.append(torch.full((n,), g, dtype=torch.long))
        off += n
    return {"z": torch.cat(z), "pos": torch.cat(pos), "src": torch.cat(src),
            "dst": torch.cat(dst), "vec": torch.cat(vec), "mode": torch.cat(mode),
            "Ps": torch.stack(Ps), "dw": torch.stack(dw),
            "batch": torch.cat(bidx), "n": len(batch)}


def _per_struct_cos2(pred_atom, true_atom, bvec, n):
    """每结构 (3N 维场) 的 cos^2 (符号无关) 与预测/真实模长。"""
    dot = torch.zeros(n, device=pred_atom.device).index_add_(0, bvec, (pred_atom * true_atom).sum(1))
    pn2 = torch.zeros(n, device=pred_atom.device).index_add_(0, bvec, (pred_atom ** 2).sum(1))
    tn2 = torch.zeros(n, device=pred_atom.device).index_add_(0, bvec, (true_atom ** 2).sum(1))
    cos2 = dot ** 2 / (pn2 * tn2 + 1e-9)
    return cos2, pn2.sqrt(), tn2.sqrt()


def sign_aligned_losses(out, b):
    """方向(cos^2)主导的 sign-invariant 多任务损失。"""
    bvec = b["batch"]; n = b["n"]
    # mode 方向损失: 1 - cos^2  (直接优化方向余弦, 符号无关)
    cos2, pn, tn = _per_struct_cos2(out["mode"], b["mode"], bvec, n)
    mode_dir = (1.0 - cos2).mean()
    mode_mag = ((pn - tn) ** 2).mean() * 0.05                    # 模长辅助
    # Ps 方向 + 模长
    ps_pn = out["Ps"].norm(dim=1); ps_tn = b["Ps"].norm(dim=1)
    ps_cos2 = (out["Ps"] * b["Ps"]).sum(1) ** 2 / (ps_pn ** 2 * ps_tn ** 2 + 1e-9)
    ps_loss = (1.0 - ps_cos2).mean() + 0.02 * ((ps_pn - ps_tn) ** 2).mean()
    # dw (失稳强度)
    dw_loss = ((out["amp"] - b["dw"]) ** 2).mean()
    return mode_dir + mode_mag + 0.3 * ps_loss + 1e-3 * dw_loss, (mode_dir, ps_loss, dw_loss)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cos_modes, ps_true, ps_pred, ps_cos = [], [], [], []
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
        bvec = b["batch"]
        for g in range(b["n"]):
            m = bvec == g
            pm = out["mode"][m]; tm = b["mode"][m]
            c = (pm * tm).sum() / (pm.norm() * tm.norm() + 1e-9)
            cos_modes.append(abs(c.item()))                      # 方向余弦 (符号无关)
            pp = out["Ps"][g]; tp = b["Ps"][g]
            ps_pred.append(pp.norm().item()); ps_true.append(tp.norm().item())
            ps_cos.append(abs((pp * tp).sum().item()) / (pp.norm().item() * tp.norm().item() + 1e-9))
    ps_true, ps_pred = np.array(ps_true), np.array(ps_pred)
    ss_res = ((ps_true - ps_pred) ** 2).sum()
    ss_tot = ((ps_true - ps_true.mean()) ** 2).sum() + 1e-9
    return {"mode_cos": float(np.mean(cos_modes)),
            "Ps_R2": float(1 - ss_res / ss_tot),
            "Ps_MAE": float(np.abs(ps_true - ps_pred).mean()),
            "Ps_vec_cos": float(np.mean(ps_cos))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ds = FEData(HERE / "dataset")
    print(f"N={len(ds)} parents | device={args.device}")
    n_val = max(1, int(0.2 * len(ds)))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed)).tolist()
    va = Subset(ds, perm[:n_val]); tr = Subset(ds, perm[n_val:])
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = LatentFEModel().to(args.device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"model params: {npar/1e3:.0f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    # 基线: 训练集平均 |Ps| 与 "随机方向" mode 余弦期望(=~0.5 for |cos| in 3D? actually E|cos|≈0.5)
    best = -1e9
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            loss, parts = sign_aligned_losses(out, b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, args.device)
            print(f"ep{ep+1:3d} loss={loss.item():.3f} | mode_cos={m['mode_cos']:.3f} "
                  f"Ps_R2={m['Ps_R2']:.3f} Ps_vec_cos={m['Ps_vec_cos']:.3f} Ps_MAE={m['Ps_MAE']:.2f}")
            if m["mode_cos"] > best:
                best = m["mode_cos"]
                torch.save(model.state_dict(), HERE / "latent_fe_model.pt")
    m = evaluate(model, vl, args.device)
    json.dump(m, open(HERE / "val_metrics.json", "w"), indent=2)
    print("\nFinal:", json.dumps(m, indent=2))
    # 正确的高维随机基线: mode 场维度 = 3N (用验证集真实原子数)
    rng = np.random.default_rng(0)
    dims = [3 * ds[i]["z"].shape[0] for i in perm[:n_val]]
    rc = []
    for D in dims:
        a, bb = rng.normal(size=D), rng.normal(size=D)
        rc.append(abs(np.dot(a, bb) / (np.linalg.norm(a) * np.linalg.norm(bb))))
    print(f"random mode-field |cos| baseline (matched 3N) ≈ {np.mean(rc):.3f}")


if __name__ == "__main__":
    main()
