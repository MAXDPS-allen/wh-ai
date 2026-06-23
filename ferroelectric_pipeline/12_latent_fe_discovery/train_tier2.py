#!/usr/bin/env python3
"""
级联 Tier 2: 等变"软模参与张量"回归 (简并无关的精细预测)
=====================================================================
本征矢方向回归病态 (72% 简并)。解决办法: 不预测单一本征矢, 而预测每原子的
**软模参与张量** S_i = Σ_{m∈软模子空间} e_i^m ⊗ e_i^m (对称 rank-2)。
S_i 在简并子空间的酉混合下**不变** (规范无关), 既给出每个原子在软模中的位移
**幅度** (tr S_i) 又给出**方向** (S_i 的主轴) —— 良定义、可收敛。

软模子空间 = 最软光学模 + 其 0.1 THz 内的简并伙伴。
等变实现: 复用 model 的 rank-2 头 (head_zstar, CartesianTensor 'ij=ji', 0e+2e)。

数据缓存需 pymatgen (fe_dft); 训练在 GPU (fe_gpu)。
用法: 先 fe_dft 建缓存, 再 fe_gpu 训练:
  conda run -n fe_dft python -c "from train_tier2 import SoftTensorData; SoftTensorData()"
  python train_tier2.py --epochs 200 --device cuda
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from model import LatentFEModel
from train import build_graph

HERE = Path(__file__).parent


def participation_tensor(freqs, eigdisp, degen_tol=0.1):
    """每原子软模参与张量 S_i (natoms,3,3), Σ_i tr S_i = 1。"""
    nb = len(freqs)
    acoustic = set(np.argsort(np.abs(freqs))[:3].tolist())
    opt = [i for i in range(nb) if i not in acoustic] or list(range(nb))
    soft = opt[int(np.argmin(freqs[opt]))]
    sf = freqs[soft]
    soft_set = [m for m in opt if abs(freqs[m] - sf) < degen_tol] or [soft]
    natoms = eigdisp.shape[1]
    S = np.zeros((natoms, 3, 3), dtype=np.float32)
    for m in soft_set:
        e = eigdisp[m]                                  # (natoms,3) real
        S += np.einsum("ia,ib->iab", e, e)
    S /= max(len(soft_set), 1)
    tot = np.trace(S, axis1=1, axis2=2).sum()
    if tot > 1e-8:
        S /= tot                                        # Σ_i tr S_i = 1
    return S, float(sf), len(soft_set)


class SoftTensorData(Dataset):
    def __init__(self, modes_dir: Path = HERE / "modes_data",
                 cache: Path = HERE / "tier2_cache.pt"):
        if cache.exists():
            self.items = torch.load(cache, weights_only=False); return
        from pymatgen.core import Structure
        self.items = []
        for f in sorted(modes_dir.glob("*.npz")):
            try:
                d = np.load(f, allow_pickle=True)
                st = Structure.from_dict(json.loads(str(d["structure"])))
                fr = d["gamma_freqs"]; ed = d["gamma_eigdisp"]
                if ed.shape[1] != len(st):
                    continue
                S, sf, nset = participation_tensor(fr, ed)
                z, pos, src, dst, vec = build_graph(st)
                if len(src) == 0:
                    continue
                self.items.append({
                    "z": torch.tensor(z), "pos": torch.tensor(pos),
                    "src": torch.tensor(src), "dst": torch.tensor(dst),
                    "vec": torch.tensor(vec),
                    "S": torch.tensor(S), "sfreq": sf})
            except Exception:
                continue
        torch.save(self.items, cache)
        print("built tier2 cache:", len(self.items))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    z, pos, src, dst, vec, S, bidx = [], [], [], [], [], [], []
    off = 0
    for g, it in enumerate(batch):
        n = it["z"].shape[0]
        z.append(it["z"]); pos.append(it["pos"])
        src.append(it["src"] + off); dst.append(it["dst"] + off)
        vec.append(it["vec"]); S.append(it["S"])
        bidx.append(torch.full((n,), g, dtype=torch.long)); off += n
    return {"z": torch.cat(z), "pos": torch.cat(pos), "src": torch.cat(src),
            "dst": torch.cat(dst), "vec": torch.cat(vec), "S": torch.cat(S),
            "batch": torch.cat(bidx), "n": len(batch)}


@torch.no_grad()
def evaluate(model, loader, device):
    """指标: 参与幅度(trace)的 R²; 张量 Frobenius 余弦 (方向+幅度, 简并无关)。"""
    model.eval()
    tr_t, tr_p, fcos = [], [], []
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
        Spred = model.ct.to_cartesian(out["zstar"].cpu()).numpy()    # (N,3,3)
        Strue = b["S"].cpu().numpy()
        tr_p += np.trace(Spred, axis1=1, axis2=2).tolist()
        tr_t += np.trace(Strue, axis1=1, axis2=2).tolist()
        for a in range(Strue.shape[0]):
            num = (Spred[a] * Strue[a]).sum()
            den = np.linalg.norm(Spred[a]) * np.linalg.norm(Strue[a]) + 1e-9
            fcos.append(num / den)
    tr_t, tr_p = np.array(tr_t), np.array(tr_p)
    ss = ((tr_t - tr_p) ** 2).sum(); st = ((tr_t - tr_t.mean()) ** 2).sum() + 1e-9
    return {"participation_R2": float(1 - ss / st),
            "tensor_frobenius_cos": float(np.mean(fcos))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pretrained", type=Path, default=HERE / "pretrained_backbone.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ds = SoftTensorData()
    print(f"N={len(ds)} device={args.device}", flush=True)
    nv = max(1, int(0.15 * len(ds)))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed)).tolist()
    tl = DataLoader(Subset(ds, perm[nv:]), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(Subset(ds, perm[:nv]), batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = LatentFEModel().to(args.device)
    if args.pretrained.exists():       # 用 Born 主干热启动 (等变电响应表示)
        try:
            model.load_state_dict(torch.load(args.pretrained, map_location=args.device,
                                             weights_only=True), strict=True)
            print(f"warm-start from {args.pretrained.name}", flush=True)
        except Exception as e:
            print("no warm-start:", str(e)[:60], flush=True)
    ct = model.ct
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = -1e9
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            target = ct.from_cartesian(b["S"]).to(args.device)
            loss = ((out["zstar"] - target) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, args.device)
            print(f"ep{ep+1:3d} loss={loss.item():.4f} | participation_R2={m['participation_R2']:.3f} "
                  f"tensor_cos={m['tensor_frobenius_cos']:.3f}", flush=True)
            score = m["tensor_frobenius_cos"]
            if score > best:
                best = score; torch.save(model.state_dict(), HERE / "tier2_softtensor.pt")
    m = evaluate(model, vl, args.device)
    json.dump(m, open(HERE / "tier2_metrics.json", "w"), indent=2)
    print(f"\nTier-2 final: {json.dumps(m, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
