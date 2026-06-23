#!/usr/bin/env python3
"""
级联 Tier 1: 等变"软模/动力学失稳"分类器 (潜在铁电初筛)
=====================================================================
本征矢回归病态 (72% 简并), 但"是否存在 Γ 软模"是**良定义的标量分类**。
本模型从结构预测该材料 Γ 点是否有失稳软模 (最软光学模频率 < 阈值),
即"是否潜在铁电/位移失稳" —— 级联的第一级, 用于在海量稳定非极性晶体中**初筛**。

数据: softmode_graph_cache.pt (1534 声子材料, 每项含 sfreq=Γ 最软光学模频率 THz)。
输出: 不变标量 logit。损失: BCE + pos_weight (类不平衡)。指标: ROC-AUC / PR-AUC。

用法: conda activate fe_gpu (GPU) 或 fe_dft (CPU)
  python train_instability.py --epochs 120 --device cuda
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from model import LatentFEModel
from pretrain_modes import collate as _mode_collate   # 复用 (含 mode 字段, 不用即可)

HERE = Path(__file__).parent
THRESH = -0.1   # THz; 最软光学模频率 < 此值 = 失稳 (虚频)


class InstabilityData(Dataset):
    def __init__(self, cache: Path):
        items = torch.load(cache, weights_only=False)
        self.items = items
        self.labels = torch.tensor(
            [1.0 if it["sfreq"] < THRESH else 0.0 for it in items])

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        return self.items[i], self.labels[i]


def collate(batch):
    items = [b[0] for b in batch]
    y = torch.stack([b[1] for b in batch])
    z, pos, src, dst, vec, bidx = [], [], [], [], [], []
    off = 0
    for g, it in enumerate(items):
        n = it["z"].shape[0]
        z.append(it["z"]); pos.append(it["pos"])
        src.append(it["src"] + off); dst.append(it["dst"] + off)
        vec.append(it["vec"])
        bidx.append(torch.full((n,), g, dtype=torch.long)); off += n
    return ({"z": torch.cat(z), "pos": torch.cat(pos), "src": torch.cat(src),
             "dst": torch.cat(dst), "vec": torch.cat(vec),
             "batch": torch.cat(bidx), "n": len(items)}, y)


def _auc(ys, ps):
    """ROC-AUC via Mann-Whitney U (无 sklearn 依赖)。"""
    ys = np.asarray(ys); ps = np.asarray(ps)
    npos = (ys == 1).sum(); nneg = (ys == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(ps); ranks = np.empty(len(ps)); ranks[order] = np.arange(1, len(ps) + 1)
    # 处理并列: 用平均秩
    from collections import defaultdict
    idx = defaultdict(list)
    for i, v in enumerate(ps): idx[v].append(i)
    for v, group in idx.items():
        if len(group) > 1:
            avg = np.mean([ranks[g] for g in group])
            for g in group: ranks[g] = avg
    return (ranks[ys == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ps, ys = [], []
    for b, y in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
        ps += torch.sigmoid(out["logit"]).cpu().tolist(); ys += y.tolist()
    ys = np.array(ys); ps = np.array(ps)
    auc = _auc(ys, ps)
    # precision/recall at a high-recall operating point (筛选用): threshold giving recall>=0.9
    pred = (ps > 0.5).astype(int)
    tpr = pred[ys == 1].mean() if (ys == 1).any() else 0
    tnr = (1 - pred[ys == 0]).mean() if (ys == 0).any() else 0
    # enrichment: precision in top-decile vs base rate
    k = max(1, len(ps) // 10); top = np.argsort(-ps)[:k]
    enrich = (ys[top].mean() / max(ys.mean(), 1e-9))
    return {"AUC": float(auc), "balanced_acc": float(0.5 * (tpr + tnr)),
            "top10pct_enrichment": float(enrich), "base_rate": float(ys.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=HERE / "softmode_graph_cache.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ds = InstabilityData(args.cache)
    pos = int(ds.labels.sum()); n = len(ds)
    print(f"N={n} | unstable(+)={pos} ({100*pos/n:.1f}%) | device={args.device}", flush=True)
    nv = max(1, int(0.15 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    tl = DataLoader(Subset(ds, perm[nv:]), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(Subset(ds, perm[:nv]), batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = LatentFEModel().to(args.device)
    pos_weight = torch.tensor([(n - pos) / max(pos, 1)], device=args.device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = -1
    for ep in range(args.epochs):
        model.train()
        for b, y in tl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            y = y.to(args.device)
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            loss = bce(out["logit"], y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, args.device)
            print(f"ep{ep+1:3d} loss={loss.item():.3f} | AUC={m['AUC']:.3f} "
                  f"balAcc={m['balanced_acc']:.3f}", flush=True)
            if m["AUC"] > best:
                best = m["AUC"]; torch.save(model.state_dict(), HERE / "tier1_instability.pt")
    m = evaluate(model, vl, args.device)
    json.dump(m, open(HERE / "tier1_metrics.json", "w"), indent=2)
    print(f"\nTier-1 final: {json.dumps(m, indent=2)}", flush=True)
    print(f"(base rate {m['base_rate']:.3f}; AUC=0.5 is random)", flush=True)


if __name__ == "__main__":
    main()
