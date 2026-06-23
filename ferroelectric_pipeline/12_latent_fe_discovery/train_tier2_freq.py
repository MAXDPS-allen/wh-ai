#!/usr/bin/env python3
"""
Tier-2 (重构): 软模"强度"标量回归 (signed 最软光学模频率, THz)
=====================================================================
参与张量(方向)回归失败 (病态/数据少)。改预测**标量软度**: Γ 最软光学模的
带符号频率 (负=失稳, 越负越强的位移失稳/铁电倾向)。标量任务 (像带隙/Z*) 更可学,
给出"失稳有多强"——与 Tier-1 (是否失稳) 互补, 用于精排候选。

数据: softmode_graph_cache.pt (含 sfreq)。输出: model 的标量头 (amp)。
用法: python train_tier2_freq.py --epochs 200 --device cuda
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from model import LatentFEModel

HERE = Path(__file__).parent


class FreqData(Dataset):
    def __init__(self, cache: Path):
        self.items = torch.load(cache, weights_only=False)
        # clip 极端值, 稳定训练
        self.y = torch.tensor([float(np.clip(it["sfreq"], -8, 12)) for it in self.items])

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i], self.y[i]


def collate(batch):
    items = [b[0] for b in batch]; y = torch.stack([b[1] for b in batch])
    z, pos, src, dst, vec, bidx = [], [], [], [], [], []
    off = 0
    for g, it in enumerate(items):
        n = it["z"].shape[0]
        z.append(it["z"]); pos.append(it["pos"]); src.append(it["src"] + off)
        dst.append(it["dst"] + off); vec.append(it["vec"])
        bidx.append(torch.full((n,), g, dtype=torch.long)); off += n
    return ({"z": torch.cat(z), "pos": torch.cat(pos), "src": torch.cat(src),
             "dst": torch.cat(dst), "vec": torch.cat(vec),
             "batch": torch.cat(bidx), "n": len(items)}, y)


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval(); yp, yt = [], []
    for b, y in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
        yp += (out["amp"].cpu() * std + mean).tolist(); yt += y.tolist()
    yp, yt = np.array(yp), np.array(yt)
    ss = ((yt - yp) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum() + 1e-9
    # 也报告"失稳"分类的 AUC (sign(freq)<0)
    from train_instability import _auc
    auc = _auc((yt < -0.1).astype(int), -yp)
    return {"R2": float(1 - ss / st), "MAE": float(np.abs(yt - yp).mean()),
            "sign_AUC": float(auc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=HERE / "softmode_graph_cache.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pretrained", type=Path, default=HERE / "pretrained_backbone.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    ds = FreqData(args.cache)
    n = len(ds); nv = max(1, int(0.15 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    tr_idx, va_idx = perm[nv:], perm[:nv]
    ytr = ds.y[tr_idx]; mean = ytr.mean(); std = ytr.std().clamp(min=1e-3)
    tl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False, collate_fn=collate)
    print(f"N={n} | target mean={mean:.2f} std={std:.2f} THz | device={args.device}", flush=True)

    model = LatentFEModel().to(args.device)
    if args.pretrained.exists():
        try:
            model.load_state_dict(torch.load(args.pretrained, map_location=args.device, weights_only=True), strict=True)
            print("warm-start from Born backbone", flush=True)
        except Exception: pass
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    mean_d, std_d = mean.to(args.device), std.to(args.device)

    best = -1e9
    for ep in range(args.epochs):
        model.train()
        for b, y in tl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            yn = ((y.to(args.device) - mean_d) / std_d)
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            loss = ((out["amp"] - yn) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, args.device, mean, std)
            print(f"ep{ep+1:3d} loss={loss.item():.3f} | R2={m['R2']:.3f} "
                  f"MAE={m['MAE']:.2f}THz sign_AUC={m['sign_AUC']:.3f}", flush=True)
            if m["R2"] > best:
                best = m["R2"]; torch.save(model.state_dict(), HERE / "tier2_freq.pt")
    m = evaluate(model, vl, args.device, mean, std)
    json.dump(m, open(HERE / "tier2_freq_metrics.json", "w"), indent=2)
    print(f"\nTier-2 freq final: {json.dumps(m, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
