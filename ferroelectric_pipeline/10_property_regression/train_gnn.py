#!/usr/bin/env python3
"""
训练几何感知 GNN 回归模型 (生产)
=====================================================================
在 Smidt et al. DFT 数据库上训练 FerroPropertyGNN, 直接从晶体结构图预测
铁电关键物性 (Ps / dw_depth / path_barrier / gap_polar) + 可切换性分类,
含异方差不确定性 (供主动学习)。

graphs 缓存到 dataset/graph_cache.pt 以加速重复实验。

CPU 可跑 (小数据集), GPU 节点更快:
  conda activate fe_dft
  CUDA_VISIBLE_DEVICES=0 python train_gnn.py --epochs 300 --device cuda
  python train_gnn.py --epochs 120 --device cpu      # 快速验证
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from featurize import graph_from_structure
from model_gnn import FerroPropertyGNN, heteroscedastic_loss

REG_TARGETS = FerroPropertyGNN.REG_TARGETS
LOG_TARGETS = {"Ps"}


class GraphDataset(Dataset):
    def __init__(self, csv_path: Path, cache: Path | None = None):
        base = csv_path.parent
        rows = list(csv.DictReader(open(csv_path)))
        if cache and cache.exists():
            blob = torch.load(cache, weights_only=False)
            self.items, self.Y, self.cls = blob["items"], blob["Y"], blob["cls"]
        else:
            self.items, Y, cls = [], [], []
            for r in rows:
                try:
                    z, ei, ev, el = graph_from_structure(base / r["structure_file"])
                except Exception as e:
                    print("  skip", r["formula"], e); continue
                if ei.shape[1] == 0:
                    continue
                self.items.append((torch.tensor(z), torch.tensor(ei), torch.tensor(el)))
                Y.append([float(r[t]) for t in REG_TARGETS])
                cls.append(int(r["is_switchable"]))
            self.Y = torch.tensor(Y, dtype=torch.float32)
            self.cls = torch.tensor(cls, dtype=torch.float32)
            if cache:
                torch.save({"items": self.items, "Y": self.Y, "cls": self.cls}, cache)
        # 目标标准化 (log1p for Ps)
        self.Yt = self.Y.clone()
        for ti, t in enumerate(REG_TARGETS):
            if t in LOG_TARGETS:
                self.Yt[:, ti] = torch.log1p(self.Yt[:, ti].clamp(min=0))
        self.mean = self.Yt.mean(0); self.std = self.Yt.std(0).clamp(min=1e-6)
        self.Yn = (self.Yt - self.mean) / self.std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.Yn[i], self.cls[i]


def collate(batch):
    zs, eis, els, ys, cs = [], [], [], [], []
    batch_idx = []
    offset = 0
    for gi, ((z, ei, el), y, c) in enumerate(batch):
        zs.append(z); els.append(el)
        eis.append(ei + offset)
        batch_idx.append(torch.full((len(z),), gi, dtype=torch.long))
        offset += len(z)
        ys.append(y); cs.append(c)
    return (torch.cat(zs), torch.cat(eis, 1), torch.cat(els),
            torch.cat(batch_idx), len(batch),
            torch.stack(ys), torch.stack(cs))


def evaluate(model, loader, ds, device):
    model.eval()
    preds = {t: [] for t in REG_TARGETS}; trues = {t: [] for t in REG_TARGETS}
    cls_p, cls_t = [], []
    with torch.no_grad():
        for z, ei, el, b, n, y, c in loader:
            out = model(z.to(device), ei.to(device), el.to(device), b.to(device), n)
            for ti, t in enumerate(REG_TARGETS):
                mu = out[t][0].cpu() * ds.std[ti] + ds.mean[ti]
                yt = y[:, ti] * ds.std[ti] + ds.mean[ti]
                if t in LOG_TARGETS:
                    mu, yt = torch.expm1(mu), torch.expm1(yt)
                preds[t] += mu.tolist(); trues[t] += yt.tolist()
            cls_p += torch.sigmoid(out["is_switchable"]).cpu().tolist(); cls_t += c.tolist()
    from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
    metrics = {}
    for t in REG_TARGETS:
        metrics[t] = {"r2": r2_score(trues[t], preds[t]),
                      "mae": mean_absolute_error(trues[t], preds[t])}
    try:
        metrics["switchable_auc"] = roc_auc_score(cls_t, cls_p)
    except Exception:
        metrics["switchable_auc"] = float("nan")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset" / "regression_dataset.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ds = GraphDataset(args.dataset, cache=args.dataset.parent / "graph_cache.pt")
    n_val = int(len(ds) * args.val_frac)
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
    val_idx, tr_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    tr = torch.utils.data.Subset(ds, tr_idx); va = torch.utils.data.Subset(ds, val_idx)
    print(f"N={len(ds)} train={len(tr)} val={len(va)} device={args.device}")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = FerroPropertyGNN().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = torch.nn.BCEWithLogitsLoss()

    best = -1e9
    for ep in range(args.epochs):
        model.train()
        for z, ei, el, b, n, y, c in tl:
            z, ei, el, b = z.to(args.device), ei.to(args.device), el.to(args.device), b.to(args.device)
            y, c = y.to(args.device), c.to(args.device)
            out = model(z, ei, el, b, n)
            loss = sum(heteroscedastic_loss(out[t][0], out[t][1], y[:, ti])
                       for ti, t in enumerate(REG_TARGETS))
            loss = loss + bce(out["is_switchable"], c)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, ds, args.device)
            avg_r2 = np.mean([m[t]["r2"] for t in REG_TARGETS])
            print(f"ep{ep+1:3d} loss={loss.item():.3f} | "
                  + " ".join(f"{t} R²={m[t]['r2']:.2f}" for t in REG_TARGETS)
                  + f" | sw_AUC={m['switchable_auc']:.2f}")
            if avg_r2 > best:
                best = avg_r2
                torch.save(model.state_dict(), args.dataset.parent.parent / "fe_property_gnn.pt")

    m = evaluate(model, vl, ds, args.device)
    out = {"val_metrics": {t: m[t] for t in REG_TARGETS},
           "switchable_auc": m["switchable_auc"]}
    json.dump(out, open(args.dataset.parent.parent / "results" / "gnn_val_metrics.json", "w"), indent=2)
    print("\nFinal val metrics:")
    for t in REG_TARGETS:
        print(f"  {t:<14} R²={m[t]['r2']:.3f}  MAE={m[t]['mae']:.3f}")
    print(f"  switchable AUC={m['switchable_auc']:.3f}")


if __name__ == "__main__":
    main()
