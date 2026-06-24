#!/usr/bin/env python3
"""
5-折交叉验证: 评估 Tier-1 / Tier-2 的稳定性与准确度 (mean ± std)
=====================================================================
对 1534 个声子材料做 5-折 CV, 每折独立训练并在留出折评估:
  Tier-1: ROC-AUC (失稳分类)
  Tier-2: R² / sign-AUC / MAE (软度回归)
输出每折指标 + 均值±标准差 → 证明指标稳定, 非单次划分侥幸。
用法: python cv_validation.py --folds 5 --device cuda
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import LatentFEModel
from train_instability import InstabilityData, collate as icol, _auc
from train_tier2_freq import FreqData, collate as fcol

HERE = Path(__file__).parent
CACHE = HERE / "softmode_graph_cache.pt"


def kfold_indices(n, k, seed=0):
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    folds = [idx[i::k] for i in range(k)]
    return folds


def train_tier1(ds, tr, va, device, epochs=80):
    pos = sum(float(ds[i][1]) for i in tr); n = len(tr)
    model = LatentFEModel().to(device)
    pw = torch.tensor([(n - pos) / max(pos, 1)], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(Subset(ds, tr), batch_size=32, shuffle=True, collate_fn=icol)
    for _ in range(epochs):
        model.train()
        for b, y in tl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            loss = bce(out["logit"], y.to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
    model.eval(); ys, ps = [], []
    vl = DataLoader(Subset(ds, va), batch_size=64, shuffle=False, collate_fn=icol)
    with torch.no_grad():
        for b, y in vl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            ps += torch.sigmoid(out["logit"]).cpu().tolist(); ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)
    k = max(1, len(ps) // 10); top = np.argsort(-ps)[:k]
    enr = ys[top].mean() / max(ys.mean(), 1e-9)
    return {"AUC": _auc(ys, ps), "enrichment_top10pct": float(enr)}


def train_tier2(ds, tr, va, device, epochs=120):
    y = ds.y; mean = y[tr].mean(); std = y[tr].std().clamp(min=1e-3)
    md, sd = mean.to(device), std.to(device)
    model = LatentFEModel().to(device)
    bb = HERE / "pretrained_backbone.pt"
    if bb.exists():
        try: model.load_state_dict(torch.load(bb, map_location=device, weights_only=True), strict=True)
        except Exception: pass
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(Subset(ds, tr), batch_size=32, shuffle=True, collate_fn=fcol)
    for _ in range(epochs):
        model.train()
        for b, yy in tl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            yn = (yy.to(device) - md) / sd
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            loss = ((out["amp"] - yn) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
    model.eval(); yp, yt = [], []
    vl = DataLoader(Subset(ds, va), batch_size=64, shuffle=False, collate_fn=fcol)
    with torch.no_grad():
        for b, yy in vl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            yp += (out["amp"].cpu() * std + mean).tolist(); yt += yy.tolist()
    yp, yt = np.array(yp), np.array(yt)
    ss = ((yt - yp) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum() + 1e-9
    return {"R2": float(1 - ss / st), "MAE": float(np.abs(yt - yp).mean()),
            "sign_AUC": float(_auc((yt < -0.1).astype(int), -yp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds1 = InstabilityData(CACHE); ds2 = FreqData(CACHE); n = len(ds1)
    folds = kfold_indices(n, args.folds)
    print(f"N={n}, {args.folds}-fold CV, device={args.device}", flush=True)
    t1, t2 = [], []
    for f in range(args.folds):
        va = folds[f]; tr = [i for j in range(args.folds) if j != f for i in folds[j]]
        r1 = train_tier1(ds1, tr, va, args.device)
        r2 = train_tier2(ds2, tr, va, args.device)
        t1.append(r1); t2.append(r2)
        print(f"fold {f+1}: Tier1 AUC={r1['AUC']:.3f} enr={r1['enrichment_top10pct']:.1f}x | "
              f"Tier2 R2={r2['R2']:.3f} signAUC={r2['sign_AUC']:.3f}", flush=True)

    def agg(lst, key): a = np.array([d[key] for d in lst]); return float(a.mean()), float(a.std())
    summary = {
        "folds": args.folds,
        "tier1_AUC": agg(t1, "AUC"), "tier1_enrichment": agg(t1, "enrichment_top10pct"),
        "tier2_R2": agg(t2, "R2"), "tier2_sign_AUC": agg(t2, "sign_AUC"), "tier2_MAE": agg(t2, "MAE"),
        "per_fold_tier1": t1, "per_fold_tier2": t2,
    }
    json.dump(summary, open(HERE / "cv_results.json", "w"), indent=2)
    print("\n==== 5-fold CV (mean ± std) ====", flush=True)
    print(f"Tier-1 AUC        : {summary['tier1_AUC'][0]:.3f} ± {summary['tier1_AUC'][1]:.3f}", flush=True)
    print(f"Tier-1 enrichment : {summary['tier1_enrichment'][0]:.1f} ± {summary['tier1_enrichment'][1]:.1f}x", flush=True)
    print(f"Tier-2 R²         : {summary['tier2_R2'][0]:.3f} ± {summary['tier2_R2'][1]:.3f}", flush=True)
    print(f"Tier-2 sign-AUC   : {summary['tier2_sign_AUC'][0]:.3f} ± {summary['tier2_sign_AUC'][1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
