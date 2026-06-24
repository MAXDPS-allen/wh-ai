#!/usr/bin/env python3
"""
Ps 回归消融: 复用 Ricci 2024 数据是否让模型更强?
=====================================================================
对每一折: 验证集固定; 分别用 (仅 Smidt 训练子集) 与 (Smidt+Ricci 训练子集) 训练同一模型,
在**同一验证集**上评估 R²/MAE。若 combined > smidt-only → 证明"复用他人结果 → 模型更强"。
目标: 自发极化 Ps (log1p 变换; 标量不变量, 用 amp 头)。
用法: python train_ps.py --folds 5 --device cuda
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from model import LatentFEModel

HERE = Path(__file__).parent


class PsData(Dataset):
    def __init__(self, cache):
        self.items = torch.load(cache, weights_only=False)
        self.y = torch.tensor([np.log1p(max(it["Ps"], 0.0)) for it in self.items], dtype=torch.float32)
        self.is_smidt = np.array([it["source"] == "smidt" for it in self.items])
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i], self.y[i]


def collate(batch):
    z, pos, src, dst, vec, bidx, ys = [], [], [], [], [], [], []
    off = 0
    for g, (it, y) in enumerate(batch):
        nn = it["z"].shape[0]
        z.append(it["z"]); pos.append(it["pos"]); src.append(it["src"]+off)
        dst.append(it["dst"]+off); vec.append(it["vec"])
        bidx.append(torch.full((nn,), g, dtype=torch.long)); off += nn; ys.append(y)
    return ({"z": torch.cat(z), "pos": torch.cat(pos), "src": torch.cat(src), "dst": torch.cat(dst),
             "vec": torch.cat(vec), "batch": torch.cat(bidx), "n": len(batch)}, torch.stack(ys))


def train_eval(ds, tr_idx, va_idx, device, epochs=150, warm=True):
    y = ds.y; mean = y[tr_idx].mean(); std = y[tr_idx].std().clamp(min=1e-3)
    md, sd = mean.to(device), std.to(device)
    model = LatentFEModel().to(device)
    if warm and (HERE / "pretrained_backbone.pt").exists():
        try: model.load_state_dict(torch.load(HERE / "pretrained_backbone.pt", map_location=device, weights_only=True), strict=True)
        except Exception: pass
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(Subset(ds, tr_idx), batch_size=32, shuffle=True, collate_fn=collate)
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
    model.eval(); pred_log, true_log = [], []
    vl = DataLoader(Subset(ds, va_idx), batch_size=64, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for b, yy in vl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            pred_log += (out["amp"].cpu()*std + mean).tolist()   # log1p(Ps) 空间
            true_log += yy.tolist()                              # 已是 log1p(Ps)
    pl, tl_ = np.array(pred_log), np.array(true_log)
    # 鲁棒指标: log 空间 R² (训练目标空间, 抗重尾) + Spearman 秩相关 + 线性 R²
    def r2(a, b): ss=((a-b)**2).sum(); st=((a-a.mean())**2).sum()+1e-9; return float(1-ss/st)
    from scipy.stats import spearmanr
    sp = float(spearmanr(tl_, pl).correlation)
    pl_lin, tl_lin = np.expm1(pl), np.expm1(tl_)
    return {"R2_log": r2(tl_, pl), "spearman": sp, "R2_linear": r2(tl_lin, pl_lin),
            "MAE_uC": float(np.abs(tl_lin-pl_lin).mean())}


def train_final(ds, device, epochs=200):
    """生产模型: 用全部数据训练, 保存权重 + 归一化 (供推理)。"""
    y = ds.y; mean = float(y.mean()); std = float(y.std().clamp(min=1e-3))
    md, sd = torch.tensor(mean, device=device), torch.tensor(std, device=device)
    model = LatentFEModel().to(device)
    if (HERE/"pretrained_backbone.pt").exists():
        try: model.load_state_dict(torch.load(HERE/"pretrained_backbone.pt", map_location=device, weights_only=True), strict=True)
        except Exception: pass
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
    for ep in range(epochs):
        model.train()
        for b, yy in tl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            yn = (yy.to(device) - md) / sd
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            loss = ((out["amp"] - yn) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
    torch.save(model.state_dict(), HERE / "tier2_ps.pt")
    json.dump({"target": "log1p(Ps[uC/cm2])", "mean": mean, "std": std,
               "n_train": len(ds), "n_smidt": int(ds.is_smidt.sum()),
               "n_ricci": int((~ds.is_smidt).sum()),
               "sources": "Smidt 2020 (ferroelectrics) + Ricci 2024 (ferroelectrics_ext, reused via MPContribs)",
               "cv_5fold_spearman": 0.49, "cv_5fold_logR2": 0.09,
               "head": "amp (invariant scalar)", "inference": "Ps = expm1(amp*std + mean)"},
              open(HERE / "tier2_ps_norm.json", "w"), indent=2)
    print(f"saved production Ps model: tier2_ps.pt + tier2_ps_norm.json (trained on all {len(ds)})", flush=True)


def run_parity(ds, device, epochs=200, seed=0):
    """80/20 划分: 训练后在留出集预测, 保存 (真值 Ps, 预测 Ps) 供 parity 图。"""
    n = len(ds); nv = int(0.2*n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    va, tr = perm[:nv], perm[nv:]
    y = ds.y; mean=ds.y[tr].mean(); std=ds.y[tr].std().clamp(min=1e-3)
    md, sd = mean.to(device), std.to(device)
    model = LatentFEModel().to(device)
    if (HERE/"pretrained_backbone.pt").exists():
        try: model.load_state_dict(torch.load(HERE/"pretrained_backbone.pt",map_location=device,weights_only=True),strict=True)
        except Exception: pass
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    tl=DataLoader(Subset(ds,tr),batch_size=32,shuffle=True,collate_fn=collate)
    for _ in range(epochs):
        model.train()
        for b,yy in tl:
            b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
            yn=(yy.to(device)-md)/sd
            out=model(b["z"],b["pos"],b["src"],b["dst"],b["vec"],b["batch"],b["n"])
            loss=((out["amp"]-yn)**2).mean(); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        sch.step()
    model.eval(); pred,true,srcs=[],[],[]
    vl=DataLoader(Subset(ds,va),batch_size=64,shuffle=False,collate_fn=collate)
    with torch.no_grad():
        for b,yy in vl:
            b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
            out=model(b["z"],b["pos"],b["src"],b["dst"],b["vec"],b["batch"],b["n"])
            pred += torch.expm1(out["amp"].cpu()*std+mean).tolist(); true += torch.expm1(yy).tolist()
    for i in va: srcs.append(ds.items[i]["source"])
    json.dump({"true_Ps":true,"pred_Ps":pred,"source":srcs}, open(HERE/"ps_parity.json","w"))
    print(f"saved ps_parity.json ({len(true)} held-out)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE/"ps_cache.pt"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--final", action="store_true", help="用全部数据训练生产模型并保存权重")
    ap.add_argument("--parity", action="store_true", help="80/20 留出集 parity 数据")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ds = PsData(args.cache); n = len(ds)
    sm = int(ds.is_smidt.sum()); print(f"N={n} (smidt={sm}, ricci={n-sm}) device={args.device}", flush=True)
    if args.final:
        train_final(ds, args.device); return
    if args.parity:
        run_parity(ds, args.device); return
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    folds = [idx[i::args.folds] for i in range(args.folds)]
    res = {"smidt_only": [], "combined": []}
    for f in range(args.folds):
        va = folds[f]; tr_all = [i for j in range(args.folds) if j != f for i in folds[j]]
        tr_sm = [i for i in tr_all if ds.is_smidt[i]]
        m_sm = train_eval(ds, tr_sm, va, args.device)
        m_co = train_eval(ds, tr_all, va, args.device)
        res["smidt_only"].append(m_sm); res["combined"].append(m_co)
        print(f"fold {f+1}: Smidt-only [R2_log={m_sm['R2_log']:.3f} ρ={m_sm['spearman']:.3f}] | "
              f"+Ricci [R2_log={m_co['R2_log']:.3f} ρ={m_co['spearman']:.3f}]  "
              f"(n_tr {len(tr_sm)}→{len(tr_all)})", flush=True)
    def agg(key, grp): a=np.array([m[key] for m in res[grp]]); return [float(a.mean()), float(a.std())]
    summary = {"n_smidt": sm, "n_ricci": n-sm,
               "smidt_only": {k: agg(k,"smidt_only") for k in ["R2_log","spearman","R2_linear","MAE_uC"]},
               "combined":   {k: agg(k,"combined")   for k in ["R2_log","spearman","R2_linear","MAE_uC"]}}
    summary["delta"] = {k: summary["combined"][k][0]-summary["smidt_only"][k][0] for k in ["R2_log","spearman","R2_linear"]}
    json.dump(summary, open(HERE/"ps_ablation_results.json","w"), indent=2)
    print(f"\n==== Ps regression: reuse ablation (same val per fold) ====", flush=True)
    for k in ["R2_log", "spearman", "R2_linear", "MAE_uC"]:
        s = summary["smidt_only"][k]; c = summary["combined"][k]
        print(f"{k:11}: Smidt-only {s[0]:.3f}±{s[1]:.3f}  →  +Ricci {c[0]:.3f}±{c[1]:.3f}", flush=True)
    print(f"\nΔ(log-R²)={summary['delta']['R2_log']:+.3f}  Δ(Spearman)={summary['delta']['spearman']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
