#!/usr/bin/env python3
"""
微调: 从 Born-电荷预训练主干迁移到 Smidt 极性软模 + Ps 任务
=====================================================================
核心实验: 验证"等变迁移学习"是否解决数据瓶颈 (从零训练验证集 mode_cos=随机0.10)。
加载在 ~1万材料 Born 电荷上预训练的主干, 在 204 个 Smidt 母相上微调 mode+Ps 头。
对照: 从零训练 (train.py)。

用法: conda activate fe_dft && python finetune.py --pretrained pretrained_backbone.pt --epochs 200
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import LatentFEModel
from train import FEData, collate, sign_aligned_losses, evaluate

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=Path, default=HERE / "pretrained_backbone.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--freeze_epochs", type=int, default=30, help="先冻结主干只训练头的轮数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ds = FEData(HERE / "dataset")
    nv = max(1, int(0.2 * len(ds)))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed)).tolist()
    tl = DataLoader(Subset(ds, perm[nv:]), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(Subset(ds, perm[:nv]), batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = LatentFEModel().to(args.device)
    if args.pretrained.exists():
        sd = torch.load(args.pretrained, map_location=args.device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        print(f"loaded pretrained backbone: {args.pretrained.name}", flush=True)
    else:
        print("WARNING: no pretrained weights, training from scratch", flush=True)

    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith(("head_mode", "head_ps", "head_scalar"))]
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith(("head_mode", "head_ps", "head_scalar"))]

    def make_opt(backbone_lr):
        return torch.optim.AdamW(
            [{"params": head_params, "lr": args.lr},
             {"params": backbone_params, "lr": backbone_lr}], weight_decay=1e-5)

    opt = make_opt(0.0)                      # 先冻结主干 (lr=0)
    best = -1e9
    for ep in range(args.epochs):
        if ep == args.freeze_epochs:
            opt = make_opt(args.lr * 0.2)    # 解冻主干, 小 lr
            print(f"-- unfreeze backbone at epoch {ep} --", flush=True)
        model.train()
        for b in tl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            loss, _ = sign_aligned_losses(out, b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, args.device)
            print(f"ep{ep+1:3d} mode_cos={m['mode_cos']:.3f} Ps_R2={m['Ps_R2']:.3f} "
                  f"Ps_vec_cos={m['Ps_vec_cos']:.3f}", flush=True)
            if m["mode_cos"] > best:
                best = m["mode_cos"]
                torch.save(model.state_dict(), HERE / "finetuned_model.pt")
    m = evaluate(model, vl, args.device)
    json.dump(m, open(HERE / "finetune_val_metrics.json", "w"), indent=2)
    print("\nFINETUNE final:", json.dumps(m, indent=2), flush=True)
    print("(from-scratch baseline was: mode_cos=0.10, Ps_R2=-0.40)", flush=True)


if __name__ == "__main__":
    main()
