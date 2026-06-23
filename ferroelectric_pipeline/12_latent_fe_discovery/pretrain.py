#!/usr/bin/env python3
"""
预训练: 在 MP 声子库 Born 有效电荷上训练等变主干 (解决数据瓶颈)
=====================================================================
目标: 每原子对称 Born 电荷 Z* (0e+2e 等变张量)。在 ~1万材料上学习丰富的等变
局域电响应表示, 供下游 (Smidt 极性软模 + Ps) 微调迁移。

用法: conda activate fe_dft && python pretrain.py --cap 2500 --epochs 60 --device cpu
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from model import LatentFEModel
from train import build_graph

HERE = Path(__file__).parent


class BornData(Dataset):
    def __init__(self, ds_dir: Path, cap=0, cache=True):
        from pymatgen.core import Structure
        recs = [json.loads(l) for l in open(ds_dir / "structures.jsonl")]
        if cap:
            recs = recs[:cap]
        z = np.load(ds_dir / "zstar.npz")
        cpath = ds_dir / f"graph_cache_{cap}.pt"
        if cache and cpath.exists():
            self.items = torch.load(cpath, weights_only=False); return
        self.items = []
        for r in recs:
            try:
                st = Structure.from_dict(r["structure"])
                zg, pos, src, dst, vec = build_graph(st)
                if len(src) == 0:
                    continue
                Zs = z[str(r["idx"])]                          # (N,3,3)
                Zsym = 0.5 * (Zs + np.transpose(Zs, (0, 2, 1)))  # 对称部分
                self.items.append({
                    "z": torch.tensor(zg), "pos": torch.tensor(pos),
                    "src": torch.tensor(src), "dst": torch.tensor(dst),
                    "vec": torch.tensor(vec),
                    "Zsym": torch.tensor(Zsym, dtype=torch.float32)})
            except Exception:
                continue
        if cache:
            torch.save(self.items, cpath)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    z, pos, src, dst, vec, Z = [], [], [], [], [], []
    off = 0
    for it in batch:
        n = it["z"].shape[0]
        z.append(it["z"]); pos.append(it["pos"])
        src.append(it["src"] + off); dst.append(it["dst"] + off)
        vec.append(it["vec"]); Z.append(it["Zsym"]); off += n
    return {"z": torch.cat(z), "pos": torch.cat(pos), "src": torch.cat(src),
            "dst": torch.cat(dst), "vec": torch.cat(vec), "Zsym": torch.cat(Z),
            "batch": None, "n": len(batch)}


@torch.no_grad()
def zstar_r2(model, loader, device):
    ct = model.ct
    yt, yp = [], []
    model.eval()
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        # 占位 batch (Z* 是每原子量, 不需池化)
        nat = b["z"].shape[0]
        batch = torch.zeros(nat, dtype=torch.long, device=device)
        out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], batch, 1)
        pred = ct.to_cartesian(out["zstar"].cpu())              # (N,3,3)
        yp.append(pred.numpy()); yt.append(b["Zsym"].cpu().numpy())
    yp = np.concatenate(yp); yt = np.concatenate(yt)
    ss_res = ((yt - yp) ** 2).sum(); ss_tot = ((yt - yt.mean()) ** 2).sum() + 1e-9
    return 1 - ss_res / ss_tot, np.abs(yt - yp).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    ds = BornData(HERE / "born_dataset", cap=args.cap)
    print(f"Born pretrain N={len(ds)} device={args.device}", flush=True)
    nv = max(1, int(0.1 * len(ds)))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
    tl = DataLoader(Subset(ds, perm[nv:]), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(Subset(ds, perm[:nv]), batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = LatentFEModel().to(args.device)
    ct = model.ct
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = -1e9
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            nat = b["z"].shape[0]
            batch = torch.zeros(nat, dtype=torch.long, device=args.device)
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], batch, 1)
            target = ct.from_cartesian(b["Zsym"]).to(args.device)   # (N,6)
            loss = ((out["zstar"] - target) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            r2, mae = zstar_r2(model, vl, args.device)
            print(f"ep{ep+1:3d} loss={loss.item():.4f} | Z* R2={r2:.3f} MAE={mae:.3f}", flush=True)
            if r2 > best:
                best = r2
                torch.save(model.state_dict(), HERE / "pretrained_backbone.pt")
    print(f"best Z* R2={best:.3f} -> saved pretrained_backbone.pt", flush=True)


if __name__ == "__main__":
    main()
