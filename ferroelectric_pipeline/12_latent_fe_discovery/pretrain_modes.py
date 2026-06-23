#!/usr/bin/env python3
"""
预训练: 在 MP 声子库 Γ 软模本征位移上训练等变 mode 头 (正确的迁移信号)
=====================================================================
与 Born 电荷不同, 这直接预训练 **mode 头** 去预测"哪个畸变会软化"——
即 Γ 点最低光学模的本征位移 (软模)。这与下游 Smidt 极性软模是同一任务,
应当真正迁移。

目标: 每材料 Γ 点最软光学模的本征位移场 (N,3), 等变 l=1。
损失: sign-invariant cos^2 (方向), 与 train.py 一致。

数据: modes_data/<id>.npz (fetch_phonon_modes.py 拉取)。
图缓存需 pymatgen (fe_dft 建), 训练在 fe_gpu/GPU。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from model import LatentFEModel
from train import build_graph, _per_struct_cos2

HERE = Path(__file__).parent


def softest_optical_mode(freqs, eigdisp):
    """freqs (nbands,), eigdisp (nbands, natoms, 3). 去掉 3 个声学模(|freq|最小),
    取剩余中 signed-freq 最低者 = 最软光学模。返回 (natoms,3)。"""
    nb = len(freqs)
    acoustic = set(np.argsort(np.abs(freqs))[:3].tolist())   # 3 个最接近 0 的 = 声学
    opt = [i for i in range(nb) if i not in acoustic]
    if not opt:
        opt = list(range(nb))
    soft = opt[int(np.argmin(freqs[opt]))]                   # 最低(含虚频)光学模
    return eigdisp[soft], float(freqs[soft])


class SoftModeData(Dataset):
    def __init__(self, ds_dir: Path, cache=True):
        from pymatgen.core import Structure
        files = sorted(ds_dir.glob("*.npz"))
        cpath = ds_dir.parent / "softmode_graph_cache.pt"
        if cache and cpath.exists():
            self.items = torch.load(cpath, weights_only=False); return
        self.items = []
        for f in files:
            try:
                d = np.load(f, allow_pickle=True)
                st = Structure.from_dict(json.loads(str(d["structure"])))
                fr = d["gamma_freqs"]; ed = d["gamma_eigdisp"]
                if ed.shape[1] != len(st):
                    continue
                mode, sfreq = softest_optical_mode(fr, ed)        # (N,3)
                z, pos, src, dst, vec = build_graph(st)
                if len(src) == 0:
                    continue
                self.items.append({
                    "z": torch.tensor(z), "pos": torch.tensor(pos),
                    "src": torch.tensor(src), "dst": torch.tensor(dst),
                    "vec": torch.tensor(vec),
                    "mode": torch.tensor(mode, dtype=torch.float32),
                    "sfreq": float(sfreq)})
            except Exception:
                continue
        if cache:
            torch.save(self.items, cpath)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    z, pos, src, dst, vec, mode, bidx = [], [], [], [], [], [], []
    off = 0
    for g, it in enumerate(batch):
        n = it["z"].shape[0]
        z.append(it["z"]); pos.append(it["pos"])
        src.append(it["src"] + off); dst.append(it["dst"] + off)
        vec.append(it["vec"]); mode.append(it["mode"])
        bidx.append(torch.full((n,), g, dtype=torch.long)); off += n
    return {"z": torch.cat(z), "pos": torch.cat(pos), "src": torch.cat(src),
            "dst": torch.cat(dst), "vec": torch.cat(vec), "mode": torch.cat(mode),
            "batch": torch.cat(bidx), "n": len(batch)}


@torch.no_grad()
def mode_cos(model, loader, device):
    model.eval(); cs = []
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
        cos2, _, _ = _per_struct_cos2(out["mode"], b["mode"], b["batch"], b["n"])
        cs += cos2.sqrt().cpu().tolist()
    return float(np.mean(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    ds = SoftModeData(HERE / "modes_data")
    print(f"soft-mode pretrain N={len(ds)} device={args.device}", flush=True)
    nv = max(1, int(0.1 * len(ds)))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
    tl = DataLoader(Subset(ds, perm[nv:]), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(Subset(ds, perm[:nv]), batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = LatentFEModel().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = -1
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["z"], b["pos"], b["src"], b["dst"], b["vec"], b["batch"], b["n"])
            cos2, pn, tn = _per_struct_cos2(out["mode"], b["mode"], b["batch"], b["n"])
            loss = (1 - cos2).mean() + 0.05 * ((pn - tn) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            mc = mode_cos(model, vl, args.device)
            print(f"ep{ep+1:3d} loss={loss.item():.3f} val_mode_cos={mc:.3f}", flush=True)
            if mc > best:
                best = mc
                torch.save(model.state_dict(), HERE / "pretrained_softmode.pt")
    print(f"best val soft-mode cos={best:.3f} -> pretrained_softmode.pt", flush=True)


if __name__ == "__main__":
    main()
