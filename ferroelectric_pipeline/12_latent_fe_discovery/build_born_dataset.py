#!/usr/bin/env python3
"""
预训练数据集: MP 声子库的 Born 有效电荷 Z* (~1.4 万材料)
=====================================================================
从本地缓存的 MP phonon delta 表提取 (structure, Z* per atom)。
Z* 是等变 rank-2 张量 (Z*->R Z* R^T), 是 B 头(量子无关极化)的预训练目标:
  Ps = (1/Ω) Σ_i Z*_i · u_i   (沿位移模式积分, 无极化量子歧义)
在 ~1.4 万材料上预训练等变主干, 解决 Smidt 仅 204 例的数据瓶颈。

输出: born_dataset/structures.jsonl + zstar.npz (按行号键)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def build(cache: str, out: Path, max_nsites: int, cap: int):
    from deltalake import DeltaTable
    dt = DeltaTable(cache)
    tbl = dt.to_pyarrow_table(columns=["identifier", "formula_pretty", "born", "structure"])
    born_col = tbl.column("born").to_pylist()
    struct_col = tbl.column("structure").to_pylist()
    ids = tbl.column("identifier").to_pylist()
    forms = tbl.column("formula_pretty").to_pylist()

    out.mkdir(parents=True, exist_ok=True)
    fjsonl = open(out / "structures.jsonl", "w")
    zstars = {}
    kept = 0
    for i in range(tbl.num_rows):
        b = born_col[i]
        if b is None:
            continue
        s = struct_col[i]
        s = json.loads(s) if isinstance(s, str) else s
        nsites = len(s.get("sites", []))
        if nsites == 0 or nsites > max_nsites:
            continue
        z = np.array(b, dtype=np.float32)              # (N,3,3)
        if z.shape != (nsites, 3, 3):
            continue
        fjsonl.write(json.dumps({"idx": kept, "mp_id": ids[i] or f"row{i}",
                                 "formula": forms[i] or "", "structure": s}) + "\n")
        zstars[str(kept)] = z
        kept += 1
        if cap and kept >= cap:
            break
    fjsonl.close()
    np.savez(out / "zstar.npz", **zstars)
    print(f"Born pretraining set: {kept} materials (nsites<= {max_nsites})")
    print(f"output -> {out}/structures.jsonl, zstar.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/share/home/caiby/mp_datasets/build/static-collections/phonon")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "born_dataset")
    ap.add_argument("--max_nsites", type=int, default=20)
    ap.add_argument("--cap", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    build(args.cache, args.out, args.max_nsites, args.cap)


if __name__ == "__main__":
    main()
