#!/usr/bin/env python3
"""
用 Tier-1 失稳分类器筛选 MP 非极性稳定绝缘体 (潜在铁电初筛)
=====================================================================
从 MP 取 非极性(高对称) + 热稳定 + 绝缘 + 小胞 的材料 (排除 1534 个声子训练集),
用 Tier-1 等变模型打"是否有 Γ 软模"分数, 排序输出 top-K 候选 → 送 DFT 验证。

逻辑: 高对称非极性相若有 Γ 极性软模, 则其畸变相是(新)铁电体 → "潜在铁电"。
用法: conda activate fe_dft && python screen_tier1.py --pool 200 --top 8
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from model import LatentFEModel
from train import build_graph

HERE = Path(__file__).parent
API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
POLAR_PG = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}


def petretto_ids():
    return {p.stem for p in (HERE / "modes_data").glob("mp-*.npz")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200, help="从 MP 取多少候选打分")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--model", type=Path, default=HERE / "tier1_instability.pt")
    args = ap.parse_args()

    from mp_api.client import MPRester
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    exclude = petretto_ids()
    print(f"excluding {len(exclude)} training (Petretto) materials", flush=True)

    # 一次 summary 查询 (稳健): 稳定 + 绝缘 + 小胞
    with MPRester(API_KEY) as m:
        docs = m.materials.summary.search(
            energy_above_hull=(0, 0.03), band_gap=(0.5, 6.0), num_sites=(2, 10),
            theoretical=False,
            fields=["material_id", "formula_pretty", "structure", "symmetry",
                    "energy_above_hull", "band_gap", "nsites"])
    print(f"MP returned {len(docs)} stable insulators (small cell)", flush=True)

    # 过滤: 非极性点群 + 不在训练集
    cands = []
    for d in docs:
        mid = str(d.material_id)
        if mid in exclude:
            continue
        try:
            pg = SpacegroupAnalyzer(d.structure, symprec=0.1).get_point_group_symbol()
        except Exception:
            continue
        if pg in POLAR_PG:                       # 只要非极性母相
            continue
        cands.append((d, pg))
        if len(cands) >= args.pool:
            break
    print(f"non-polar candidates to score: {len(cands)}", flush=True)

    # 载入 Tier-1 模型
    model = LatentFEModel(); model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()

    scored = []
    for d, pg in cands:
        try:
            z, pos, src, dst, vec = build_graph(d.structure)
            if len(src) == 0:
                continue
            with torch.no_grad():
                out = model(torch.tensor(z), torch.tensor(pos), torch.tensor(src),
                            torch.tensor(dst), torch.tensor(vec),
                            torch.zeros(len(z), dtype=torch.long), 1)
                score = float(torch.sigmoid(out["logit"])[0])
        except Exception:
            continue
        scored.append({"mp_id": str(d.material_id), "formula": d.formula_pretty,
                       "point_group": pg, "spacegroup": d.symmetry.symbol,
                       "band_gap": round(d.band_gap, 2), "nsites": d.nsites,
                       "e_above_hull": round(d.energy_above_hull, 4),
                       "instability_score": round(score, 4),
                       "structure": d.structure.as_dict()})
    scored.sort(key=lambda x: -x["instability_score"])

    out_dir = HERE / "tier1_screen"; out_dir.mkdir(exist_ok=True)
    # 存全部打分 (不含结构) + top-K 结构
    json.dump([{k: v for k, v in s.items() if k != "structure"} for s in scored],
              open(out_dir / "ranked_all.json", "w"), indent=2)
    top = scored[:args.top]
    for s in top:
        cdir = out_dir / f"{s['formula']}_{s['mp_id']}".replace(" ", "")
        cdir.mkdir(exist_ok=True)
        json.dump(s["structure"], open(cdir / "polar.json", "w"))  # 命名复用 s6 (其实是母相)
    json.dump([{k: v for k, v in s.items() if k != "structure"} for s in top],
              open(out_dir / "top_candidates.json", "w"), indent=2)

    print(f"\nscored {len(scored)} | top {len(top)} latent-FE candidates:", flush=True)
    print(f"{'rank':<5}{'mp_id':<13}{'formula':<12}{'PG':<7}{'gap':>5}{'score':>8}")
    for i, s in enumerate(top):
        print(f"{i+1:<5}{s['mp_id']:<13}{s['formula']:<12}{s['point_group']:<7}"
              f"{s['band_gap']:>5}{s['instability_score']:>8}")
    print(f"\n-> {out_dir}/top_candidates.json (+ per-candidate dirs for DFT)", flush=True)


if __name__ == "__main__":
    main()
