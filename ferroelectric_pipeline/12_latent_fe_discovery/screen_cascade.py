#!/usr/bin/env python3
"""
级联大规模筛选: Tier-1 (二分类筛) + Tier-2-freq (软度排序) → 潜在铁电发现清单
=====================================================================
对大批 MP 非极性稳定绝缘体 (排除 1534 训练集) 打两个分:
  Tier-1 prob  : 是否有 Γ 软模 (失稳概率)
  Tier-2 freq  : 预测的最软光学模频率 (越负=失稳越强) → 用于精排
输出按预测软度排序的发现清单, 并标注新颖性 (是否双钙钛矿)。

用法: conda activate fe_dft && python screen_cascade.py --pool 1500 --top 25
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


def tier2_meanstd():
    """复算 Tier-2-freq 训练时的 target mean/std (seed-0 划分, 与 train_tier2_freq 一致)。"""
    items = torch.load(HERE / "softmode_graph_cache.pt", weights_only=False)
    y = torch.tensor([float(np.clip(it["sfreq"], -8, 12)) for it in items])
    n = len(items); nv = max(1, int(0.15 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    ytr = y[perm[nv:]]
    return float(ytr.mean()), float(ytr.std().clamp(min=1e-3))


def load(model_path):
    m = LatentFEModel(); m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True)); m.eval()
    return m


def score(model, struct):
    z, pos, src, dst, vec = build_graph(struct)
    if len(src) == 0:
        return None
    with torch.no_grad():
        out = model(torch.tensor(z), torch.tensor(pos), torch.tensor(src),
                    torch.tensor(dst), torch.tensor(vec),
                    torch.zeros(len(z), dtype=torch.long), 1)
    return out


def is_double_perovskite(formula):
    import re
    # 粗判: A2BB'O6 型
    return bool(re.search(r"O6$", formula)) and formula.count("(") <= 1 and "2" in formula


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    from mp_api.client import MPRester
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    exclude = {p.stem for p in (HERE / "modes_data").glob("mp-*.npz")}
    mean, std = tier2_meanstd()
    print(f"tier2 target mean={mean:.2f} std={std:.2f} | excluding {len(exclude)} training", flush=True)

    with MPRester(API_KEY) as m:
        docs = m.materials.summary.search(
            energy_above_hull=(0, 0.05), band_gap=(0.3, 7.0), num_sites=(2, 16),
            theoretical=False,
            fields=["material_id", "formula_pretty", "structure", "symmetry",
                    "energy_above_hull", "band_gap", "nsites"])
    print(f"MP pool: {len(docs)} stable insulators", flush=True)

    t1 = load(HERE / "tier1_instability.pt")
    t2 = load(HERE / "tier2_freq.pt")

    rows = []
    seen = 0
    for d in docs:
        mid = str(d.material_id)
        if mid in exclude:
            continue
        try:
            pg = SpacegroupAnalyzer(d.structure, symprec=0.1).get_point_group_symbol()
        except Exception:
            continue
        if pg in POLAR_PG:
            continue
        seen += 1
        if seen > args.pool:
            break
        o1 = score(t1, d.structure); o2 = score(t2, d.structure)
        if o1 is None or o2 is None:
            continue
        prob = float(torch.sigmoid(o1["logit"])[0])
        freq = float(o2["amp"][0]) * std + mean
        rows.append({"mp_id": mid, "formula": d.formula_pretty, "point_group": pg,
                     "spacegroup": d.symmetry.symbol, "band_gap": round(d.band_gap, 2),
                     "nsites": d.nsites, "e_above_hull": round(d.energy_above_hull, 4),
                     "tier1_prob": round(prob, 3), "tier2_pred_freq_THz": round(freq, 3),
                     "double_perovskite": is_double_perovskite(d.formula_pretty)})
    # 发现清单: Tier-1 判失稳 (prob>0.5), 按预测软度 (freq 升序) 排
    cand = [r for r in rows if r["tier1_prob"] > 0.5]
    cand.sort(key=lambda x: x["tier2_pred_freq_THz"])
    out = HERE / "tier1_screen"; out.mkdir(exist_ok=True)
    json.dump(rows, open(out / "cascade_all_scored.json", "w"), indent=2)
    json.dump(cand, open(out / "discovery_list.json", "w"), indent=2)

    novel = [r for r in cand if not r["double_perovskite"]]
    print(f"\nscored {len(rows)} non-polar materials | predicted unstable (latent-FE): {len(cand)}", flush=True)
    print(f"  of which non-double-perovskite (novel families): {len(novel)}", flush=True)
    print(f"\nTop {min(args.top,len(cand))} latent-FE candidates (by predicted softness):")
    print(f"{'mp_id':<12}{'formula':<14}{'PG':<7}{'pred_freq':>10}{'t1_prob':>8}{'novel':>7}")
    for r in cand[:args.top]:
        print(f"{r['mp_id']:<12}{r['formula']:<14}{r['point_group']:<7}"
              f"{r['tier2_pred_freq_THz']:>10}{r['tier1_prob']:>8}{'':>4}{'*' if not r['double_perovskite'] else ''}")
    print(f"\n-> discovery_list.json ({len(cand)}), cascade_all_scored.json ({len(rows)})", flush=True)


if __name__ == "__main__":
    main()
