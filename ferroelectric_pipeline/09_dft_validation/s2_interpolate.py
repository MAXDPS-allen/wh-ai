#!/usr/bin/env python3
"""
Stage 2: 非极性母相 ↔ 极性相 之间生成线性插值结构
=====================================================================
复刻论文: 在非极性 (λ=0) 与极性 (λ=1) 之间生成 N=8 个中间结构, 形成
畸变路径 (共 10 个结构: nonpolar + 8 interp + polar)。沿路径计算能量与
极化, 用于势阱/能垒分析与极化分支跟踪。

关键: 两端结构须有一一对应的原子映射 (同原子数、同顺序)。pymatgen 的
Structure.interpolate 处理晶格与分数坐标的线性插值, 并可选最近镜像对齐。

输出: {cid}/path/image_00.json ... image_09.json  (00=非极性, 09=极性)

依赖: pymatgen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymatgen.core import Structure


def interpolate_path(nonpolar: Structure, polar: Structure, n_interp: int = 8,
                     autosort_tol: float = 0.5):
    """返回从非极性到极性的 (n_interp + 2) 个结构。
    autosort_tol: 自动匹配两端原子顺序的容差 (Å)。"""
    # interpolate 返回 nimages+1 个结构 (含两端)
    images = nonpolar.interpolate(polar, nimages=n_interp + 1,
                                  interpolate_lattices=True,
                                  autosort_tol=autosort_tol)
    return images


def run(cid_dir: Path, n_interp: int = 8):
    nonpolar = Structure.from_dict(json.load(open(cid_dir / "nonpolar.json")))
    polar = Structure.from_dict(json.load(open(cid_dir / "polar.json")))

    if len(nonpolar) != len(polar):
        raise ValueError(f"原子数不一致: nonpolar={len(nonpolar)} polar={len(polar)}; "
                         "需先用群-子群超胞对齐")

    path_dir = cid_dir / "path"
    path_dir.mkdir(exist_ok=True)
    try:
        images = interpolate_path(nonpolar, polar, n_interp)
    except Exception:
        # autosort 失败时退回无排序 (要求两端原子顺序已对应)
        images = interpolate_path(nonpolar, polar, n_interp, autosort_tol=0)

    for i, img in enumerate(images):
        json.dump(img.as_dict(), open(path_dir / f"image_{i:02d}.json", "w"))

    meta = {"n_images": len(images), "n_interp": n_interp,
            "natoms": len(polar), "formula": polar.composition.reduced_formula}
    json.dump(meta, open(path_dir / "path_meta.json", "w"), indent=2)
    print(f"[{cid_dir.name}] 生成 {len(images)} 个路径结构 (image_00=非极性 ... "
          f"image_{len(images)-1:02d}=极性)")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Generate nonpolar->polar interpolation path")
    ap.add_argument("cid_dir", type=Path, help="Stage 1 输出的候选目录 (含 polar/nonpolar.json)")
    ap.add_argument("--n_interp", type=int, default=8)
    args = ap.parse_args()
    run(args.cid_dir, args.n_interp)


if __name__ == "__main__":
    main()
