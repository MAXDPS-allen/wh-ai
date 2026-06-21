#!/usr/bin/env python3
"""
结构特征化 (供物性回归模型)
=============================================
提供两种特征:
  1. graph_from_structure : pymatgen Structure -> 原子图 (节点=原子, 边=近邻),
     供 CGCNN / e3nn 等变 GNN 使用 (production)。
  2. vector_from_structure : 组分 + 晶格的物理描述符向量,
     供轻量 sklearn 基线模型使用 (可在 CPU 快速验证)。

依赖: pymatgen (graph + structure 解析); numpy。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from pymatgen.core import Structure, Element
    from pymatgen.analysis.local_env import CrystalNN
    _HAS_PMG = True
except Exception:
    _HAS_PMG = False


# ---------------------------------------------------------------------------
# 轻量向量特征 (组分统计 + 晶格)，无重依赖
# ---------------------------------------------------------------------------
def vector_from_structure(struct) -> np.ndarray:
    """组分加权的元素物性统计 + 晶格几何 -> 固定长度向量。"""
    if isinstance(struct, (str, Path)):
        struct = Structure.from_dict(json.load(open(struct)))
    elif isinstance(struct, dict):
        struct = Structure.from_dict(struct)

    comp = struct.composition.fractional_composition
    props = {"Z": [], "X": [], "row": [], "group": [], "atomic_mass": [], "atomic_radius": []}
    weights = []
    for el, frac in comp.items():
        e = Element(el.symbol)
        props["Z"].append(e.Z)
        props["X"].append(e.X if e.X else 0.0)
        props["row"].append(e.row)
        props["group"].append(e.group)
        props["atomic_mass"].append(float(e.atomic_mass))
        props["atomic_radius"].append(float(e.atomic_radius) if e.atomic_radius else 1.0)
        weights.append(frac)
    weights = np.array(weights)

    feats = []
    for k in props:
        v = np.array(props[k], dtype=float)
        feats += [
            float(np.sum(v * weights)),          # 加权平均
            float(v.max() - v.min()),            # 极差
            float(np.sqrt(np.sum(weights * (v - np.sum(v * weights)) ** 2))),  # 加权标准差
        ]

    lat = struct.lattice
    feats += [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma,
              lat.volume, lat.volume / len(struct), float(len(struct)),
              float(len(comp)), float(struct.density)]
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# 图特征 (供 GNN)
# ---------------------------------------------------------------------------
def graph_from_structure(struct, cutoff: float = 6.0, max_neigh: int = 12):
    """返回 (atom_z, edge_index, edge_vec, edge_len)。
    edge_vec 为笛卡尔位移 (供等变 GNN)，edge_len 为距离 (供不变 GNN)。"""
    if isinstance(struct, (str, Path)):
        struct = Structure.from_dict(json.load(open(struct)))
    elif isinstance(struct, dict):
        struct = Structure.from_dict(struct)

    atom_z = np.array([site.specie.Z for site in struct], dtype=np.int64)
    src, dst, vecs, lens = [], [], [], []
    all_neigh = struct.get_all_neighbors(cutoff)
    for i, neighbors in enumerate(all_neigh):
        neighbors = sorted(neighbors, key=lambda n: n.nn_distance)[:max_neigh]
        for n in neighbors:
            j = n.index
            disp = n.coords - struct[i].coords
            src.append(i); dst.append(j)
            vecs.append(disp); lens.append(n.nn_distance)
    edge_index = np.array([src, dst], dtype=np.int64)
    edge_vec = np.array(vecs, dtype=np.float32) if vecs else np.zeros((0, 3), np.float32)
    edge_len = np.array(lens, dtype=np.float32) if lens else np.zeros((0,), np.float32)
    return atom_z, edge_index, edge_vec, edge_len


if __name__ == "__main__":
    import sys
    if not _HAS_PMG:
        print("pymatgen 未安装，请先 source environment/setup_env.sh"); sys.exit(1)
    ds = Path(__file__).parent / "dataset"
    sf = next((ds / "structures").glob("*_polar.json"))
    v = vector_from_structure(sf)
    z, ei, ev, el = graph_from_structure(sf)
    print(f"vector feature dim: {v.shape}")
    print(f"graph: {len(z)} atoms, {ei.shape[1]} edges, edge_vec {ev.shape}")
