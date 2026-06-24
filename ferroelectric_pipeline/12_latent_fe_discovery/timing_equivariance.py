#!/usr/bin/env python3
"""
(1) 等变性验证: 旋转输入 → 标量不变、矢量/张量随旋转。
(2) 计时对比: ML 推理 per-material vs DFT 验证 per-material (解析 VASP 墙钟)。
用法: conda run -n fe_dft python timing_equivariance.py
"""
import json, re, time
from pathlib import Path
import numpy as np
import torch

from model import LatentFEModel
from train import build_graph

HERE = Path(__file__).parent


# ---------------- (1) 等变性 ----------------
def rand_rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R))
    if torch.det(Q) < 0: Q[:, 0] = -Q[:, 0]
    return Q


def check_equivariance():
    items = torch.load(HERE / "softmode_graph_cache.pt", weights_only=False)
    it = items[0]
    m = LatentFEModel(); m.load_state_dict(torch.load(HERE / "tier1_instability.pt", map_location="cpu", weights_only=True)); m.eval()
    R = rand_rotation(1)
    z, pos, src, dst, vec = it["z"], it["pos"], it["src"], it["dst"], it["vec"]
    batch = torch.zeros(len(z), dtype=torch.long)
    with torch.no_grad():
        o1 = m(z, pos, src, dst, vec, batch, 1)
        o2 = m(z, pos @ R.T, src, dst, vec @ R.T, batch, 1)
    res = {}
    res["logit_invariance_err"] = float((o1["logit"] - o2["logit"]).abs().max())
    res["amp_invariance_err"] = float((o1["amp"] - o2["amp"]).abs().max())
    if "mode" in o1:  # 矢量输出应随旋转
        rotated = o1["mode"] @ R.T
        res["mode_equivariance_err"] = float((rotated - o2["mode"]).abs().max())
    print("=== Equivariance (random rotation) ===")
    for k, v in res.items(): print(f"  {k}: {v:.2e}")
    return res


# ---------------- (2) 计时 ----------------
def time_ml_inference(n_materials=200, device="cpu"):
    items = torch.load(HERE / "softmode_graph_cache.pt", weights_only=False)[:n_materials]
    m = LatentFEModel().to(device)
    m.load_state_dict(torch.load(HERE / "tier1_instability.pt", map_location=device, weights_only=True))
    m.eval()
    # 计时含图构建已在缓存; 这里测前向 (推理). 另测从结构构图的时间.
    t0 = time.perf_counter()
    with torch.no_grad():
        for it in items:
            b = {k: (it[k].to(device) if torch.is_tensor(it[k]) else it[k]) for k in ("z","pos","src","dst","vec")}
            m(b["z"], b["pos"], b["src"], b["dst"], b["vec"], torch.zeros(len(it["z"]),dtype=torch.long,device=device), 1)
    fwd = (time.perf_counter() - t0) / len(items)
    return {"forward_per_material_ms": fwd * 1e3, "device": device, "n": len(items)}


def parse_walltime(outcar: Path):
    """返回该计算的墙钟秒 (Elapsed time)。"""
    try:
        txt = outcar.read_text(errors="ignore")
    except Exception:
        return None
    m = re.search(r"Elapsed time \(sec\):\s*([0-9.]+)", txt)
    if m: return float(m.group(1))
    m = re.search(r"Total CPU time used \(sec\):\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def dft_walltimes():
    """统计已完成 DFT 计算的墙钟: 按材料聚合。"""
    roots = {
        "SrAlGeH (full FE proof)": HERE.parent / "test_run" / "SrAlGeH_mp-980057",
        "Ba2CaMoO6 (cascade verify)": HERE / "tier1_screen" / "verify" / "Ba2CaMoO6_mp-19403",
        "Ac2O3 (cascade verify)": HERE / "tier1_screen" / "verify" / "Ac2O3_mp-11107",
    }
    out = {}
    for name, root in roots.items():
        if not root.exists(): continue
        times = [parse_walltime(oc) for oc in root.rglob("OUTCAR")]
        times = [t for t in times if t]
        if times:
            out[name] = {"n_vasp_calcs": len(times), "total_wall_sec": sum(times),
                         "total_wall_hr": sum(times) / 3600, "per_calc_sec_mean": np.mean(times)}
    return out


def main():
    eq = check_equivariance()
    print("\n=== ML inference timing ===")
    ml = time_ml_inference(device="cpu")
    print(f"  forward: {ml['forward_per_material_ms']:.2f} ms/material (CPU, n={ml['n']})")
    print("\n=== DFT validation wall time (completed runs) ===")
    dft = dft_walltimes()
    for name, d in dft.items():
        print(f"  {name}: {d['n_vasp_calcs']} VASP calcs, {d['total_wall_hr']:.2f} h total "
              f"({d['per_calc_sec_mean']/60:.1f} min/calc avg)")
    # 对比/外推
    if dft:
        per_mat_hr = np.mean([d["total_wall_hr"] for d in dft.values()])
        ml_ms = ml["forward_per_material_ms"]
        speedup = (per_mat_hr * 3600 * 1e3) / ml_ms
        print(f"\n  ML inference: ~{ml_ms:.1f} ms/material")
        print(f"  DFT validation: ~{per_mat_hr:.1f} h/material (mean over above)")
        print(f"  → ML is ~{speedup:.0e}× faster per material")
    json.dump({"equivariance": eq, "ml_inference": ml, "dft_walltimes": dft},
              open(HERE / "timing_results.json", "w"), indent=2)
    print("\nsaved timing_results.json")


if __name__ == "__main__":
    main()
