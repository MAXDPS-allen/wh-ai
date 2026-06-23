#!/usr/bin/env python3
"""
DFT 验证 Tier-1 候选: GPU 有限位移声子 (2x2x2) → 确认软模
=====================================================================
对 2 个高分(预测失稳) + 1 个低分(对照) 候选, 用 phonopy 有限位移 + GPU 静态力
算 Γ/布里渊区声子, 检查最小频率 (<0 = 失稳, 确认 Tier-1)。比 CPU-DFPT 快且可靠。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "09_dft_validation"))
import s4_run_cluster as s4, s6_phonon as s6

CANDS = [("BCl3_mp-23184", 1.0), ("Ba2CaMoO6_mp-19403", 1.0), ("Ac2O3_mp-11107", 0.0)]
SUP = (1, 1, 1)
VER = HERE / "tier1_screen" / "verify"


def main():
    results = []
    for cid, score in CANDS:
        cdir = VER / cid
        print(f"\n=== {cid} (Tier-1 score {score}) ===", flush=True)
        s6.generate(cdir, supercell=SUP)
        s4.run(cdir / "phonon" / "disps", ["disp_*"], nproc=1, poll=20, backend="gpu")
        r = s6.collect(cdir, supercell=SUP)
        r["tier1_score"] = score
        r["DFT_confirms"] = (r["dynamically_stable"] is False) == (score > 0.5)
        results.append(r)
    json.dump(results, open(VER / "verification_gamma.json", "w"), indent=2, default=str)
    print("\n==== Tier-1 DFT verification ====", flush=True)
    for r in results:
        print(f"{r['cid']:22} score={r['tier1_score']} min_freq={r['min_frequency_THz']:.2f}THz "
              f"unstable={not r['dynamically_stable']} confirms_tier1={r['DFT_confirms']}", flush=True)
    ok = sum(1 for r in results if r["DFT_confirms"])
    print(f"\nDFT confirms Tier-1 on {ok}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
