#!/usr/bin/env python3
"""
DFT 验证发现清单中的新颖家族候选 (Γ 声子)
=====================================================================
从大规模级联筛选的发现清单里挑非钙钛矿-氧化物的新颖候选, 取 MP 结构,
做 1x1x1 Γ 有限位移声子, 验证是否真有软模 (确认潜在铁电)。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "09_dft_validation"))
from pymatgen.core import Structure
import s4_run_cluster as s4, s6_phonon as s6

API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
# 新颖家族 top 候选 (mp_id, 预测软度): 氟化物 elpasolite + 层状卤化物
NOVEL = [("mp-1079339", "K2NaMnF6", -2.905), ("mp-23202", "InI", -1.145)]
SUP = (1, 1, 1)
VER = HERE / "tier1_screen" / "verify_novel"


def fetch():
    from mp_api.client import MPRester
    VER.mkdir(parents=True, exist_ok=True)
    with MPRester(API_KEY) as m:
        for mid, name, _ in NOVEL:
            st = m.materials.summary.search(material_ids=[mid], fields=["structure"])[0].structure
            d = VER / f"{name}_{mid}"; d.mkdir(exist_ok=True)
            json.dump(st.as_dict(), open(d / "polar.json", "w"))
            print(f"fetched {name} ({len(st)} atoms)", flush=True)


def main():
    fetch()
    results = []
    for mid, name, pred in NOVEL:
        cdir = VER / f"{name}_{mid}"
        print(f"\n=== {name} ({mid}) pred_freq={pred} ===", flush=True)
        s6.generate(cdir, supercell=SUP)
        s4.run(cdir / "phonon" / "disps", ["disp_*"], nproc=1, poll=20, backend="gpu")
        r = s6.collect(cdir, supercell=SUP)
        r.update({"mp_id": mid, "formula": name, "tier2_pred_freq": pred,
                  "DFT_confirms": (not r["dynamically_stable"])})
        results.append(r)
    json.dump(results, open(VER / "verification_novel.json", "w"), indent=2, default=str)
    print("\n==== novel-family DFT verification ====", flush=True)
    for r in results:
        print(f"{r['formula']:12} pred={r['tier2_pred_freq']:+.2f} "
              f"DFT_min={r['min_frequency_THz']:+.2f}THz unstable={not r['dynamically_stable']} "
              f"confirms={r['DFT_confirms']}", flush=True)


if __name__ == "__main__":
    main()
