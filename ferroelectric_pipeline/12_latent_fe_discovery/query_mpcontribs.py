#!/usr/bin/env python3
"""
查询/复用已有高通量铁电数据库 (MPContribs)
=====================================================================
两个已发布数据库 (与本工作同组 Neaton, 同 atomate/pymatgen 工作流):
  - ferroelectrics      : Smidt et al. Sci.Data 2020 (413 条; 本项目训练数据来源)
  - ferroelectrics_ext  : Ricci et al. npj Comput.Mater. 2024 (517 条候选铁电)
合计 ~930 个已算好极化/能量/弛豫结构的铁电体, 可:
  (1) 直接复用 (我们的候选若已被他们算过 → 跳过 DFT);
  (2) 作为额外训练数据 (含真值 Ps/能量, 直接缓解 Tier-2 数据瓶颈);
  (3) 独立交叉验证我们的预测。
VPN 下 SSL 不稳, 内置重试。
用法:
  python query_mpcontribs.py --lookup mp-980057 mp-938
  python query_mpcontribs.py --dump prior_ferroelectrics.json   # 拉全部 930 条 (后台)
"""
import argparse, json, time
import requests

KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
H = {"X-API-KEY": KEY, "accept": "application/json"}
BASE = "https://contribs-api.materialsproject.org"
PROJECTS = ["ferroelectrics_ext", "ferroelectrics"]


def _get(params, retries=12):
    for _ in range(retries):
        try:
            return requests.get(f"{BASE}/contributions/", headers=H, params=params, timeout=50)
        except Exception:
            time.sleep(4)
    return None


def lookup(mpids):
    for m in mpids:
        hit = None
        for proj in PROJECTS:
            r = _get({"project": proj, "identifier": m,
                      "_fields": "identifier,formula,data"})
            if r and r.json().get("data"):
                hit = (proj, r.json()["data"][0]); break
        if hit:
            proj, c = hit; d = c.get("data", {})
            pol = (d.get("Polarization") or {}).get("value") if isinstance(d.get("Polarization"), dict) else d.get("Polarization")
            en = (d.get("Energy|diff") or {}).get("value") if isinstance(d.get("Energy|diff"), dict) else d.get("Energy|diff")
            print(f"{m:12} {c.get('formula',''):14} [{proj}] Ps={pol} μC/cm²  ΔE={en} meV/atom  "
                  f"info={d.get('Information','')}")
        else:
            print(f"{m:12} NOT in either prior database (novel to our search)")


def dump(out):
    rows = []
    for proj in PROJECTS:
        page = 1
        while True:
            r = _get({"project": proj, "_fields": "identifier,formula,data", "_limit": 200, "_page": page})
            if not r:
                break
            d = r.json().get("data", [])
            if not d:
                break
            for c in d:
                da = c.get("data", {})
                def val(x): return x.get("value") if isinstance(x, dict) else x
                rows.append({"mp_id": c["identifier"], "formula": c.get("formula", ""),
                             "project": proj, "Ps_uC_cm2": val(da.get("Polarization")),
                             "energy_diff_meV_atom": val(da.get("Energy|diff")),
                             "info": da.get("Information", "")})
            print(f"  {proj} page {page}: total {len(rows)}", flush=True)
            if len(d) < 200:
                break
            page += 1
    json.dump(rows, open(out, "w"), indent=2)
    print(f"saved {len(rows)} prior ferroelectric entries -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", nargs="+")
    ap.add_argument("--dump")
    args = ap.parse_args()
    if args.lookup:
        lookup(args.lookup)
    if args.dump:
        dump(args.dump)


if __name__ == "__main__":
    main()
