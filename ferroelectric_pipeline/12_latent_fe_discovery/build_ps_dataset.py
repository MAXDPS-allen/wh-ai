#!/usr/bin/env python3
"""
构建 Ps 回归数据集 (复用已有高通量结果)
=====================================================================
合并两个来源的 自发极化 Ps 真值 + 极性结构, 建图缓存:
  - Smidt 2020 (本地 workflow_data.json.gz): polarization_change_norm
  - Ricci 2024 (MPContribs ferroelectrics_ext): Polarization, 结构经 MP API 按 mp-id 取
目的: 把"复用他人 Ps 标签"变成可训练数据, 攻克我们 Tier-2 的 Ps 数据瓶颈。
输出: ps_cache.pt (每项: 图 + Ps + source 标签)
"""
import gzip, json, time
from pathlib import Path
import numpy as np
import torch
import requests
from train import build_graph

HERE = Path(__file__).parent
KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
H = {"X-API-KEY": KEY, "accept": "application/json"}


def _get(url, params, retries=12):
    for _ in range(retries):
        try:
            return requests.get(url, headers=H, params=params, timeout=50)
        except Exception:
            time.sleep(4)
    return None


CB = "https://contribs-api.materialsproject.org"


def ricci_entries():
    """ferroelectrics_ext: identifier → {Ps μC/cm², polar 结构 id}。结构与 Ps 同源, 保证配对。"""
    out = {}; page = 1
    while page <= 30:
        prev = len(out)
        r = _get(f"{CB}/contributions/",
                 {"project": "ferroelectrics_ext",
                  "_fields": "identifier,data.Polarization.value,structures",
                  "_limit": 200, "_skip": (page - 1) * 200})
        if not r: break
        d = r.json().get("data", [])
        if not d: break
        for c in d:
            ps = (((c.get("data") or {}).get("Polarization") or {}).get("value"))
            sid = None
            for s in (c.get("structures") or []):
                if "polar" in s.get("name", "").lower() and "nonpolar" not in s.get("name", "").lower():
                    sid = s["id"]; break
            if ps is not None and sid:
                out[c["identifier"]] = {"Ps": float(ps), "sid": sid}
        print(f"  ricci page {page}: {len(out)}", flush=True)
        if len(d) < 200 or len(out) == prev:
            break
        page += 1
    return out


def fetch_structures(entries):
    """按 contribs 结构 id 取极性结构 (与 Ps 同源配对), 增量缓存。"""
    from pymatgen.core import Structure
    cache_f = HERE / "ricci_structs_cache.json"
    raw = json.load(open(cache_f)) if cache_f.exists() else {}
    todo = [(mid, e["sid"]) for mid, e in entries.items() if mid not in raw]
    for n, (mid, sid) in enumerate(todo):
        for _ in range(10):
            try:
                r = requests.get(f"{CB}/structures/{sid}/",
                                 headers=H, timeout=30,
                                 params={"_fields": "lattice,sites"})
                if r.status_code == 200:
                    raw[mid] = r.json(); break
            except Exception:
                pass
            time.sleep(2)
        if n % 25 == 0:
            json.dump(raw, open(cache_f, "w")); print(f"  structures {len(raw)}", flush=True)
    json.dump(raw, open(cache_f, "w"))
    structs = {}
    for mid, sd in raw.items():
        try:
            sd = dict(sd); sd["@module"] = "pymatgen.core.structure"; sd["@class"] = "Structure"
            structs[mid] = Structure.from_dict(sd)
        except Exception:
            continue
    return structs




def main():
    items = []
    # Smidt (local)
    data = json.load(gzip.open(HERE.parent.parent / "data_files" / "workflow_data.json.gz"))
    from pymatgen.core import Structure
    n_sm = 0
    for r in data:
        if r.get("workflow_status") != "COMPLETED":
            continue
        ps = r.get("polarization_change_norm")
        if ps is None:
            continue
        try:
            st = Structure.from_dict(r["structures"][-1])   # polar endpoint
            z, pos, src, dst, vec = build_graph(st)
            if len(src) == 0: continue
            items.append({"z": torch.tensor(z), "pos": torch.tensor(pos),
                          "src": torch.tensor(src), "dst": torch.tensor(dst),
                          "vec": torch.tensor(vec), "Ps": float(ps), "source": "smidt"})
            n_sm += 1
        except Exception:
            continue
    print(f"Smidt: {n_sm} entries", flush=True)

    # Ricci (reuse)
    rps = ricci_entries()
    print(f"Ricci Ps entries: {len(rps)}", flush=True)
    structs = fetch_structures(rps)
    n_ri = 0
    for mid, e in rps.items():
        ps = e["Ps"]; st = structs.get(mid)
        if st is None: continue
        try:
            z, pos, src, dst, vec = build_graph(st)
            if len(src) == 0: continue
            items.append({"z": torch.tensor(z), "pos": torch.tensor(pos),
                          "src": torch.tensor(src), "dst": torch.tensor(dst),
                          "vec": torch.tensor(vec), "Ps": float(ps), "source": "ricci"})
            n_ri += 1
        except Exception:
            continue
    print(f"Ricci: {n_ri} entries", flush=True)
    torch.save(items, HERE / "ps_cache.pt")
    print(f"saved ps_cache.pt: {len(items)} total ({n_sm} smidt + {n_ri} ricci)", flush=True)


if __name__ == "__main__":
    main()
