#!/usr/bin/env python3
"""
铁电关键物性多任务回归 (基线模型)
=====================================================================
在 Smidt et al. DFT 数据库上训练回归模型, 直接从结构预测:
  - Ps           自发极化            [μC/cm²]   (log1p 变换, 量纲跨度大)
  - dw_depth     双势阱深度          [meV/atom]
  - path_barrier 切换能垒            [meV/atom]
  - gap_polar    带隙                [eV]
并训练一个 is_switchable 分类头 (高质量可切换铁电)。

基线: 梯度提升 (sklearn), 5-fold CV, 留出测试。在 CPU 上数分钟完成, 用于
建立可复现的参考指标。生产级模型见 model_e3nn.py (E(3) 等变 GNN)。

用法:
  conda activate fe_dft
  python train.py --dataset dataset/regression_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from featurize import vector_from_structure

REG_TARGETS = ["Ps", "dw_depth", "path_barrier", "gap_polar"]
LOG_TARGETS = {"Ps"}  # 这些目标用 log1p 变换


def load_dataset(csv_path: Path):
    base = csv_path.parent
    rows = list(csv.DictReader(open(csv_path)))
    X, Y, cls, formulas = [], [], [], []
    for r in rows:
        try:
            feat = vector_from_structure(base / r["structure_file"])
        except Exception as exc:
            print(f"  skip {r['formula']}: {exc}")
            continue
        X.append(feat)
        Y.append([float(r[t]) for t in REG_TARGETS])
        cls.append(int(r["is_switchable"]))
        formulas.append(r["formula"])
    return np.array(X), np.array(Y), np.array(cls), formulas


def cv_regression(X, Y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {t: {"r2": [], "mae": []} for t in REG_TARGETS}
    for ti, t in enumerate(REG_TARGETS):
        y = Y[:, ti].copy()
        if t in LOG_TARGETS:
            y = np.log1p(np.clip(y, 0, None))
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            model = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                              learning_rate=0.05, subsample=0.8,
                                              random_state=seed)
            model.fit(sc.transform(X[tr]), y[tr])
            pred = model.predict(sc.transform(X[te]))
            ytrue, ypred = y[te], pred
            if t in LOG_TARGETS:                      # 还原到物理量纲评估
                ytrue, ypred = np.expm1(ytrue), np.expm1(ypred)
            results[t]["r2"].append(r2_score(ytrue, ypred))
            results[t]["mae"].append(mean_absolute_error(ytrue, ypred))
    return results


def cv_classification(X, cls, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accs = [], []
    for tr, te in kf.split(X):
        if len(np.unique(cls[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                         learning_rate=0.05, random_state=seed)
        clf.fit(sc.transform(X[tr]), cls[tr])
        prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(cls[te], prob))
        accs.append(((prob > 0.5).astype(int) == cls[te]).mean())
    return aucs, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset" / "regression_dataset.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    print("Loading + featurizing dataset...")
    X, Y, cls, formulas = load_dataset(args.dataset)
    print(f"  N={len(X)}  feature_dim={X.shape[1]}  switchable={int(cls.sum())}")

    units = {"Ps": "μC/cm²", "dw_depth": "meV/atom", "path_barrier": "meV/atom", "gap_polar": "eV"}
    print("\n5-fold CV regression (baseline GBT):")
    print(f"{'target':<14}{'R²':>14}{'MAE':>18}")
    reg = cv_regression(X, Y)
    summary = {}
    for t in REG_TARGETS:
        r2 = np.mean(reg[t]["r2"]); r2s = np.std(reg[t]["r2"])
        mae = np.mean(reg[t]["mae"]); maes = np.std(reg[t]["mae"])
        print(f"{t:<14}{r2:>8.3f}±{r2s:<4.3f}{mae:>10.3f}±{maes:<5.3f} {units[t]}")
        summary[t] = {"r2": r2, "r2_std": r2s, "mae": mae, "mae_std": maes, "unit": units[t]}

    aucs, accs = cv_classification(X, cls)
    print(f"\nswitchability classifier:  AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
          f"Acc={np.mean(accs):.3f}±{np.std(accs):.3f}")
    summary["is_switchable"] = {"auc": float(np.mean(aucs)), "acc": float(np.mean(accs))}

    json.dump(summary, open(args.out / "baseline_cv_metrics.json", "w"), indent=2)
    print(f"\nmetrics -> {args.out/'baseline_cv_metrics.json'}")


if __name__ == "__main__":
    main()
