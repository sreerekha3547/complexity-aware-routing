"""EXP-08: Significance, confidence intervals, and calibration (reviewers R4, R9).

Bootstraps 95% confidence intervals for (i) routing AUC and (ii) cost saving at
the operating point, per dataset and pooled, and reports the router's expected
calibration error (ECE). All from the stored test predictions -- no model
retraining or API calls.

Outputs
-------
results/tables/exp08_significance.csv

Run
---
    python experiments/exp_08_significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.exp_03_routing_model import F1_GAP_TOLERANCE

TABLE_DIR = Path("results/tables")
PRED = TABLE_DIR / "routing_predictions.csv"
SCORE = "p_large_rf_cal"
N_BOOT = 2000
RNG = np.random.default_rng(42)


def operating_threshold(df: pd.DataFrame, al_cost: float, al_qual: float) -> float:
    """Min-cost threshold meeting the quality tolerance on the full sample."""
    best_t, best_cost = 1.0, np.inf
    for t in np.linspace(0, 1, 101):
        route = df[SCORE] >= t
        qual = np.where(route, df["f1_large"], df["f1_small"]).mean()
        if qual < al_qual - F1_GAP_TOLERANCE:
            continue
        cost = np.where(route, df["cost_large"], df["cost_small"]).mean()
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t


def saving_at(df: pd.DataFrame, t: float, al_cost: float) -> float:
    route = df[SCORE] >= t
    cost = np.where(route, df["cost_large"], df["cost_small"]).mean()
    return (1 - cost / al_cost) * 100


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return e


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return round(lo, 3), round(hi, 3)


def main() -> int:
    print("EXP-08: Bootstrap CIs + calibration\n")
    if not PRED.exists():
        print(f"ERROR: {PRED} not found -- run exp_03 first.")
        return 1
    df = pd.read_csv(PRED)

    groups = {ds: df[df["dataset"] == ds] for ds in df["dataset"].unique()}
    groups["pooled"] = df

    rows = []
    print(f"{'group':8} {'AUC':>6} {'AUC 95% CI':>16} "
          f"{'save%':>6} {'save 95% CI':>16} {'ECE':>6}")
    print("-" * 64)
    for name, g in groups.items():
        y = g["label"].to_numpy()
        p = g[SCORE].to_numpy()
        idx = np.arange(len(g))

        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")

        # operating point + saving (skip pooled saving: per-dataset costs differ)
        do_saving = name != "pooled"
        if do_saving:
            al_cost = g["cost_large"].mean()
            al_qual = g["f1_large"].mean()
            t_op = operating_threshold(g, al_cost, al_qual)
            saving = saving_at(g, t_op, al_cost)
        else:
            saving = float("nan")

        aucs, saves = [], []
        for _ in range(N_BOOT):
            b = RNG.choice(idx, size=len(idx), replace=True)
            gb = g.iloc[b]
            yb = gb["label"].to_numpy()
            if len(np.unique(yb)) > 1:
                aucs.append(roc_auc_score(yb, gb[SCORE].to_numpy()))
            if do_saving:
                saves.append(saving_at(gb, t_op, gb["cost_large"].mean()))

        auc_lo, auc_hi = boot_ci(np.array(aucs)) if aucs else (float("nan"),) * 2
        if do_saving and saves:
            s_lo, s_hi = boot_ci(np.array(saves))
        else:
            s_lo = s_hi = float("nan")
        e = ece(p, y)

        rows.append({"group": name, "auc": round(auc, 3),
                     "auc_lo": auc_lo, "auc_hi": auc_hi,
                     "saving": round(saving, 1) if do_saving else None,
                     "saving_lo": round(s_lo, 1) if do_saving else None,
                     "saving_hi": round(s_hi, 1) if do_saving else None,
                     "ece": round(e, 3)})
        sv = f"{saving:>5.0f}" if do_saving else "  ---"
        svci = f"[{s_lo:.0f}, {s_hi:.0f}]" if do_saving else "---"
        print(f"{name:8} {auc:>6.3f} [{auc_lo:.3f}, {auc_hi:.3f}] "
              f"{sv} {svci:>16} {e:>6.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "exp08_significance.csv", index=False)
    print(f"\n  -> {TABLE_DIR}/exp08_significance.csv")
    print(f"\n  ({N_BOOT} bootstrap resamples; ECE over 10 bins.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
