"""EXP-06: No-peek threshold transfer.

The headline savings in EXP-02 read the operating point t* off the *test* Pareto
(the cost-minimal threshold that holds test quality within delta of always-large).
That characterises the achievable frontier but selects the threshold on the
evaluation set. This experiment answers the reviewer question "how is the
threshold chosen without peeking at test?":

  1. Get OUT-OF-FOLD probabilities on the pooled CORD+SROIE training set
     (cross_val_predict), so the threshold is not chosen on data the model fit.
  2. Per dataset, select t* = cost-minimal threshold whose TRAIN quality stays
     within delta of train always-large.
  3. Fit the router on the full training set, predict the held-out test set, and
     apply the FIXED, train-selected t* unchanged to test.
  4. Report test saving + test quality at that threshold, and compare to the
     test-selected frontier-best saving.

If the train-selected threshold transfers (saving close to frontier-best, test
quality still within delta), the cost saving is not an artifact of test peeking
and the paper can state the stronger no-peek protocol.

Outputs
-------
results/tables/exp06_threshold_transfer.csv

Run
---
    python experiments/exp_06_threshold_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from experiments.exp_03_routing_model import (
    FEATURE_COLS,
    F1_GAP_TOLERANCE,
    apply_tau,
    compute_pareto,
    load_raw_data,
)

TABLE_DIR = Path("results/tables")
TAU = 0.02


def build_rf() -> Pipeline:
    return Pipeline([("clf", CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42),
        method="isotonic", cv=3,
    ))])


def select_threshold(train_sub: pd.DataFrame, score_col: str,
                     always_large_qual: float) -> float | None:
    """Cost-minimal threshold whose TRAIN quality stays within delta."""
    pareto = compute_pareto(train_sub, score_col)
    target = always_large_qual - F1_GAP_TOLERANCE
    ok = pareto[pareto["quality"] >= target]
    if ok.empty:
        return None
    return float(ok.loc[ok["cost"].idxmin(), "threshold"])


def eval_at_threshold(test_sub: pd.DataFrame, score_col: str, t: float,
                      al_cost: float, al_qual: float):
    route = test_sub[score_col] >= t
    q = float(np.where(route, test_sub["f1_large"], test_sub["f1_small"]).mean())
    c = float(np.where(route, test_sub["cost_large"], test_sub["cost_small"]).mean())
    saving = (1 - c / al_cost) * 100
    within = q >= al_qual - F1_GAP_TOLERANCE
    return round(saving, 1), round(q, 4), round(100 * route.mean(), 1), bool(within)


def frontier_best(test_sub: pd.DataFrame, score_col: str,
                  al_cost: float, al_qual: float):
    """Test-selected (peeking) saving -- the EXP-02 headline number."""
    pareto = compute_pareto(test_sub, score_col)
    target = al_qual - F1_GAP_TOLERANCE
    ok = pareto[pareto["quality"] >= target]
    if ok.empty:
        return None
    best = ok.loc[ok["cost"].idxmin()]
    return round((1 - best["cost"] / al_cost) * 100, 1)


def main() -> int:
    print("EXP-06: No-peek threshold transfer\n")
    train_raw, test_raw, _ = load_raw_data()

    train_df = apply_tau(train_raw, TAU)
    test_df  = apply_tau(test_raw, TAU)

    X_tr = train_df[FEATURE_COLS].fillna(0).values
    y_tr = train_df["label"].values
    X_te = test_df[FEATURE_COLS].fillna(0).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Out-of-fold train probabilities -> threshold selection never sees the
    # data the fold's model was fit on.
    print("Computing out-of-fold train probabilities (nested CV)...")
    train_df = train_df.copy()
    train_df["p_oof"] = cross_val_predict(build_rf(), X_tr, y_tr, cv=cv,
                                          method="predict_proba")[:, 1]

    # Fit on full train, predict test.
    rf = build_rf().fit(X_tr, y_tr)
    test_df = test_df.copy()
    test_df["p"] = rf.predict_proba(X_te)[:, 1]

    rows = []
    print(f"\n{'dataset':8} {'t*':>5} {'no-peek save':>13} {'frontier save':>14} "
          f"{'test F1':>8} {'within delta':>13}")
    print("-" * 70)
    for ds in test_df["dataset"].unique():
        tr = train_df[train_df["dataset"] == ds]
        te = test_df[test_df["dataset"] == ds]

        al_qual_tr = tr["f1_large"].mean()
        al_cost_te = te["cost_large"].mean()
        al_qual_te = te["f1_large"].mean()

        t_star = select_threshold(tr, "p_oof", al_qual_tr)
        if t_star is None:
            print(f"{ds:8}  train cannot reach target quality -- skipping")
            continue

        save_np, q_te, frac, within = eval_at_threshold(
            te, "p", t_star, al_cost_te, al_qual_te)
        save_fr = frontier_best(te, "p", al_cost_te, al_qual_te)

        rows.append({
            "dataset": ds, "t_star_train": round(t_star, 2),
            "saving_nopeek": save_np, "saving_frontier": save_fr,
            "test_f1": q_te, "frac_large": frac,
            "within_delta": within,
        })
        print(f"{ds:8} {t_star:>5.2f} {save_np:>12.1f}% {str(save_fr)+'%':>14} "
              f"{q_te:>8.3f} {str(within):>13}")

    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_DIR / "exp06_threshold_transfer.csv", index=False)
    print(f"\n  -> {TABLE_DIR}/exp06_threshold_transfer.csv")

    print("\nVERDICT")
    print("  no-peek saving = threshold selected on TRAIN, applied unchanged to TEST.")
    print("  frontier saving = threshold selected on TEST (EXP-02 headline).")
    print("  If the two are close and 'within delta' is True, the saving is NOT")
    print("  an artifact of test-set threshold selection -- the no-peek protocol")
    print("  can be stated in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
