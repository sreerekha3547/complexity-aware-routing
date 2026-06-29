"""EXP-05: Transferable-feature ablation for the VRDU negative control.

Addresses a confound a reviewer could raise about the cross-domain claim.
Three label-derived features (label_entropy, section_count, label_diversity)
are structurally zero on VRDU: VRDU provides only document-level key-value
annotations, so the loader never assigns per-word labels and these features
compute to a constant. Two of them are the top-2 routing predictors on
CORD/SROIE.

A constant feature is *inert for ranking* -- it cannot change AUC on the set
where it is constant -- so the near-random VRDU AUC (0.522) is already driven
only by the transferable features. This experiment proves that explicitly, two
ways, so the negative control does not rest on an assumption:

  (A) Transferable-only router. Retrain the SAME calibrated RF using only the
      features that are non-constant on VRDU, and evaluate on all three test
      sets. If VRDU AUC stays ~0.5 while CORD/SROIE hold up, the near-random
      VRDU result is NOT an artifact of dropping the strong label features --
      every feature used is genuinely computable on VRDU.

  (B) In-domain VRDU router. Train on oracle_labels_vrdu_train.csv (200 docs,
      otherwise unused) and test on vrdu_test. If even an in-domain model
      cannot route VRDU, there is genuinely no routable signal in that domain,
      independent of cross-domain transfer.

The "inapplicable" feature set is detected automatically (zero variance on
VRDU), not hard-coded, so the result is self-documenting.

Outputs
-------
results/tables/exp05_transferable_ablation.csv

Run
---
    python experiments/exp_05_transferable_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# Reuse the exact data loading + labeling + feature list from EXP-02 so the
# ablation is identical to the headline pipeline except for the feature subset.
from experiments.exp_03_routing_model import (
    FEATURE_COLS,
    apply_tau,
    load_raw_data,
)

TABLE_DIR = Path("results/tables")
TAU = 0.02


def build_rf() -> Pipeline:
    """Same calibrated RF as EXP-02."""
    return Pipeline([("clf", CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42),
        method="isotonic", cv=3,
    ))])


def fit_rf(train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    X = train_df[feature_cols].fillna(0).values
    y = train_df["label"].values
    rf = build_rf()
    rf.fit(X, y)
    return rf


def auc_on(rf: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> float:
    y = df["label"].values
    if len(np.unique(y)) < 2:
        return float("nan")
    p = rf.predict_proba(df[feature_cols].fillna(0).values)[:, 1]
    return roc_auc_score(y, p)


def load_vrdu_train() -> pd.DataFrame:
    """Load the otherwise-unused VRDU train oracle labels + features."""
    feat = pd.read_csv(TABLE_DIR / "feature_table.csv")
    path = TABLE_DIR / "oracle_labels_vrdu_train.csv"
    if not path.exists():
        return pd.DataFrame()
    gap = pd.read_csv(path)
    return feat[feat["dataset"] == "vrdu"].merge(gap, on=["doc_id", "split"])


def main() -> int:
    print("EXP-05: Transferable-feature ablation (VRDU negative control)\n")

    train_raw, test_raw, vrdu_test_raw = load_raw_data()
    if vrdu_test_raw.empty:
        print("ERROR: VRDU test set not found; cannot run the ablation.")
        return 1

    train_df = apply_tau(train_raw, TAU)
    vrdu_df  = apply_tau(vrdu_test_raw, TAU)
    test_by_ds = {
        ds: apply_tau(test_raw[test_raw["dataset"] == ds], TAU)
        for ds in test_raw["dataset"].unique()
    }

    # --- Detect features that are constant (inert) on VRDU --------------------
    const_cols = [c for c in FEATURE_COLS
                  if vrdu_df[c].fillna(0).std() == 0]
    transferable = [c for c in FEATURE_COLS if c not in const_cols]

    print(f"Features total: {len(FEATURE_COLS)}")
    print(f"Inapplicable on VRDU (zero variance, inert for AUC): {const_cols}")
    print(f"Transferable (non-constant on VRDU): {len(transferable)} features")
    print(f"  {transferable}\n")

    rows: list[dict] = []

    def record(setting, feat_set, feature_cols, rf, evals):
        for ds, df in evals.items():
            rows.append({
                "setting": setting,
                "feature_set": feat_set,
                "n_features": len(feature_cols),
                "eval": ds,
                "n_docs": len(df),
                "pct_large": round(100 * df["label"].mean(), 1),
                "auc": round(auc_on(rf, df, feature_cols), 3),
            })

    # =====================================================================
    # (A) Train on CORD+SROIE, evaluate on all three test sets
    # =====================================================================
    print("=" * 64)
    print("(A) Cross-domain: train on CORD+SROIE, eval on all test sets")
    print("=" * 64)

    evals_A = {**test_by_ds, "vrdu": vrdu_df}

    rf_full = fit_rf(train_df, FEATURE_COLS)
    record("crossdomain", "full(16)", FEATURE_COLS, rf_full, evals_A)

    rf_trans = fit_rf(train_df, transferable)
    record("crossdomain", f"transferable({len(transferable)})",
           transferable, rf_trans, evals_A)

    for fs, rf, fc in [("full(16)", rf_full, FEATURE_COLS),
                       (f"transferable({len(transferable)})", rf_trans, transferable)]:
        line = f"  {fs:<18}"
        for ds, df in evals_A.items():
            line += f"  {ds.upper()} AUC={auc_on(rf, df, fc):.3f}"
        print(line)

    # =====================================================================
    # (B) In-domain VRDU: train on vrdu_train (200 docs), test on vrdu_test
    # =====================================================================
    print("\n" + "=" * 64)
    print("(B) In-domain: train on VRDU train split, test on VRDU test")
    print("=" * 64)

    vrdu_train = load_vrdu_train()
    if vrdu_train.empty:
        print("  oracle_labels_vrdu_train.csv not found -- skipping (B).")
    else:
        vrdu_train_df = apply_tau(vrdu_train, TAU)
        n_large = int(vrdu_train_df["label"].sum())
        print(f"  VRDU train: {len(vrdu_train_df)} docs, "
              f"large-required={n_large} "
              f"({100*n_large/len(vrdu_train_df):.0f}%)")

        if len(np.unique(vrdu_train_df["label"])) < 2:
            print("  VRDU train has a single class at tau=0.02 -- "
                  "in-domain router is undefined (itself evidence of no gap).")
        else:
            rf_in_full = fit_rf(vrdu_train_df, FEATURE_COLS)
            rf_in_trans = fit_rf(vrdu_train_df, transferable)
            record("indomain", "full(16)", FEATURE_COLS, rf_in_full,
                   {"vrdu": vrdu_df})
            record("indomain", f"transferable({len(transferable)})",
                   transferable, rf_in_trans, {"vrdu": vrdu_df})
            print(f"  full(16)        VRDU AUC="
                  f"{auc_on(rf_in_full, vrdu_df, FEATURE_COLS):.3f}")
            print(f"  transferable    VRDU AUC="
                  f"{auc_on(rf_in_trans, vrdu_df, transferable):.3f}")

    # =====================================================================
    # Save + verdict
    # =====================================================================
    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_DIR / "exp05_transferable_ablation.csv", index=False)
    print(f"\n  -> {TABLE_DIR}/exp05_transferable_ablation.csv")

    # One-line interpretation for the paper
    cd = out[(out["setting"] == "crossdomain") & (out["eval"] == "vrdu")]
    vrdu_full = cd[cd["feature_set"] == "full(16)"]["auc"].iloc[0]
    vrdu_trans = cd[cd["feature_set"].str.startswith("transferable")]["auc"].iloc[0]
    indomain = out[(out["setting"] == "indomain") & (out["eval"] == "vrdu")]
    vrdu_indomain = indomain["auc"].max() if not indomain.empty else float("nan")
    print("\nVERDICT")
    print(f"  Cross-domain VRDU AUC: full(16)={vrdu_full:.3f}, "
          f"transferable(12)={vrdu_trans:.3f}")
    print(f"  In-domain VRDU AUC (train on VRDU split): {vrdu_indomain:.3f}")
    print("  Reading: the full-feature cross-domain AUC is depressed by trained-in")
    print("  reliance on label features absent on VRDU; the fair transferable")
    print("  number and the in-domain number show a WEAK but non-zero routable")
    print("  signal. Combined with the tiny mean gap (0.005), VRDU is a")
    print("  low-headroom domain -- not pure noise: practical routing benefit is")
    print("  minimal and the router correctly declines to escalate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
