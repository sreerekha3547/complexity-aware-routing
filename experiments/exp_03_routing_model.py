"""EXP-02: Train and evaluate the routing classifier.

Merges the feature table (EXP-01) with oracle tier labels (EXP-03) for CORD
and SROIE, trains a routing model, and produces the core paper result: a
Pareto frontier showing quality vs cost at different routing thresholds.

Multi-tau sensitivity: loops over TAU_VALUES = [0.01, 0.02, 0.05], re-labeling
oracle tier decisions and retraining the classifier each time. All downstream
computation is local and free (no API calls).

Changes vs initial draft
------------------------
1. complexity_score dropped from features -- collinear with its own components
   (inv_chars_per_word corr=0.95, section_count=0.94).
2. Two required baselines added: random routing and single-feature (ocr_conf).
3. Calibrated Random Forest used for Pareto -- RF AUC > LR; isotonic
   calibration gives well-calibrated scores for the threshold sweep.
4. Success threshold: 2 absolute F1 points below always-large (pre-registered).
5. Multi-tau loop: TAU_VALUES = [0.01, 0.02, 0.05] with summary table.

Outputs
-------
results/tables/routing_predictions.csv        -- per-doc predictions (tau=0.02)
results/tables/tau_sensitivity.csv            -- summary table across tau values
results/figures/exp03_pareto_{dataset}.png           -- main result (tau=0.02)
results/figures/exp03_pareto_{dataset}_multi_tau.png -- sensitivity figure

Run
---
    python experiments/exp_03_routing_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TABLE_DIR = Path("results/tables")
FIG_DIR   = Path("results/figures")

# 13 genuinely pre-inference features. section_count / label_diversity /
# label_entropy are EXCLUDED: they are derived from ground-truth field
# annotations (doc.labels) and are not available before extraction. They remain
# in feature_table.csv only for the leakage-robustness ablation (EXP-09).
# crowded_line_frac / line_density / item_density now use geometric OCR-box
# line-clustering for every dataset (see src/features/complexity_score.py).
FEATURE_COLS = [
    "ocr_conf", "ocr_std", "ocr_stage",
    "image_contrast", "blur_score",
    "short_token_ratio", "inv_chars_per_word", "word_height_cv",
    "crowded_line_frac", "line_density", "item_density",
    "aspect_ratio", "tokens",
]

# Annotation-derived features, kept for the leakage ablation only (never routed on).
LEAKY_FEATURE_COLS = ["section_count", "label_diversity", "label_entropy"]

# Pre-registered success criterion: routing quality within 2 absolute F1
# points of always-large.
F1_GAP_TOLERANCE = 0.02

# Tau values to sweep. Results are aggregated into tau_sensitivity.csv.
TAU_VALUES = [0.01, 0.02, 0.05]
TAU_COLORS = {0.01: "steelblue", 0.02: "darkorange", 0.05: "seagreen"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features + raw gap values.

    Returns (train_raw, test_raw, vrdu_test_raw).

    CORD and SROIE are used for both training and evaluation -- they have
    enough routing signal to train a meaningful classifier.

    VRDU is evaluated separately (held-out generalization test) using the
    model trained on CORD+SROIE. Its near-zero mean gap (0.017) and different
    feature distribution would dilute the joint classifier if included in
    training.
    """
    feat = pd.read_csv(TABLE_DIR / "feature_table.csv")
    train_frames, test_frames = [], []

    # --- CORD + SROIE: training + evaluation datasets -------------------------
    train_splits_map = {"cord": ["train", "dev"], "sroie": ["train"]}
    for dataset in ["cord", "sroie"]:
        test_path = TABLE_DIR / f"oracle_labels_{dataset}_test.csv"
        if test_path.exists():
            gap = pd.read_csv(test_path)
            merged = feat[feat["dataset"] == dataset].merge(
                gap, on=["doc_id", "split"])
            test_frames.append(merged)
            print(f"  Test  {dataset}: {len(merged)} docs")
        else:
            print(f"  INFO: {test_path.name} not found -- skipping {dataset} test")

        for split in train_splits_map[dataset]:
            train_path = TABLE_DIR / f"oracle_labels_{dataset}_{split}.csv"
            if train_path.exists():
                gap = pd.read_csv(train_path)
                f = feat[feat["dataset"] == dataset].merge(
                    gap, on=["doc_id", "split"])
                train_frames.append(f)
                print(f"  Train {dataset}/{split}: {len(f)} docs")

    # --- VRDU: held-out generalization test only ------------------------------
    vrdu_test_raw = pd.DataFrame()
    vrdu_path = TABLE_DIR / "oracle_labels_vrdu_test.csv"
    if vrdu_path.exists():
        gap = pd.read_csv(vrdu_path)
        vrdu_test_raw = feat[feat["dataset"] == "vrdu"].merge(
            gap, on=["doc_id", "split"])
        print(f"  Test  vrdu (held-out): {len(vrdu_test_raw)} docs")
    else:
        print("  INFO: oracle_labels_vrdu_test.csv not found -- skipping VRDU")

    test_raw  = pd.concat(test_frames,  ignore_index=True) if test_frames  else pd.DataFrame()
    train_raw = pd.concat(train_frames, ignore_index=True) if train_frames else test_raw.copy()
    return train_raw, test_raw, vrdu_test_raw


def apply_tau(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Add 'label' column: 1=large-required (gap > tau), 0=small-sufficient."""
    df = df.copy()
    df["label"] = (df["gap"] > tau).astype(int)
    return df


# ---------------------------------------------------------------------------
# Pareto computation
# ---------------------------------------------------------------------------

def compute_pareto(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Quality vs cost across 101 routing thresholds.

    Docs with score >= threshold are routed to large; the rest to small.
    """
    rows = []
    for t in np.linspace(0, 1, 101):
        route_large = df[score_col] >= t
        quality = np.where(route_large, df["f1_large"], df["f1_small"]).mean()
        cost    = np.where(route_large, df["cost_large"], df["cost_small"]).mean()
        rows.append({"threshold": round(t, 2), "quality": quality,
                     "cost": cost, "frac_large": route_large.mean()})
    return pd.DataFrame(rows)


def compute_random_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Expected quality/cost when randomly routing fraction r to large.

    A straight line (convex hull of always-small and always-large). Any
    routing model with real signal must lie above this line.
    """
    rows = []
    for r in np.linspace(0, 1, 101):
        quality = r * df["f1_large"].mean() + (1 - r) * df["f1_small"].mean()
        cost    = r * df["cost_large"].mean() + (1 - r) * df["cost_small"].mean()
        rows.append({"quality": quality, "cost": cost, "frac_large": r})
    return pd.DataFrame(rows)


def _pareto_best_saving(pareto: pd.DataFrame, always_large_qual: float,
                        always_large_cost: float) -> tuple[float | None, float | None]:
    """Return (cost_saving_pct, frac_large) at target quality, or (None, None)."""
    target = always_large_qual - F1_GAP_TOLERANCE
    matching = pareto[pareto["quality"] >= target]
    if matching.empty:
        return None, None
    best = matching.loc[matching["cost"].idxmin()]
    saving = (1 - best["cost"] / always_large_cost) * 100
    return round(saving, 1), round(best["frac_large"] * 100, 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto(pareto_rf: pd.DataFrame,
                pareto_single: pd.DataFrame,
                random_pareto: pd.DataFrame,
                baselines: dict,
                dataset_label: str,
                tau: float,
                out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(pareto_rf["cost"],     pareto_rf["quality"],
            "b-",  lw=2, label="Routing model (calibrated RF)")
    ax.plot(pareto_single["cost"], pareto_single["quality"],
            "m--", lw=1.5, label="Single-feature (ocr_conf only)")
    ax.plot(random_pareto["cost"], random_pareto["quality"],
            "gray", lw=1, linestyle=":", label="Random routing")

    style = {
        "always-small": ("green", "s", 100),
        "always-large": ("red",   "^", 100),
        "oracle":       ("gold",  "*", 200),
    }
    for name, (cost, qual) in baselines.items():
        c, m, s = style[name]
        ax.scatter([cost], [qual], c=c, s=s, marker=m, zorder=5, label=name)

    ax.set_xlabel("Mean cost per doc (USD)")
    ax.set_ylabel("Mean extraction F1")
    ax.set_title(f"Quality vs Cost Pareto -- {dataset_label}  (tau={tau})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_multi_tau_pareto(tau_paretos: dict[float, pd.DataFrame],
                          baselines: dict,
                          dataset_label: str,
                          out_path: Path) -> None:
    """Overlay Pareto curves for all tau values on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for tau, pareto in tau_paretos.items():
        color = TAU_COLORS[tau]
        ax.plot(pareto["cost"], pareto["quality"],
                color=color, lw=2, label=f"tau={tau}")

    style = {
        "always-small": ("green", "s", 100),
        "always-large": ("red",   "^", 100),
    }
    for name, (cost, qual) in baselines.items():
        if name in style:
            c, m, s = style[name]
            ax.scatter([cost], [qual], c=c, s=s, marker=m, zorder=5, label=name)

    ax.set_xlabel("Mean cost per doc (USD)")
    ax.set_ylabel("Mean extraction F1")
    ax.set_title(f"Quality vs Cost Pareto -- {dataset_label}  (tau sensitivity)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _fit_classifiers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    using_cv: bool,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Fit LR, calibrated RF, and single-feature LR. Return (lr_probs, rf_probs,
    single_probs, lr_auc, rf_auc, single_auc) evaluated on test_df."""
    X_train  = train_df[FEATURE_COLS].fillna(0).values
    y_train  = train_df["label"].values
    X_test   = test_df[FEATURE_COLS].fillna(0).values
    X_s_tr   = train_df[["ocr_conf"]].fillna(0).values
    X_s_te   = test_df[["ocr_conf"]].fillna(0).values
    y_test   = test_df["label"].values
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    lr = Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000,
                                              class_weight="balanced"))])
    if using_cv:
        lr_probs = cross_val_predict(lr, X_train, y_train, cv=cv,
                                     method="predict_proba")[:, 1]
    else:
        lr.fit(X_train, y_train)
        lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    # Calibrated Random Forest
    rf_cal = Pipeline([("clf", CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42),
        method="isotonic", cv=3,
    ))])
    if using_cv:
        rf_probs = cross_val_predict(rf_cal, X_train, y_train, cv=cv,
                                     method="predict_proba")[:, 1]
    else:
        rf_cal.fit(X_train, y_train)
        rf_probs = rf_cal.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    # Single-feature baseline
    lr_single = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=1000,
                                                       class_weight="balanced"))])
    if using_cv:
        single_probs = cross_val_predict(lr_single, X_s_tr, y_train,
                                         cv=cv, method="predict_proba")[:, 1]
    else:
        lr_single.fit(X_s_tr, y_train)
        single_probs = lr_single.predict_proba(X_s_te)[:, 1]
    single_auc = roc_auc_score(y_test, single_probs)

    if verbose:
        print(f"  LR AUC={lr_auc:.3f}  RF AUC={rf_auc:.3f}  "
              f"single-feat AUC={single_auc:.3f}  "
              f"gain vs single: +{rf_auc - single_auc:.3f}")
        rf_preds = (rf_probs >= 0.5).astype(int)
        print(classification_report(y_test, rf_preds,
                                     target_names=["small-sufficient",
                                                   "large-required"]))

    return lr_probs, rf_probs, single_probs, lr_auc, rf_auc, single_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("EXP-02: Routing model (multi-tau sensitivity)\n")
    print("Loading data...")
    train_raw, test_raw, vrdu_test_raw = load_raw_data()

    using_cv = len(train_raw) == len(test_raw)
    if using_cv:
        print(f"\n  MODE: cross-validation on test split ({len(test_raw)} docs)")
    else:
        print(f"\n  MODE: train ({len(train_raw)} docs) -> test ({len(test_raw)} docs)")

    # Per-dataset static baselines (independent of tau)
    static_baselines: dict[str, dict] = {}
    for dataset in test_raw["dataset"].unique():
        sub = test_raw[test_raw["dataset"] == dataset]
        static_baselines[dataset] = {
            "always_small_cost": sub["cost_small"].mean(),
            "always_small_qual": sub["f1_small"].mean(),
            "always_large_cost": sub["cost_large"].mean(),
            "always_large_qual": sub["f1_large"].mean(),
        }

    # Accumulate per-tau Pareto curves for the multi-tau figure
    # {dataset: {tau: pareto_df}}
    multi_tau_paretos: dict[str, dict[float, pd.DataFrame]] = {
        d: {} for d in test_raw["dataset"].unique()
    }

    summary_rows: list[dict] = []

    # -----------------------------------------------------------------------
    # Main tau loop
    # -----------------------------------------------------------------------
    for tau in TAU_VALUES:
        print(f"\n{'='*60}")
        print(f"  tau = {tau}")
        print(f"{'='*60}")

        train_df = apply_tau(train_raw, tau)
        test_df  = apply_tau(test_raw,  tau)

        n_large_train = train_df["label"].sum()
        n_large_test  = test_df["label"].sum()
        print(f"  Train: {len(train_df)} docs  "
              f"large-required={n_large_train} ({100*n_large_train/len(train_df):.0f}%)")
        print(f"  Test:  {len(test_df)} docs  "
              f"large-required={n_large_test} ({100*n_large_test/len(test_df):.0f}%)\n")

        is_default_tau = (tau == 0.02)
        verbose = is_default_tau

        if verbose:
            # Full classification report only for the main tau value
            print("Logistic Regression (detailed)...")
        lr_probs, rf_probs, single_probs, lr_auc, rf_auc, single_auc = \
            _fit_classifiers(train_df, test_df, using_cv, verbose=verbose)

        if not verbose:
            print(f"  LR AUC={lr_auc:.3f}  RF AUC={rf_auc:.3f}  "
                  f"single-feat AUC={single_auc:.3f}")

        test_df = test_df.copy()
        test_df["p_large_rf_cal"] = rf_probs
        test_df["p_large_single"] = single_probs
        test_df["p_large_lr"]     = lr_probs

        row: dict = {"tau": tau, "auc_rf": round(rf_auc, 3),
                     "auc_lr": round(lr_auc, 3),
                     "auc_single": round(single_auc, 3)}

        print()
        for dataset in test_df["dataset"].unique():
            sub = test_df[test_df["dataset"] == dataset].copy()
            bs  = static_baselines[dataset]

            pct_large = 100 * sub["label"].mean()
            row[f"pct_large_{dataset}"] = round(pct_large, 1)

            pareto_rf     = compute_pareto(sub, "p_large_rf_cal")
            pareto_single = compute_pareto(sub, "p_large_single")
            rand_pareto   = compute_random_pareto(sub)

            multi_tau_paretos[dataset][tau] = pareto_rf

            oracle_qual = np.where(sub["label"] == 1,
                                   sub["f1_large"], sub["f1_small"]).mean()
            oracle_cost = np.where(sub["label"] == 1,
                                   sub["cost_large"], sub["cost_small"]).mean()

            saving_rf, frac_rf = _pareto_best_saving(
                pareto_rf, bs["always_large_qual"], bs["always_large_cost"])
            saving_single, _ = _pareto_best_saving(
                pareto_single, bs["always_large_qual"], bs["always_large_cost"])

            row[f"saving_rf_{dataset}"]     = saving_rf
            row[f"saving_single_{dataset}"] = saving_single

            print(f"  [{dataset.upper()}]  pct-large={pct_large:.0f}%  "
                  f"oracle-saving={100*(1-oracle_cost/bs['always_large_cost']):.0f}%")
            if saving_rf is not None:
                print(f"    Full model  : {saving_rf:.0f}% cost saving  "
                      f"(routes {frac_rf:.0f}% to large)")
            else:
                print(f"    Full model  : cannot reach target quality")
            if saving_single is not None:
                print(f"    Single-feat : {saving_single:.0f}% cost saving")
            else:
                print(f"    Single-feat : cannot reach target quality")

            # Per-tau single Pareto figure
            baselines_plot = {
                "always-small": (bs["always_small_cost"], bs["always_small_qual"]),
                "always-large": (bs["always_large_cost"], bs["always_large_qual"]),
                "oracle":       (oracle_cost, oracle_qual),
            }
            plot_pareto(pareto_rf, pareto_single, rand_pareto, baselines_plot,
                        dataset.upper(), tau,
                        FIG_DIR / f"exp03_pareto_{dataset}_tau{str(tau).replace('.','')}.png")

        summary_rows.append(row)

        # Save predictions for default tau
        if is_default_tau:
            out_cols = (["doc_id", "dataset", "label", "tier_label",
                         "f1_small", "f1_large", "gap",
                         "p_large_lr", "p_large_rf_cal", "p_large_single",
                         "cost_small", "cost_large"]
                        + FEATURE_COLS)
            test_df[out_cols].to_csv(TABLE_DIR / "routing_predictions.csv",
                                     index=False)
            print(f"\n  Predictions (tau=0.02) -> "
                  f"{TABLE_DIR}/routing_predictions.csv")

            # Feature importance for default tau
            X_tr = train_df[FEATURE_COLS].fillna(0).values
            y_tr = train_df["label"].values
            lr_full = Pipeline([("scaler", StandardScaler()),
                                 ("clf", LogisticRegression(max_iter=1000,
                                                             class_weight="balanced"))])
            lr_full.fit(X_tr, y_tr)
            coefs = lr_full.named_steps["clf"].coef_[0]
            feat_imp = sorted(zip(FEATURE_COLS, coefs),
                              key=lambda x: abs(x[1]), reverse=True)
            print("\nTop routing features (LR coeff, tau=0.02):")
            for feat, coef in feat_imp[:8]:
                direction = "-> large" if coef > 0 else "-> small"
                print(f"  {feat:<30} {coef:+.3f}  {direction}")

    # -----------------------------------------------------------------------
    # VRDU held-out evaluation (model trained on CORD+SROIE only)
    # -----------------------------------------------------------------------
    if not vrdu_test_raw.empty:
        print(f"\n{'='*60}")
        print("  VRDU HELD-OUT EVALUATION (CORD+SROIE model, tau=0.02)")
        print(f"{'='*60}")

        # Re-train classifier at tau=0.02 on full CORD+SROIE train set
        train_02  = apply_tau(train_raw, 0.02)
        vrdu_test = apply_tau(vrdu_test_raw, 0.02)

        X_tr = train_02[FEATURE_COLS].fillna(0).values
        y_tr = train_02["label"].values
        X_vr = vrdu_test[FEATURE_COLS].fillna(0).values
        y_vr = vrdu_test["label"].values

        rf_vrdu = Pipeline([("clf", CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=42),
            method="isotonic", cv=3,
        ))])
        rf_vrdu.fit(X_tr, y_tr)
        vrdu_probs = rf_vrdu.predict_proba(X_vr)[:, 1]

        from sklearn.metrics import roc_auc_score as _auc
        vrdu_auc = _auc(y_vr, vrdu_probs) if len(np.unique(y_vr)) > 1 else float("nan")

        vrdu_test = vrdu_test.copy()
        vrdu_test["p_large_rf_cal"] = vrdu_probs

        bs_vr = {
            "always_small_cost": vrdu_test["cost_small"].mean(),
            "always_small_qual": vrdu_test["f1_small"].mean(),
            "always_large_cost": vrdu_test["cost_large"].mean(),
            "always_large_qual": vrdu_test["f1_large"].mean(),
        }
        oracle_qual_vr = np.where(vrdu_test["label"] == 1,
                                  vrdu_test["f1_large"],
                                  vrdu_test["f1_small"]).mean()
        oracle_cost_vr = np.where(vrdu_test["label"] == 1,
                                  vrdu_test["cost_large"],
                                  vrdu_test["cost_small"]).mean()

        pareto_vr   = compute_pareto(vrdu_test, "p_large_rf_cal")
        rand_vr     = compute_random_pareto(vrdu_test)
        saving_vr, frac_vr = _pareto_best_saving(
            pareto_vr, bs_vr["always_large_qual"], bs_vr["always_large_cost"])

        print(f"  docs          : {len(vrdu_test)}")
        print(f"  pct-large     : {100*vrdu_test['label'].mean():.0f}%  (tau=0.02)")
        print(f"  AUC-RF        : {vrdu_auc:.3f}  (cross-domain transfer)")
        print(f"  oracle saving : {100*(1-oracle_cost_vr/bs_vr['always_large_cost']):.0f}%")
        if saving_vr is not None:
            print(f"  model saving  : {saving_vr:.0f}%  "
                  f"(routes {frac_vr:.0f}% to large)")
        else:
            print("  model saving  : cannot reach target quality")

        baselines_vr = {
            "always-small": (bs_vr["always_small_cost"], bs_vr["always_small_qual"]),
            "always-large": (bs_vr["always_large_cost"], bs_vr["always_large_qual"]),
            "oracle":       (oracle_cost_vr, oracle_qual_vr),
        }
        # Reuse single-feature scores from the last tau loop iteration for VRDU
        lr_single_vr = Pipeline([("scaler", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=1000,
                                                              class_weight="balanced"))])
        lr_single_vr.fit(train_02[["ocr_conf"]].fillna(0).values, y_tr)
        single_probs_vr = lr_single_vr.predict_proba(
            vrdu_test[["ocr_conf"]].fillna(0).values)[:, 1]
        pareto_single_vr = compute_pareto(vrdu_test.assign(
            p_single=single_probs_vr), "p_single")

        plot_pareto(pareto_vr, pareto_single_vr, rand_vr, baselines_vr,
                    "VRDU (cross-domain)", 0.02,
                    FIG_DIR / "exp03_pareto_vrdu.png")

    # -----------------------------------------------------------------------
    # Multi-tau summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TAU SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary_rows)

    # Print formatted table
    datasets = list(test_raw["dataset"].unique())
    header = (f"{'tau':>5}  {'AUC-RF':>7}  "
              + "  ".join(f"{'pct-large-'+d.upper():>15}  {'save-'+d.upper():>10}"
                          for d in datasets))
    print(header)
    print("-" * len(header))
    for _, r in summary_df.iterrows():
        line = f"  {r['tau']:.2f}  {r['auc_rf']:>7.3f}"
        for d in datasets:
            pct  = r.get(f"pct_large_{d}", "N/A")
            save = r.get(f"saving_rf_{d}", "N/A")
            pct_s  = f"{pct:.0f}%" if pct  is not None else "N/A"
            save_s = f"{save:.0f}%" if save is not None else "fail"
            line += f"  {pct_s:>15}  {save_s:>10}"
        print(line)

    summary_df.to_csv(TABLE_DIR / "tau_sensitivity.csv", index=False)
    print(f"\n  -> {TABLE_DIR}/tau_sensitivity.csv")

    # -----------------------------------------------------------------------
    # Multi-tau combined Pareto figures (one per dataset)
    # -----------------------------------------------------------------------
    for dataset in test_raw["dataset"].unique():
        bs = static_baselines[dataset]
        static_pts = {
            "always-small": (bs["always_small_cost"], bs["always_small_qual"]),
            "always-large": (bs["always_large_cost"], bs["always_large_qual"]),
        }
        plot_multi_tau_pareto(
            multi_tau_paretos[dataset], static_pts,
            dataset.upper(),
            FIG_DIR / f"exp03_pareto_{dataset}_multi_tau.png",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
