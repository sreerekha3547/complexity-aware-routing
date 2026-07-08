"""EXP-13: Baselines and robustness (reviewers R5-R8), all on the CLEAN roster.

One reproducible place for the comparisons that decide the paper's honesty:

  (1) Text baseline vs. our features, MATCHED classifier (plain RF for both, so
      it is a fair fight -- not our calibrated model vs. a plain text model).
      Text = TF-IDF+LR and TF-IDF->SVD(100) LSA+RF over the OCR text.
  (2) Leave-one-dataset-out: train CORD -> test SROIE and vice versa. Tests
      whether the signal is domain-general difficulty or dataset-specific.
  (3) Dataset-identity baseline: how much pooled AUC is just base-rate
      separation between datasets.
  (4) Leakage ablation: full-16 (with the annotation-derived features) vs the
      clean-13 roster -- confirms the leakage was not load-bearing.

Local, no API. Uses the 13 clean pre-inference features (exp_03.FEATURE_COLS)
and the annotation-derived LEAKY_FEATURE_COLS only for ablation (4).

    python experiments/exp_13_text_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from src.data import get_loader
from src.extraction.base_extractor import reconstruct_text
from experiments.exp_03_routing_model import FEATURE_COLS, LEAKY_FEATURE_COLS

TABLE_DIR = Path("results/tables")
TAU = 0.02
RNG = np.random.default_rng(42)
feat = pd.read_csv(TABLE_DIR / "feature_table.csv")


def load(specs):
    rows = []
    for ds, split in specs:
        lab = pd.read_csv(TABLE_DIR / f"oracle_labels_{ds}_{split}.csv")[["doc_id", "split", "gap"]]
        m = feat[feat.dataset == ds].merge(lab, on=["doc_id", "split"])
        txt = {d.doc_id: (reconstruct_text(d) or " ") for d in get_loader(ds).load_split(split)}
        m["text"] = m.doc_id.map(txt)
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    d["label"] = (d.gap > TAU).astype(int)
    return d


def rf():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)


def paired(y, a, b, B=5000):
    n = len(y); ds = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        ds.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
    return ds.mean(), lo, hi


def main() -> int:
    tr = load([("cord", "train"), ("sroie", "train")])
    te = load([("cord", "test"), ("sroie", "test")])
    y_tr, y_te = tr.label.to_numpy(), te.label.to_numpy()
    masks = {"pooled": np.ones(len(te), bool),
             "CORD": (te.dataset == "cord").values, "SROIE": (te.dataset == "sroie").values}

    def aucs(p):
        return {k: (roc_auc_score(y_te[m], p[m]) if len(np.unique(y_te[m])) > 1 else float("nan"))
                for k, m in masks.items()}

    # ---- text representations ----
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2).fit(tr.text)
    svd = TruncatedSVD(100, random_state=42).fit(vec.transform(tr.text))
    Xtr_txt, Xte_txt = svd.transform(vec.transform(tr.text)), svd.transform(vec.transform(te.text))
    Xtr_f, Xte_f = tr[FEATURE_COLS].fillna(0).values, te[FEATURE_COLS].fillna(0).values

    p_feat = rf().fit(Xtr_f, y_tr).predict_proba(Xte_f)[:, 1]
    p_text = rf().fit(Xtr_txt, y_tr).predict_proba(Xte_txt)[:, 1]
    p_both = rf().fit(np.hstack([Xtr_f, Xtr_txt]), y_tr).predict_proba(np.hstack([Xte_f, Xte_txt]))[:, 1]

    print("(1) MATCHED text vs. clean features (plain RF)   pooled / CORD / SROIE")
    for name, p in [("clean features (13)", p_feat), ("text LSA(100)", p_text), ("features + text", p_both)]:
        a = aucs(p); print(f"    {name:20} {a['pooled']:.3f} {a['CORD']:.3f} {a['SROIE']:.3f}")
    m, lo, hi = paired(y_te, p_text, p_feat)
    print(f"    text - features: {m:+.3f} CI [{lo:+.3f},{hi:+.3f}]  "
          f"{'text sig. better' if lo > 0 else 'not significant'}")

    # ---- (2) leave-one-dataset-out ----
    print("\n(2) Leave-one-dataset-out (clean features)")
    full = pd.concat([tr, te], ignore_index=True)
    for a, b in [("cord", "sroie"), ("sroie", "cord")]:
        A, B = full[full.dataset == a], full[full.dataset == b]
        p = rf().fit(A[FEATURE_COLS].fillna(0).values, A.label.to_numpy()).predict_proba(
            B[FEATURE_COLS].fillna(0).values)[:, 1]
        print(f"    train {a.upper()} -> test {b.upper()}: AUC {roc_auc_score(B.label, p):.3f}")

    # ---- (3) dataset-identity baseline ----
    rate = {ds: tr[tr.dataset == ds].label.mean() for ds in ["cord", "sroie"]}
    p_id = te.dataset.map(rate).to_numpy()
    print(f"\n(3) Dataset-identity baseline: pooled AUC {roc_auc_score(y_te, p_id):.3f} "
          f"(base rates CORD={rate['cord']:.2f} SROIE={rate['sroie']:.2f})")

    # ---- (4) leakage ablation ----
    Xtr_16 = tr[FEATURE_COLS + LEAKY_FEATURE_COLS].fillna(0).values
    Xte_16 = te[FEATURE_COLS + LEAKY_FEATURE_COLS].fillna(0).values
    p16 = rf().fit(Xtr_16, y_tr).predict_proba(Xte_16)[:, 1]
    m, lo, hi = paired(y_te, p16, p_feat)
    print(f"\n(4) Leakage ablation: full-16 {aucs(p16)['pooled']:.3f} vs clean-13 {aucs(p_feat)['pooled']:.3f}"
          f"  (delta {m:+.3f} CI [{lo:+.3f},{hi:+.3f}] -> leakage {'load-bearing' if lo > 0 else 'NOT load-bearing'})")

    pd.DataFrame([
        {"metric": "clean_features_auc_pooled", "value": round(aucs(p_feat)["pooled"], 3)},
        {"metric": "text_lsa_auc_pooled", "value": round(aucs(p_text)["pooled"], 3)},
        {"metric": "features_minus_text", "value": round(paired(y_te, p_feat, p_text)[0], 3)},
        {"metric": "lodo_cord_to_sroie", "value": round(roc_auc_score(
            full[full.dataset == "sroie"].label,
            rf().fit(full[full.dataset == "cord"][FEATURE_COLS].fillna(0), full[full.dataset == "cord"].label)
            .predict_proba(full[full.dataset == "sroie"][FEATURE_COLS].fillna(0))[:, 1]), 3)},
        {"metric": "dataset_identity_auc", "value": round(roc_auc_score(y_te, p_id), 3)},
        {"metric": "full16_auc", "value": round(aucs(p16)["pooled"], 3)},
    ]).to_csv(TABLE_DIR / "exp13_baselines.csv", index=False)
    print(f"\n  -> {TABLE_DIR}/exp13_baselines.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
