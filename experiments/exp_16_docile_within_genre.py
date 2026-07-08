"""EXP-16: DocILE (invoices) within-genre routing -- the third genre.

Runs the FULL pipeline on a random invoice subset, refit within-genre (train the
router on invoices, test on invoices), exactly as for receipts:

  extraction (both tiers, canonical F1, gap)  -> oracle labels
  13 clean pre-inference features (OCR/image/geometric only, no annotations)
  calibrated RF router  -> test AUC + cost savings (frontier + no-peek deployable)

Only the extraction step calls the API (cached, resumable). Features (Tesseract)
and routing are local.

    python experiments/exp_16_docile_within_genre.py                  # 400 train / 200 test
    python experiments/exp_16_docile_within_genre.py --n-train 400 --n-test 200 --workers 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import get_loader
from src.data.docile_loader import _SPLIT_FILES
from src.extraction.evaluation import evaluate
from src.extraction.model_large import LargeModel
from src.extraction.model_small import SmallModel
from src.features.complexity_score import extract_signals
from src.features.ocr_features import extract_ocr_features
from src.features.quality_features import extract_quality_features
from experiments.exp_02_oracle_labeling import DOCILE_SCHEMA
from experiments.exp_03_routing_model import (
    FEATURE_COLS, F1_GAP_TOLERANCE, compute_pareto, _pareto_best_saving)

TAU = 0.02
TABLE_DIR = Path("results/tables")
HARD_COST_CAP = 30.0


def sample_docs(loader, split, n, seed):
    ids = json.load(open(loader.root / _SPLIT_FILES[split], encoding="utf-8"))
    keep = set(random.Random(seed).sample(ids, min(n, len(ids))))
    return list(loader.iter_documents(split, keep=keep))


def label_and_feature(doc, small, large):
    rs = small.extract(doc, schema=DOCILE_SCHEMA)
    rl = large.extract(doc, schema=DOCILE_SCHEMA)
    fs = evaluate({"fields": rs.fields}, doc).f1
    fl = evaluate({"fields": rl.fields}, doc).f1
    row = {"doc_id": doc.doc_id, "split": doc.split,
           "f1_small": round(fs, 4), "f1_large": round(fl, 4), "gap": round(fl - fs, 4),
           "cost_small": round(rs.cost_usd, 6), "cost_large": round(rl.cost_usd, 6),
           "cached": rs.cached and rl.cached}
    row.update(extract_ocr_features(doc))
    row.update(extract_quality_features(doc))
    row.update(extract_signals(doc).as_dict())
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    loader = get_loader("docile")
    docs = ([(d, "train") for d in sample_docs(loader, "train", args.n_train, args.seed)]
            + [(d, "test") for d in sample_docs(loader, "test", args.n_test, args.seed)])
    print(f"DocILE within-genre: {args.n_train} train + {args.n_test} test invoices "
          f"(seed {args.seed})\n")

    small, large = SmallModel(), LargeModel()
    rows, spent, done = [], 0.0, 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(label_and_feature, d, small, large): sp for d, sp in docs}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    m = str(e).lower()
                    if "api_key" in m or "authentication" in m:
                        print("ERROR: ANTHROPIC_API_KEY not set/invalid."); return 2
                    if "rate_limit" in m or "429" in m:
                        print("Rate limit -- re-run with --workers 4 (cache kept)."); return 3
                    raise
                rows.append(r); done += 1
                if not r["cached"]:
                    spent += r["cost_small"] + r["cost_large"]
                if done % 100 == 0:
                    print(f"  {done}/{len(docs)}  spent=${spent:.2f}")
                if spent > HARD_COST_CAP:
                    print(f"\nABORT: ${spent:.2f} exceeded cap ${HARD_COST_CAP}."); return 4
    except KeyboardInterrupt:
        print("Interrupted -- cached extractions kept, safe to resume.")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "oracle_labels_docile.csv", index=False)
    print(f"\n  extraction spend: ${spent:.2f}   labels+features -> oracle_labels_docile.csv")

    tr, te = df[df.split == "train"], df[df.split == "test"]
    Xtr, ytr = tr[FEATURE_COLS].fillna(0).values, (tr.gap > TAU).astype(int).values
    Xte, yte = te[FEATURE_COLS].fillna(0).values, (te.gap > TAU).astype(int).values

    rf = Pipeline([("c", CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        method="isotonic", cv=3))])
    rf.fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]
    lr = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    auc_rf = roc_auc_score(yte, p_rf); auc_lr = roc_auc_score(yte, p_lr)
    print(f"\n  large-required (test): {100*yte.mean():.0f}%")
    print(f"  RF AUC = {auc_rf:.3f}   LR AUC = {auc_lr:.3f}   gain +{auc_rf-auc_lr:.3f}")

    sub = te.copy(); sub["p"] = p_rf
    al_cost, al_qual = sub.cost_large.mean(), sub.f1_large.mean()
    pareto = compute_pareto(sub.rename(columns={"p": "score"}), "score")
    saving, frac = _pareto_best_saving(pareto, al_qual, al_cost)
    # no-peek: threshold from TRAIN, applied to TEST
    subtr = tr.copy(); subtr["p"] = rf.predict_proba(Xtr)[:, 1]
    best_t, best_c = 1.0, np.inf
    for t in np.linspace(0, 1, 101):
        q = np.where(subtr.p >= t, subtr.f1_large, subtr.f1_small).mean()
        if q < subtr.f1_large.mean() - F1_GAP_TOLERANCE:
            continue
        c = np.where(subtr.p >= t, subtr.cost_large, subtr.cost_small).mean()
        if c < best_c:
            best_c, best_t = c, t
    route = sub.p >= best_t
    nopeek_cost = np.where(route, sub.cost_large, sub.cost_small).mean()
    nopeek_qual = np.where(route, sub.f1_large, sub.f1_small).mean()
    nopeek_save = (1 - nopeek_cost / al_cost) * 100
    print(f"  frontier saving : {saving:.0f}%  (routes {frac:.0f}% to large)")
    print(f"  no-peek saving  : {nopeek_save:.0f}%  quality {nopeek_qual:.3f} vs always-large "
          f"{al_qual:.3f} (gap {al_qual-nopeek_qual:.3f}, within {F1_GAP_TOLERANCE}: "
          f"{al_qual-nopeek_qual <= F1_GAP_TOLERANCE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
