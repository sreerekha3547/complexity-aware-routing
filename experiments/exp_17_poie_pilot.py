"""EXP-17: POIE (nutrition labels) feature-signal pilot.

The invoice pilot passed a HEADROOM check but the full run then showed the
features carry no signal (AUC ~0.5). So this pilot checks BOTH, on a larger
random sample: (a) headroom exists, AND (b) the 13 clean features actually
predict which documents are hard (5-fold CV AUC significantly above chance).

Go/no-go (pre-registered): proceed to the full run only if
    large-required > 20%  AND  5-fold CV feature AUC lower 95% CI bound > 0.50.

Canonical F1; extractions cached per (tier, doc). ~120 docs, ~$3-4.

    python experiments/exp_17_poie_pilot.py            # 120 random train docs
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from src.data import get_loader
from src.data.poie_loader import _SPLIT_FILES
from src.extraction.evaluation import evaluate
from src.extraction.model_large import LargeModel
from src.extraction.model_small import SmallModel
from src.features.complexity_score import extract_signals
from src.features.ocr_features import extract_ocr_features
from src.features.quality_features import extract_quality_features
from experiments.exp_02_oracle_labeling import POIE_SCHEMA
from experiments.exp_03_routing_model import FEATURE_COLS

TAU = 0.02
HARD_COST_CAP = 6.0


def _one(doc, small, large):
    rs = small.extract(doc, schema=POIE_SCHEMA)
    rl = large.extract(doc, schema=POIE_SCHEMA)
    fs = evaluate({"fields": rs.fields}, doc).f1
    fl = evaluate({"fields": rl.fields}, doc).f1
    row = {"f1_small": fs, "f1_large": fl, "gap": fl - fs,
           "cost": rs.cost_usd + rl.cost_usd, "cached": rs.cached and rl.cached}
    row.update(extract_ocr_features(doc))
    row.update(extract_quality_features(doc))
    row.update(extract_signals(doc).as_dict())
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    loader = get_loader("poie")
    ids = [json.loads(l)["file_name"] for l in open(loader.root / _SPLIT_FILES["train"], encoding="utf-8")]
    ids = [Path(f).stem for f in ids]
    keep = set(random.Random(args.seed).sample(ids, min(args.n, len(ids))))
    docs = list(loader.iter_documents("train", keep=keep))
    print(f"POIE train: {len(ids)} docs; RANDOM sample of {len(docs)} (seed={args.seed})\n")

    small, large = SmallModel(), LargeModel()
    rows, spent = [], 0.0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_one, d, small, large): d for d in docs}
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
                rows.append(r)
                if not r["cached"]:
                    spent += r["cost"]
                if spent > HARD_COST_CAP:
                    print(f"\nABORT: ${spent:.2f} exceeded cap ${HARD_COST_CAP}."); return 4
    except KeyboardInterrupt:
        print("Interrupted -- cache kept.")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/poie_pilot.csv", index=False)
    gaps = df.gap.values
    large_req = float((gaps > TAU).mean())

    # feature-signal: 5-fold CV AUC on the pilot sample
    X = df[FEATURE_COLS].fillna(0).values
    y = (gaps > TAU).astype(int)
    auc_lo = auc = float("nan")
    if len(np.unique(y)) > 1 and y.sum() >= 5 and (len(y) - y.sum()) >= 5:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        p = cross_val_predict(RandomForestClassifier(200, class_weight="balanced", random_state=42),
                              X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) > 1:
                bs.append(roc_auc_score(y[i], p[i]))
        auc_lo = np.percentile(bs, 2.5)

    print(f"\n{'='*52}")
    print(f"  n                 : {len(df)}")
    print(f"  F1 small / large  : {df.f1_small.mean():.3f} / {df.f1_large.mean():.3f}")
    print(f"  mean gap          : {gaps.mean():+.3f}   large-required: {100*large_req:.0f}%")
    print(f"  FEATURE CV AUC    : {auc:.3f}   (95% CI lower bound {auc_lo:.3f})")
    print(f"  cost/doc (both)   : ${df.cost.mean():.4f}   pilot: ${df.cost.sum():.2f}")
    go = large_req > 0.20 and auc_lo > 0.50
    print(f"\n  GO/NO-GO (large-req>20% AND feature-AUC CI>0.50): "
          f"{'GO -- features carry signal, proceed' if go else 'NO-GO -- no usable feature signal'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
