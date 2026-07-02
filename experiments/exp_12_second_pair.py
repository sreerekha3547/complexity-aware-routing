"""EXP-12: Second model pair (Haiku vs Sonnet 4.6) — cross-pair generalization.

The paper's reusability claim ("refit the router per model pair") is tested on
exactly one pair (Haiku vs Opus). This runs a second pair to show the
methodology transfers: small = Haiku 4.5 (cached under tier "small"), large =
Sonnet 4.6 (new, cached under "large_sonnet"). Only Sonnet extractions cost
money; Haiku is reused from the existing cache.

Scores with the field-type canonical matcher (evaluation.score_pairs, default),
builds oracle labels on the Haiku-vs-Sonnet gap in results/tables_pair2/, then
runs the router on this pair (refit on its own gaps). Extraction is paid on the
first run and cached thereafter; the labeling + routing steps are local/free.

    python experiments/exp_12_second_pair.py --dry-run   # free: estimate cost
    python experiments/exp_12_second_pair.py             # live: extract Sonnet
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data import get_loader
from src.extraction.evaluation import document_to_pairs, prediction_to_pairs, score_pairs
from src.extraction.model_small import SmallModel
from src.extraction.model_sonnet import SonnetModel

from experiments.exp_02_oracle_labeling import _SCHEMAS

TAU = 0.02
PAIRS = [("cord", "train"), ("cord", "test"), ("sroie", "train"), ("sroie", "test")]
SMALL_CACHE = Path("data/processed/extraction_cache/small")
OUT_DIR = Path("results/tables_pair2")
HARD_COST_CAP = 25.0
# Opus large cost on CORD+SROIE train+test (from oracle_labels), for projection.
# Sonnet input/output both price at 0.6x of Opus, so Sonnet ~= 0.6 * this.
_OPUS_LARGE_REF = {"cord_train": 12.048, "cord_test": 1.457,
                   "sroie_train": 6.524, "sroie_test": 3.602}


def _small_cached(ds, split, doc_id):
    f = SMALL_CACHE / ds / split / f"{doc_id}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if args.dry_run:
        proj = 0.6 * sum(_OPUS_LARGE_REF.values())
        n = 0
        for ds, split in PAIRS:
            n += len(get_loader(ds).load_split(split))
        print(f"Second pair: small=Haiku (cached), large=Sonnet 4.6 (new)")
        print(f"Docs to extract with Sonnet: {n}  (Haiku reused from cache)")
        print(f"Projected Sonnet spend: ~${proj:.2f}   [hard cap ${HARD_COST_CAP}]")
        print("\nDRY RUN — no API calls. Re-run without --dry-run to execute.")
        return 0

    sonnet = SonnetModel()
    schema_by_ds = _SCHEMAS

    # ---- extract Sonnet (paid; cached + resumable) ----
    tasks = []
    docs_by = {}
    for ds, split in PAIRS:
        docs = get_loader(ds).load_split(split)
        docs_by[(ds, split)] = docs
        for d in docs:
            tasks.append((d, ds, split))
    spent = 0.0
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(sonnet.extract, d, schema=schema_by_ds[ds]): (d, ds, split)
                    for (d, ds, split) in tasks}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    msg = str(e).lower()
                    if "api_key" in msg or "authentication" in msg:
                        print("\nERROR: ANTHROPIC_API_KEY not set/invalid. Aborting.")
                        return 3
                    if "rate_limit" in msg or "429" in msg:
                        print("\nRate limit — re-run with --workers 6 (cache kept).")
                        return 4
                    raise
                if not r.cached:
                    spent += r.cost_usd
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{len(tasks)} Sonnet extractions  spent=${spent:.2f}")
                if spent > HARD_COST_CAP:
                    print(f"\nABORT: spend ${spent:.2f} exceeded cap ${HARD_COST_CAP}.")
                    return 2
    except KeyboardInterrupt:
        print(f"\nInterrupted at {done}/{len(tasks)} — cache kept, safe to resume.")

    print(f"\nSonnet extractions done: {done}/{len(tasks)}  new spend=${spent:.2f}")

    # ---- canonical oracle labels (Haiku small vs Sonnet large), free ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy("results/tables/feature_table.csv", OUT_DIR / "feature_table.csv")
    sonnet_cache = Path("data/processed/extraction_cache/large_sonnet")
    for ds, split in PAIRS:
        rows = []
        for d in docs_by[(ds, split)]:
            cs = _small_cached(ds, split, d.doc_id)
            lf = sonnet_cache / ds / split / f"{d.doc_id}.json"
            if cs is None or not lf.exists():
                continue
            cl = json.loads(lf.read_text(encoding="utf-8"))
            gold = document_to_pairs(d)
            fs = score_pairs(prediction_to_pairs(cs), gold).f1
            fl = score_pairs(prediction_to_pairs(cl), gold).f1
            rows.append({
                "doc_id": d.doc_id, "split": split,
                "f1_small": round(fs, 4), "f1_large": round(fl, 4),
                "gap": round(fl - fs, 4),
                "cost_small": round(float(cs.get("cost_usd", 0)), 6),
                "cost_large": round(float(cl.get("cost_usd", 0)), 6),
                "tier_label": "large-required" if (fl - fs) > TAU else "small-sufficient",
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / f"oracle_labels_{ds}_{split}.csv", index=False)
        if len(df):
            print(f"  {ds}/{split}: F1 Haiku={df.f1_small.mean():.3f} "
                  f"Sonnet={df.f1_large.mean():.3f} gap={df.gap.mean():+.3f} "
                  f"large-req={100*(df.tier_label=='large-required').mean():.0f}% (n={len(df)})")
    print(f"\nWrote Haiku-vs-Sonnet canonical labels to {OUT_DIR}/")

    # ---- routing on the second pair (local, free) ----
    import experiments.exp_03_routing_model as exp03
    exp03.TABLE_DIR = OUT_DIR
    exp03.FIG_DIR = Path("results/figures_pair2")
    exp03.FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n>>> Routing on the Haiku-vs-Sonnet pair (refit on this pair's gaps):\n")
    exp03.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
