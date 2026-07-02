"""EXP-10: De-noise oracle labels by averaging F1 over multiple draws.

The oracle gap g(d)=F1_large-F1_small is a single stochastic draw at default
sampling, and tau=0.02 sits inside decoding-noise scale. This re-extracts the
documents whose label could plausibly flip (the flippable band, |gap_canon| <=
BAND) K extra times per tier, and re-labels each on the MEAN gap over all draws.
A stable-band control sample is also re-drawn to verify those labels don't move.

Draw 0 is the existing single cache (data/processed/extraction_cache/...).
Draws 1..K are fetched fresh (use_cache=False) and stored under
data/processed/extraction_draws/ so re-runs and interruptions are free/safe.

Scoring uses the field-type canonical matcher (evaluation.score_pairs, default).

    python experiments/exp_10_denoise.py --dry-run     # free: counts calls + $
    python experiments/exp_10_denoise.py               # live: makes API calls
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import get_loader
from src.extraction.evaluation import document_to_pairs, prediction_to_pairs, score_pairs
from src.extraction.model_large import LargeModel
from src.extraction.model_small import SmallModel

from experiments.exp_02_oracle_labeling import _SCHEMAS

TAU = 0.02
BAND = 0.10                       # |gap| <= BAND => label could flip
CANON_DIR = Path("results/tables")  # canonical oracle labels are the default
DRAW_ROOT = Path("data/processed/extraction_draws")
SINGLE_ROOT = Path("data/processed/extraction_cache")
OUT_DIR = Path("results/tables")
PAIRS = [("cord", "train"), ("cord", "test"), ("sroie", "train"), ("sroie", "test")]
HARD_COST_CAP = 45.0             # abort live run if projected spend exceeds this


def build_manifest(n_control: int, seed: int = 42, pilot: int = 0) -> pd.DataFrame:
    """Flippable-band docs + a stratified stable-band control, from canonical labels.

    Ceiling-ties (gap==0 and both tiers at F1=1.0) are excluded from flippable:
    they cannot move above 1.0, so their label is stable. If pilot>0, take a
    stratified sample of `pilot` flippable docs across sign buckets
    (exact-zero / small-positive / small-negative) to measure decoding variance.
    """
    frames = []
    for ds, split in PAIRS:
        df = pd.read_csv(CANON_DIR / f"oracle_labels_{ds}_{split}.csv")
        df["dataset"], df["split"] = ds, split
        frames.append(df)
    alld = pd.concat(frames, ignore_index=True)
    alld["abs_gap"] = alld["gap"].abs()

    ceiling = (alld.gap == 0) & (alld.f1_small >= 0.999) & (alld.f1_large >= 0.999)
    flippable = alld[(alld["abs_gap"] <= BAND) & (~ceiling)].copy()
    flippable["role"] = "flippable"

    if pilot:
        rng = np.random.default_rng(seed)
        buckets = [
            flippable[flippable.gap == 0],
            flippable[(flippable.gap > 0)],
            flippable[(flippable.gap < 0)],
        ]
        per = max(1, pilot // 3)
        picks = []
        for b in buckets:
            if len(b):
                k = min(per, len(b))
                picks.append(b.iloc[rng.choice(len(b), k, replace=False)])
        flippable = pd.concat(picks, ignore_index=True)

    stable = alld[alld["abs_gap"] > BAND].copy()
    zero = stable[stable["gap"] == 0]
    clear = stable[stable["abs_gap"] > BAND]
    rng = np.random.default_rng(seed)
    ctrl_parts = []
    for pool in (zero, clear):
        if len(pool):
            k = min(n_control // 2, len(pool))
            ctrl_parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    control = pd.concat(ctrl_parts, ignore_index=True)
    control["role"] = "control"

    man = pd.concat([flippable, control], ignore_index=True)
    return man[["dataset", "split", "doc_id", "gap", "role",
                "cost_small", "cost_large"]]


def _draw_path(tier: str, ds: str, split: str, doc_id: str, k: int) -> Path:
    return DRAW_ROOT / tier / ds / split / f"{doc_id}__draw{k}.json"


def _load_draw0(tier: str, ds: str, split: str, doc_id: str):
    f = SINGLE_ROOT / tier / ds / split / f"{doc_id}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _fetch_draw(model, doc, schema, tier, ds, split, k):
    """Return cached draw k or fetch it fresh (one API call). Caches to disk."""
    p = _draw_path(tier, ds, split, doc.doc_id, k)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), True
    r = model.extract(doc, use_cache=False, schema=schema)
    obj = {"fields": r.fields, "input_tokens": r.input_tokens,
           "output_tokens": r.output_tokens, "cost_usd": r.cost_usd}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return obj, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--draws", type=int, default=5, help="extra draws beyond draw 0")
    ap.add_argument("--control", type=int, default=30)
    ap.add_argument("--pilot", type=int, default=0,
                    help="if >0, sample this many flippable docs (variance pilot)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    man = build_manifest(args.control, pilot=args.pilot)
    n_flip = (man.role == "flippable").sum()
    n_ctrl = (man.role == "control").sum()
    n_docs = len(man)

    # cost projection from existing per-doc costs (each draw ~ one single-pass cost)
    proj = (man["cost_small"].sum() + man["cost_large"].sum()) * args.draws
    n_calls = n_docs * 2 * args.draws

    print(f"Manifest: {n_docs} docs  (flippable={n_flip}, control={n_ctrl})")
    print(f"Draws/tier: {args.draws} extra (+ draw0 from cache = {args.draws+1} total)")
    print(f"Fresh API calls (worst case, nothing cached): {n_calls}")
    print(f"Projected spend (worst case): ${proj:.2f}   [hard cap ${HARD_COST_CAP}]")

    if args.dry_run:
        print("\nDRY RUN — no API calls made. Re-run without --dry-run to execute.")
        return 0

    if proj > HARD_COST_CAP:
        print(f"\nABORT: projection ${proj:.2f} exceeds hard cap ${HARD_COST_CAP}.")
        return 2

    # ---- live ----
    loaders = {}
    for ds, split in PAIRS:
        loaders.setdefault(ds, {})[split] = {
            d.doc_id: d for d in get_loader(ds).load_split(split)
        }
    small, large = SmallModel(), LargeModel()

    tasks = []  # (model, doc, schema, tier, ds, split, k)
    for _, row in man.iterrows():
        ds, split, doc_id = row.dataset, row.split, row.doc_id
        doc = loaders[ds][split].get(doc_id)
        if doc is None:
            continue
        schema = _SCHEMAS[ds]
        for k in range(1, args.draws + 1):
            tasks.append((small, doc, schema, "small", ds, split, k))
            tasks.append((large, doc, schema, "large", ds, split, k))

    spent = 0.0
    done = 0
    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_fetch_draw, m, d, s, t, ds, sp, k):
                    (t, d.doc_id, k) for (m, d, s, t, ds, sp, k) in tasks}
            for fut in as_completed(futs):
                try:
                    obj, was_cached = fut.result()
                    if not was_cached:
                        spent += float(obj.get("cost_usd", 0.0))
                except Exception as e:
                    errors += 1
                    msg = str(e).lower()
                    if "api_key" in msg or "authentication" in msg:
                        print("\nERROR: ANTHROPIC_API_KEY not set/invalid. Aborting.")
                        return 3
                    if "rate_limit" in msg or "429" in msg:
                        print("\nRate limit — re-run (cached draws are kept) with --workers 8")
                        return 4
                    continue
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{len(tasks)} draws fetched  spent=${spent:.2f}")
    except KeyboardInterrupt:
        print(f"\nInterrupted at {done}/{len(tasks)} — cached draws kept, safe to resume.")

    print(f"\nDraws fetched: {done}/{len(tasks)}  new spend=${spent:.2f}  errors={errors}")

    # ---- aggregate: mean F1 over draws 0..K per tier, recompute gap/label ----
    rows = []
    for _, row in man.iterrows():
        ds, split, doc_id, role = row.dataset, row.split, row.doc_id, row.role
        doc = loaders[ds][split].get(doc_id)
        if doc is None:
            continue
        gold = document_to_pairs(doc)
        per_tier = {}
        for tier in ("small", "large"):
            f1s = []
            d0 = _load_draw0(tier, ds, split, doc_id)
            if d0:
                f1s.append(score_pairs(prediction_to_pairs(d0), gold).f1)
            for k in range(1, args.draws + 1):
                p = _draw_path(tier, ds, split, doc_id, k)
                if p.exists():
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    f1s.append(score_pairs(prediction_to_pairs(obj), gold).f1)
            per_tier[tier] = f1s
        if not per_tier["small"] or not per_tier["large"]:
            continue
        fs_mean, fl_mean = float(np.mean(per_tier["small"])), float(np.mean(per_tier["large"]))
        rows.append({
            "doc_id": doc_id, "dataset": ds, "split": split, "role": role,
            "n_draws_small": len(per_tier["small"]), "n_draws_large": len(per_tier["large"]),
            "gap_single": round(float(row.gap), 4),
            "f1_small_mean": round(fs_mean, 4), "f1_large_mean": round(fl_mean, 4),
            "f1_small_std": round(float(np.std(per_tier["small"])), 4),
            "f1_large_std": round(float(np.std(per_tier["large"])), 4),
            "gap_denoised": round(fl_mean - fs_mean, 4),
            "label_single": int(float(row.gap) > TAU),
            "label_denoised": int((fl_mean - fs_mean) > TAU),
        })
    res = pd.DataFrame(rows)
    res["label_changed"] = res.label_single != res.label_denoised
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "denoise_results.csv", index=False)

    flip = res[res.role == "flippable"]
    ctrl = res[res.role == "control"]
    print("\n=== DE-NOISE SUMMARY ===")
    print(f"  flippable docs        : {len(flip)}")
    print(f"  label changes (flip)  : {flip.label_changed.sum()} "
          f"({100*flip.label_changed.mean():.1f}%)")
    print(f"  control docs          : {len(ctrl)}")
    print(f"  control label changes : {ctrl.label_changed.sum()} "
          f"(expect ~0 — validates the stable-band shortcut)")
    print(f"  mean |gap shift|      : {(res.gap_denoised - res.gap_single).abs().mean():.4f}")
    print(f"  mean per-doc F1 std   : small={res.f1_small_std.mean():.3f}  "
          f"large={res.f1_large_std.mean():.3f}  (decoding variance, was hidden)")
    print(f"\n  -> {OUT_DIR}/denoise_results.csv")
    print("  Next: fold gap_denoised into labels and re-run exp_03 (free).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
