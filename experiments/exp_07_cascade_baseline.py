"""EXP-07: Cascade baseline (reviewer R2).

The paper argues that pre-inference routing beats a confidence cascade because a
cascade pays for the cheap-model call on *every* document before deciding whether
to escalate. This quantifies that argument exactly, with no new API calls.

For a fixed escalation decision r(d) in {small, escalate}:

  single-tier router (ours):   cost = r ? c_large : c_small ;  quality = r ? f1_large : f1_small
  cascade (always run small):  cost = c_small + (r ? c_large : 0) ; quality = r ? f1_large : f1_small

Both achieve the SAME quality for the same decisions; the cascade simply pays an
extra c_small on every escalated document. Using the router's own probability as
the (best-case) escalation trigger, we sweep the threshold and compare the
cost--quality frontiers and the cost saving at the quality-tolerance operating
point.

Note: a cascade triggered on the cheap model's *output* confidence (logprobs)
could discriminate better, but (a) it still pays the c_small floor on every
document, and (b) those logprobs were not stored. We therefore give the cascade
the router's pre-inference signal and report the structural cost penalty, which
no output-confidence trigger can remove.

Outputs
-------
results/tables/exp07_cascade_baseline.csv

Run
---
    python experiments/exp_07_cascade_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from experiments.exp_03_routing_model import F1_GAP_TOLERANCE

TABLE_DIR = Path("results/tables")
PRED = TABLE_DIR / "routing_predictions.csv"
SCORE = "p_large_rf_cal"


def best_saving(df: pd.DataFrame, cost_col: str, al_cost: float,
                al_qual: float) -> tuple[float | None, float | None]:
    """Min-cost threshold whose quality stays within delta of always-large."""
    best_s, best_t = None, None
    for t in np.linspace(0, 1, 101):
        route = df[SCORE] >= t
        qual = np.where(route, df["f1_large"], df["f1_small"]).mean()
        if qual < al_qual - F1_GAP_TOLERANCE:
            continue
        if cost_col == "single":
            cost = np.where(route, df["cost_large"], df["cost_small"]).mean()
        else:  # cascade always pays small, plus large when escalating
            cost = (df["cost_small"] + np.where(route, df["cost_large"], 0)).mean()
        saving = (1 - cost / al_cost) * 100
        if best_s is None or saving > best_s:
            best_s, best_t = saving, t
    return (round(best_s, 1) if best_s is not None else None,
            round(best_t, 2) if best_t is not None else None)


def oracle_savings(df: pd.DataFrame, al_cost: float) -> tuple[float, float]:
    """Savings when escalation is the perfect oracle label (best possible trigger).

    Returns (oracle_single_route, oracle_cascade). The oracle cascade is the best
    physically possible confidence cascade: it escalates exactly the documents
    that need the large model, yet still pays the cheap tier on every document.
    """
    route = df["label"] == 1
    single = (1 - np.where(route, df["cost_large"], df["cost_small"]).mean() / al_cost) * 100
    cascade = (1 - (df["cost_small"] + np.where(route, df["cost_large"], 0)).mean() / al_cost) * 100
    return round(single, 1), round(cascade, 1)


def main() -> int:
    print("EXP-07: Cascade baseline (with oracle upper bound)\n")
    if not PRED.exists():
        print(f"ERROR: {PRED} not found -- run exp_03 first.")
        return 1
    df = pd.read_csv(PRED)

    rows = []
    print(f"{'dataset':8} {'RF route':>9} {'RF cascade':>11} "
          f"{'oracle cascade':>15} {'oracle route':>13}")
    print("-" * 60)
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        al_cost = sub["cost_large"].mean()
        al_qual = sub["f1_large"].mean()

        router_s, _  = best_saving(sub, "single",  al_cost, al_qual)
        cascade_s, _ = best_saving(sub, "cascade", al_cost, al_qual)
        orc_route, orc_cascade = oracle_savings(sub, al_cost)

        rows.append({"dataset": ds, "router_saving": router_s,
                     "cascade_saving": cascade_s,
                     "oracle_cascade_saving": orc_cascade,
                     "oracle_route_saving": orc_route,
                     "cascade_floor_saving": round((1 - sub["cost_small"].mean() / al_cost) * 100, 1)})
        cs = f"{cascade_s:.0f}%" if cascade_s is not None else "none"
        print(f"{ds:8} {router_s:>8.0f}% {cs:>11} {orc_cascade:>14.0f}% {orc_route:>12.0f}%")

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "exp07_cascade_baseline.csv", index=False)
    print(f"\n  -> {TABLE_DIR}/exp07_cascade_baseline.csv")
    print("\nVERDICT")
    print("  RF route    = pre-inference routing, pay exactly one tier (ours).")
    print("  RF cascade  = same signal, but pay cheap tier every doc + large on escalated.")
    print("  oracle cascade = BEST possible cascade (perfect trigger) -- still pays the")
    print("    speculative cheap call. Where we beat it (frequent escalation), no")
    print("    output-confidence cascade can win. (Anthropic exposes no token logprobs,")
    print("    so the textbook logprob cascade is not implementable here regardless.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
