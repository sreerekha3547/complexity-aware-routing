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


def main() -> int:
    print("EXP-07: Cascade baseline\n")
    if not PRED.exists():
        print(f"ERROR: {PRED} not found -- run exp_03 first.")
        return 1
    df = pd.read_csv(PRED)

    rows = []
    print(f"{'dataset':8} {'router save':>11} {'cascade save':>13} "
          f"{'always-small floor':>19}")
    print("-" * 56)
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        al_cost = sub["cost_large"].mean()
        al_qual = sub["f1_large"].mean()
        as_cost = sub["cost_small"].mean()
        # always-small floor for a cascade, expressed as saving vs always-large
        floor_saving = (1 - as_cost / al_cost) * 100

        router_s, _  = best_saving(sub, "single",  al_cost, al_qual)
        cascade_s, _ = best_saving(sub, "cascade", al_cost, al_qual)

        rows.append({"dataset": ds, "router_saving": router_s,
                     "cascade_saving": cascade_s,
                     "cascade_floor_saving": round(floor_saving, 1)})
        cs = f"{cascade_s:.0f}%" if cascade_s is not None else "none"
        print(f"{ds:8} {router_s:>10.0f}% {cs:>13} {floor_saving:>18.0f}%")

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "exp07_cascade_baseline.csv", index=False)
    print(f"\n  -> {TABLE_DIR}/exp07_cascade_baseline.csv")
    print("\nVERDICT")
    print("  router_saving = pay exactly one tier (ours).")
    print("  cascade_saving = pay cheap tier on every doc + expensive on escalated.")
    print("  The cascade can never beat the always-small cost floor, and pays an")
    print("  extra cheap-inference cost on every escalated document; pre-inference")
    print("  routing avoids that speculative call entirely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
