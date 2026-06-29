"""EXP-04: Gap-distribution histograms.

Visualizes the per-document difficulty gap = F1(large) - F1(small) for each
dataset. This is the figure behind the "router knows when not to route"
argument: CORD/SROIE have a substantial right tail (docs that genuinely need
the large model), while VRDU is concentrated at gap=0 (a near-zero-gap domain
where routing should not help -- the negative control).

Every value is read directly from results/tables/oracle_labels_{ds}_test.csv,
so the figure is fully reproducible from the oracle label files.

    python experiments/exp_04_gap_histogram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLE_DIR = Path("results/tables")
FIG_DIR   = Path("results/figures")

TAU = 0.02
DATASETS = ["cord", "sroie", "vrdu"]
COLORS   = {"cord": "steelblue", "sroie": "darkorange", "vrdu": "seagreen"}
LABELS   = {"cord": "CORD (receipts)", "sroie": "SROIE (receipts)",
            "vrdu": "VRDU (legal forms)"}


def _load(ds: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / f"oracle_labels_{ds}_test.csv")


def main() -> int:
    data = {ds: _load(ds) for ds in DATASETS}

    # Shared bin edges so the three panels are directly comparable.
    bins = np.arange(-0.55, 0.80, 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        df = data[ds]
        gaps = df["gap"].values
        mean_gap   = gaps.mean()
        pct_large  = 100 * (gaps > TAU).mean()
        pct_zero   = 100 * (gaps == 0).mean()

        ax.hist(gaps, bins=bins, color=COLORS[ds], alpha=0.8,
                edgecolor="white", linewidth=0.5)

        # tau threshold + mean markers
        ax.axvline(TAU, color="red", linestyle="--", lw=1.3,
                   label=f"tau = {TAU}")
        ax.axvline(mean_gap, color="black", linestyle="-", lw=1.3,
                   label=f"mean = {mean_gap:+.3f}")

        ax.set_title(
            f"{LABELS[ds]}\n"
            f"{pct_large:.0f}% large-required  |  {pct_zero:.0f}% exactly zero",
            fontsize=10,
        )
        ax.set_xlabel("Difficulty gap  F1(large) - F1(small)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Number of documents")
    fig.suptitle(
        "Per-document difficulty gap by dataset  "
        "(right tail = docs that need the large model)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "exp04_gap_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    # --- Standalone VRDU-only panel (for the negative-control section) -------
    df = data["vrdu"]
    gaps = df["gap"].values
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(gaps, bins=bins, color=COLORS["vrdu"], alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(TAU, color="red", linestyle="--", lw=1.3, label=f"tau = {TAU}")
    ax.axvline(gaps.mean(), color="black", linestyle="-", lw=1.3,
               label=f"mean = {gaps.mean():+.3f}")
    pct_under_1pt = 100 * (gaps < 0.01).mean()
    pct_zero      = 100 * (gaps == 0).mean()
    ax.set_title(
        f"VRDU difficulty gap (negative control)\n"
        f"{pct_under_1pt:.0f}% gain < 1 F1 point  |  {pct_zero:.0f}% exactly tied",
        fontsize=11,
    )
    ax.set_xlabel("Difficulty gap  F1(large) - F1(small)")
    ax.set_ylabel("Number of documents")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_vrdu = FIG_DIR / "exp04_gap_histogram_vrdu.png"
    plt.savefig(out_vrdu, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_vrdu}")

    # --- Print the exact numbers behind the figure (provenance) --------------
    print("\nProvenance (read from oracle_labels_*_test.csv):")
    print(f"  {'Dataset':6} {'n':>4} {'mean_gap':>9} {'large-req':>10} "
          f"{'gap<0.01':>9} {'gap==0':>8}")
    for ds in DATASETS:
        g = data[ds]["gap"].values
        print(f"  {ds:6} {len(g):>4} {g.mean():>+9.4f} "
              f"{100*(g>TAU).mean():>9.1f}% "
              f"{100*(g<0.01).mean():>8.1f}% {100*(g==0).mean():>7.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
