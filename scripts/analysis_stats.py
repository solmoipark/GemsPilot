"""Repeat-aware statistics for the experiment matrix.

Produces, from experiment_results.csv:

1. stats_by_model_condition.csv — accuracy mean±95% CI across repeats
   (t-distribution, n = number of repeats), plus episode counts.
2. gain_loss_retention.csv — per model and tools-on condition (tf/tt/rt),
   the PHREEQC-MCQ-200 decomposition against the no-tool baseline using
   item-level majority votes over repeats: gained (wrong@nt -> right@cond),
   lost (right@nt -> wrong@cond), retention = kept / right@nt.

Usage:
    python scripts/analysis_stats.py RESULTS_CSV [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}  # two-sided 95%


def _majority(series: pd.Series) -> bool:
    values = series.dropna().astype(bool)
    if not len(values):
        return False
    return values.sum() * 2 > len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.results_csv)
    ran = frame[frame["status"].isin(["passed", "failed"])].copy()
    ran["answer_correct"] = ran["answer_correct"].map(
        {True: True, False: False, "True": True, "False": False}
    )

    # --- 1. accuracy mean ± 95% CI across repeats -------------------------
    per_repeat = (
        ran.groupby(["model", "tier", "condition", "repeat"])["answer_correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    rows = []
    for (model, tier, condition), group in per_repeat.groupby(["model", "tier", "condition"]):
        acc = group["accuracy"].to_numpy(dtype=float)
        n = len(acc)
        mean = float(acc.mean())
        if n > 1:
            half = T95.get(n - 1, 1.96) * float(acc.std(ddof=1)) / np.sqrt(n)
        else:
            half = float("nan")
        rows.append({
            "model": model, "tier": tier, "condition": condition,
            "repeats": n, "accuracy_mean": round(mean, 4),
            "ci95_half": round(half, 4),
            "accuracy_sd": round(float(acc.std(ddof=1)) if n > 1 else float("nan"), 4),
        })
    stats = pd.DataFrame(rows)

    # --- 2. gain / loss / retention vs the no-tool baseline ---------------
    majority = (
        ran.groupby(["model", "tier", "condition", "item"])["answer_correct"]
        .apply(_majority)
        .reset_index(name="correct")
    )
    decomposition = []
    for (model, tier), group in majority.groupby(["model", "tier"]):
        pivot = group.pivot_table(index="item", columns="condition",
                                  values="correct", aggfunc="first")
        if "nt" not in pivot:
            continue
        base = pivot["nt"].fillna(False).astype(bool)
        for condition in [c for c in ["tf", "tt", "rt"] if c in pivot]:
            with_tools = pivot[condition].fillna(False).astype(bool)
            gained = int(((~base) & with_tools).sum())
            lost = int((base & (~with_tools)).sum())
            kept = int((base & with_tools).sum())
            base_right = int(base.sum())
            decomposition.append({
                "model": model, "tier": tier, "condition": condition,
                "items": int(len(pivot)),
                "baseline_right": base_right,
                "gained": gained, "lost": lost, "kept": kept,
                "retention": round(kept / base_right, 3) if base_right else float("nan"),
                "net": gained - lost,
            })
    decomp = pd.DataFrame(decomposition)

    pd.set_option("display.width", 220)
    print("=== accuracy mean ± 95% CI (across repeats) ===")
    wide = stats.pivot_table(index=["model", "tier"], columns="condition",
                             values=["accuracy_mean", "ci95_half"])
    print(wide.round(3).to_string())
    print("\n=== gain / loss / retention vs no-tool baseline ===")
    print(decomp.to_string(index=False))

    out_dir = Path(args.out) if args.out else Path(args.results_csv).parent
    stats.to_csv(out_dir / "stats_by_model_condition.csv", index=False)
    decomp.to_csv(out_dir / "gain_loss_retention.csv", index=False)
    print(f"\nwritten -> {out_dir}\\stats_by_model_condition.csv, gain_loss_retention.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
