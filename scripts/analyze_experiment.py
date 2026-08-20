"""Aggregate experiment_results.csv into per-model condition tables.

Computes, per model: answer accuracy under each condition (tf/tt/nt),
grounding gain (tf - nt), the output-protocol delta (tt - tf), mean steps
and tool calls under tf, and total cost. Also breaks accuracy down by
scenario family.

Usage:
    python scripts/analyze_experiment.py RESULTS_CSV [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ran = frame[frame["status"].isin(["passed", "failed"])].copy()
    ran["answer_correct"] = ran["answer_correct"].astype("boolean")

    by_condition = (
        ran.pivot_table(
            index=["model", "tier"], columns="condition",
            values="answer_correct", aggfunc="mean",
        )
        .rename(columns={"tf": "acc_tf", "tt": "acc_tt", "nt": "acc_nt"})
    )
    for column in ["acc_tf", "acc_tt", "acc_nt"]:
        if column not in by_condition:
            by_condition[column] = float("nan")
    by_condition["gain_tf_vs_nt"] = by_condition["acc_tf"] - by_condition["acc_nt"]
    by_condition["delta_toc"] = by_condition["acc_tt"] - by_condition["acc_tf"]

    tf = ran[ran["condition"] == "tf"]
    extras = tf.groupby(["model", "tier"]).agg(
        steps_tf=("steps", "mean"),
        tool_calls_tf=("tool_calls", "mean"),
    )
    cost = ran.groupby(["model", "tier"]).agg(cost_usd=("cost_usd", "sum"))
    episodes = ran.groupby(["model", "tier"]).size().rename("episodes")
    model_table = by_condition.join([extras, cost, episodes]).reset_index()
    tier_order = {"frontier": 0, "mid": 1, "small": 2, "anchor": 3}
    model_table = model_table.sort_values(
        by=["tier", "acc_tf"], key=lambda s: s.map(tier_order) if s.name == "tier" else -s
    ).reset_index(drop=True)

    family_table = (
        ran[ran["condition"] == "tf"]
        .pivot_table(index="family", columns="model", values="answer_correct", aggfunc="mean")
        .reset_index()
    )
    return model_table, family_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.results_csv)
    model_table, family_table = summarize(frame)

    pd.set_option("display.width", 200)
    print("=== accuracy by model x condition ===")
    print(model_table.round(3).to_string(index=False))
    print("\n=== tf accuracy by family x model ===")
    print(family_table.round(2).to_string(index=False))

    skipped = frame[~frame["status"].isin(["passed", "failed"])]
    if len(skipped):
        print(f"\nnot run: {len(skipped)} episodes "
              f"({skipped['status'].value_counts().to_dict()})")

    out_dir = Path(args.out) if args.out else Path(args.results_csv).parent
    model_table.to_csv(out_dir / "summary_by_model.csv", index=False)
    family_table.to_csv(out_dir / "summary_by_family.csv", index=False)
    totals = {
        "episodes_run": int(frame["status"].isin(["passed", "failed"]).sum()),
        "total_cost_usd": round(float(frame["cost_usd"].fillna(0).sum()), 4),
    }
    (out_dir / "summary_totals.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    print(f"\ntotals: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
