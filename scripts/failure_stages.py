"""Label the failure stage of answer-incorrect tool-condition episodes.

For every episode with tools available (conditions tf/tt/rt) whose answer
was wrong, read its trajectory and assign one stage label, following the
PHREEQC-MCQ-200 taxonomy adapted to this tool layer:

- no_tool_use          the model answered without calling any tool
- input_construction   execution was attempted but every attempt failed on
                       argument/schema validation
- execution            execution was attempted and failed inside the kernel
- output_navigation    execution succeeded but the episode ran out of steps,
                       or the model never read the artifacts holding the value
- answer_mapping       execution and artifact reads succeeded; the final
                       answer was still wrong (wrong row, wrong label, wrong
                       transcription)
- guardrail            approval/ambiguous-family failures: the model executed
                       or guessed where the correct behavior was to refuse or
                       ask (labeled from the family, not the trajectory)

Usage:
    python scripts/failure_stages.py EXP_DIR [--out DIR]
    (EXP_DIR is the experiment output directory holding experiment_results.csv)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXEC_TOOLS = {"run_forward", "run_task", "run_design_with_recovery", "calibrate_scm_kinetics"}
READ_TOOLS = {"read_artifact", "list_run_artifacts", "query_past_runs", "recall_session"}
INPUT_ERROR_MARKERS = ["validation error", "field required", "extra inputs", "bad arguments",
                       "invalid tool arguments", "unknown tool", "value error, ", "input_value="]
BEHAVIOR_FAMILIES = {"approval", "ambiguous"}


def classify(episode_dir: Path, family: str) -> str:
    if family in BEHAVIOR_FAMILIES:
        return "guardrail"
    outcome_path = episode_dir / "s" / "run_0" / "ws" / "episode_outcome.json"
    if not outcome_path.exists():
        return "missing_trajectory"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    calls = outcome.get("tool_calls") or []
    if not calls:
        return "no_tool_use"
    exec_calls = [c for c in calls if c.get("tool") in EXEC_TOOLS]
    exec_ok = any(c.get("ok") for c in exec_calls)
    if not exec_ok:
        if not exec_calls:
            return "no_tool_use"
        # inspect trajectory errors to split input construction from kernel failure
        trajectory = episode_dir / "s" / "run_0" / "ws" / "trajectory.jsonl"
        error_text = ""
        if trajectory.exists():
            for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
                event = json.loads(line)
                payload = (event.get("tool_result") or {}).get("payload") or {}
                if payload.get("ok") is False:
                    error_text += " " + str(payload.get("error", "")).lower()
        if any(marker in error_text for marker in INPUT_ERROR_MARKERS):
            return "input_construction"
        return "execution"
    if outcome.get("stop_reason") == "max_steps":
        return "output_navigation"
    read_ok = any(c.get("tool") in READ_TOOLS and c.get("ok") for c in calls)
    return "answer_mapping" if read_ok else "output_navigation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)

    frame = pd.read_csv(exp_dir / "experiment_results.csv")
    failures = frame[
        frame["status"].isin(["passed", "failed"])
        & (frame["answer_correct"] == False)  # noqa: E712 - pandas mask
        & frame["condition"].isin(["tf", "tt", "rt"])
    ].copy()

    labels = []
    for _, row in failures.iterrows():
        episode_dir = exp_dir / str(row["model"]) / f"{row['item']}__{row['condition']}_r{row['repeat']}"
        labels.append(classify(episode_dir, str(row["family"])))
    failures["failure_stage"] = labels

    summary = (
        failures.groupby(["condition", "failure_stage"]).size().unstack(fill_value=0)
    )
    by_model = (
        failures.groupby(["model", "failure_stage"]).size().unstack(fill_value=0)
    )
    pd.set_option("display.width", 200)
    print("=== failure stages by condition ===")
    print(summary.to_string())
    print("\n=== failure stages by model (tf/tt/rt pooled) ===")
    print(by_model.to_string())

    out_dir = Path(args.out) if args.out else exp_dir
    failures.to_csv(out_dir / "failure_stages.csv", index=False)
    print(f"\nwritten -> {out_dir}\\failure_stages.csv ({len(failures)} failures labeled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
