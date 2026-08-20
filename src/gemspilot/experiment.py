"""Experiment matrix driver: models x conditions x scenarios x repeats.

Wraps ``agent_bench.run_agent_bench`` one episode at a time so that runs are
resumable (an episode with an existing report is not re-run), per-model cost
caps are enforced mid-sweep, and results aggregate into a flat CSV for
analysis.

Conditions (output-access ablation + no-tool baseline, short codes keep
Windows paths under MAX_PATH):

- ``tf``  tools on, protocol full
- ``tt``  tools on, protocol toc
- ``nt``  no tools (baseline; same prompt, no tool schemas)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CONDITIONS: dict[str, dict[str, Any]] = {
    "tf": {"protocol": "full", "no_tools": False},
    "tt": {"protocol": "toc", "no_tools": False},
    "nt": {"protocol": "full", "no_tools": True},
}
# Checks that grade the *answer* (vs. trajectory constraints); used to derive
# the primary correctness metric that is comparable across conditions.
ANSWER_CHECK_NAMES = {
    "numeric_answer", "choice_answer", "refusal_language", "no_execution_before_refusal",
}


def load_models(models_config: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(models_config).read_text(encoding="utf-8"))


def load_scenarios(scenario_files: list[str | Path]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in scenario_files:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for scenario in data.get("scenarios") or []:
            if scenario.get("kind") != "agent_qa":
                continue
            scenario_id = str(scenario["id"])
            if scenario_id in seen:
                raise ValueError(f"duplicate scenario id {scenario_id!r} in {path}")
            seen.add(scenario_id)
            scenarios.append(scenario)
    return scenarios


def _episode_row(
    model_entry: dict[str, Any],
    scenario: dict[str, Any],
    condition: str,
    repeat: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    result = report["scenarios"][0]
    checks = result.get("checks") or []
    answer_checks = [c for c in checks if c["name"] in ANSWER_CHECK_NAMES]
    metrics = (result.get("metrics") or [{}])[0]
    return {
        "model": model_entry["label"],
        "tier": model_entry.get("tier"),
        "vendor": model_entry.get("vendor"),
        "item": scenario["id"],
        "family": scenario.get("family"),
        "condition": condition,
        "repeat": repeat,
        "status": result.get("status"),
        "answer_correct": (
            all(c["ok"] for c in answer_checks) if answer_checks else None
        ),
        "checks_passed": sum(1 for c in checks if c["ok"]),
        "checks_total": len(checks),
        "steps": metrics.get("steps"),
        "tool_calls": metrics.get("tool_calls"),
        "unnecessary_calls": metrics.get("unnecessary_calls"),
        "stop_reason": metrics.get("stop_reason"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "cost_usd": metrics.get("cost_usd"),
        "providers": ";".join(metrics.get("providers") or []),
        "reason": result.get("reason"),
    }


def run_experiment(
    models_config: str | Path,
    scenario_files: list[str | Path],
    out: str | Path,
    *,
    repeats: int = 1,
    conditions: list[str] | None = None,
    only_models: list[str] | None = None,
    only_items: list[str] | None = None,
    exclude_items: list[str] | None = None,
    max_steps: int = 12,
) -> dict[str, Any]:
    from .agent_bench import run_agent_bench

    roster = load_models(models_config)
    scenarios = load_scenarios(scenario_files)
    if only_items:
        scenarios = [s for s in scenarios if s["id"] in set(only_items)]
    if exclude_items:
        scenarios = [s for s in scenarios if s["id"] not in set(exclude_items)]
    condition_codes = list(conditions or CONDITIONS)
    for code in condition_codes:
        if code not in CONDITIONS:
            raise ValueError(f"unknown condition {code!r}; choose from {sorted(CONDITIONS)}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped_models: list[dict[str, Any]] = []

    for model_entry in roster["models"]:
        label = str(model_entry["label"])
        if only_models and label not in set(only_models):
            continue
        key_env = model_entry.get("api_key_env") or roster.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            skipped_models.append({"model": label, "reason": f"{key_env} unset"})
            continue
        cap = float(model_entry.get("max_cost_usd", float("inf")))
        spent = 0.0
        model_dir = out_dir / label
        for scenario in scenarios:
            for condition in condition_codes:
                for repeat in range(repeats):
                    episode_id = f"{scenario['id']}__{condition}_r{repeat}"
                    episode_dir = model_dir / episode_id
                    report_path = episode_dir / "bench_report.json"
                    if report_path.exists():
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        row = _episode_row(model_entry, scenario, condition, repeat, report)
                        row["resumed"] = True
                        rows.append(row)
                        spent += row.get("cost_usd") or 0.0
                        continue
                    if spent >= cap:
                        rows.append({
                            "model": label, "tier": model_entry.get("tier"),
                            "vendor": model_entry.get("vendor"),
                            "item": scenario["id"], "family": scenario.get("family"),
                            "condition": condition, "repeat": repeat,
                            "status": "skipped_budget",
                            "reason": f"max_cost_usd={cap} reached (spent={spent:.2f})",
                        })
                        continue
                    episode = dict(scenario)
                    # Short id: the episode directory already encodes identity,
                    # and long nested ids overflow Windows MAX_PATH.
                    episode["id"] = "s"
                    episode["model"] = str(model_entry["id"])
                    episode["max_steps"] = int(scenario.get("max_steps", max_steps))
                    episode["completion_params"] = dict(model_entry.get("params") or {})
                    episode.update(CONDITIONS[condition])
                    report = run_agent_bench({"scenarios": [episode]}, out=episode_dir)
                    row = _episode_row(model_entry, scenario, condition, repeat, report)
                    row["resumed"] = False
                    rows.append(row)
                    spent += row.get("cost_usd") or 0.0
                    print(
                        f"[{label}] {episode_id}: {row['status']} "
                        f"answer_correct={row['answer_correct']} "
                        f"cost=${(row.get('cost_usd') or 0):.4f} (total ${spent:.2f})",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    results_csv = out_dir / "experiment_results.csv"
    frame.to_csv(results_csv, index=False)
    summary = {
        "episodes": len(rows),
        "models": sorted(frame["model"].unique().tolist()) if len(frame) else [],
        "skipped_models": skipped_models,
        "total_cost_usd": round(float(frame.get("cost_usd", pd.Series(dtype=float)).fillna(0).sum()), 4),
        "results_csv": str(results_csv),
    }
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return summary
