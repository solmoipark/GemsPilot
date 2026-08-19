"""Phase 3 of the agent roadmap: the autonomous coverage-growth campaign.

"Raise coverage of this target region" becomes a bounded multi-cycle loop
built entirely from existing deterministic machinery:

    per cycle:
      1. analyze the target's nonzero region in the current global model table
      2. acquire candidate chemistries near that region (existing acquisition
         scoring: novelty, domain distance, target-region proximity)
      3. run the batch (mock or real xGEMS), fold results into the DB
      4. refresh the global model table and retrain the global surrogate
      5. re-measure the target metric (R2, nonzero support)

The campaign stops on: cycle limit, candidate budget exhaustion, a reached
R2 goal, or an empty acquisition. Every cycle's stage manifest and the
metric trajectory land in ``campaign_report.json``/``.md`` - the "agent
that grows its own database" artifact, fully auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from inverse_gems.global_chemistry_cycle import run_global_chemistry_acquisition_cycle
from inverse_gems.target_region_analysis import resolve_target_column, write_target_region_analysis
from inverse_gems.utils import write_json


def _global_model_table(db: Path) -> Path:
    return db / "global_chemistry" / "global_model_table.csv"


def _target_metrics_path(db: Path) -> Path:
    return db / "global_chemistry" / "global_surrogate" / "target_metrics.csv"


def read_target_metric(db: str | Path, target: str) -> dict[str, Any]:
    """Read the current global-surrogate metric row for one target."""
    db = Path(db)
    table_path = _global_model_table(db)
    metrics_path = _target_metrics_path(db)
    if not table_path.exists():
        raise FileNotFoundError(f"Global model table not found: {table_path}")
    frame = pd.read_csv(table_path, nrows=1)
    target_column = resolve_target_column(frame, target)
    result: dict[str, Any] = {"target_column": target_column}
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        row = metrics[metrics["target"] == target_column]
        if not row.empty:
            record = row.iloc[0].to_dict()
            result.update(
                {
                    "r2": float(record["r2"]) if pd.notna(record.get("r2")) else None,
                    "rmse": float(record["rmse"]) if pd.notna(record.get("rmse")) else None,
                    "nonzero_count": int(record.get("full_nonzero_count") or 0),
                    "nonzero_fraction": float(record.get("full_nonzero_fraction") or 0.0),
                    "n_total": int(record.get("n_total") or 0),
                }
            )
    return result


def run_coverage_campaign(
    *,
    target: str,
    db: str | Path,
    out: str | Path,
    cycles: int = 2,
    candidates_per_cycle: int = 10,
    max_total_candidates: int | None = None,
    stop_r2: float | None = None,
    recipes_csv: str | Path | None = None,
    candidate_table: str | Path | None = None,
    candidate_source: str = "pool",
    generate_n: int | None = None,
    generate_seed: int = 42,
    sampling_config: str | Path | None = None,
    use_mock: bool = True,
    dat_lst: str | Path | None = None,
    run_batch: bool = True,
    refresh: bool = True,
    train_surrogate: bool = True,
    retry_water_on_failure: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Grow global-DB coverage of ``target`` over bounded acquisition cycles.

    ``candidate_source`` selects where cycle candidates come from:
    - ``"pool"`` (default): select from an existing recipes_csv pool.
    - ``"region_generate"``: generate ``generate_n`` fresh recipes per cycle
      inside the target region (profile auto-derived from the region
      analysis), so acquisition is not limited by pool coverage.

    Note: cycle results (including mock ones when ``use_mock=True``) are
    written into ``db``. Point demos at a scratch copy of the DB, not the
    curated real one.
    """
    if candidate_source not in {"pool", "region_generate"}:
        raise ValueError("candidate_source must be 'pool' or 'region_generate'.")
    db_path = Path(db)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = read_target_metric(db_path, target)
    target_column = baseline["target_column"]
    trajectory: list[dict[str, Any]] = []
    warnings: list[str] = []
    stop_reason = "cycle_limit_reached"
    candidates_used = 0

    if use_mock:
        warnings.append(
            f"use_mock=True: mock xGEMS rows are being written into {db_path}; "
            "use a scratch DB copy for demonstrations."
        )

    for cycle_index in range(int(cycles)):
        cycle_dir = out_dir / f"cycle_{cycle_index}"
        budget_left = (
            None
            if max_total_candidates is None
            else max(0, int(max_total_candidates) - candidates_used)
        )
        request = (
            int(candidates_per_cycle)
            if budget_left is None
            else min(int(candidates_per_cycle), budget_left)
        )
        if request <= 0:
            stop_reason = "candidate_budget_exhausted"
            break

        region_dir = write_target_region_analysis(
            model_table=_global_model_table(db_path),
            target=target_column,
            out=cycle_dir / "target_region",
        )
        region_table = Path(region_dir) / "target_region_nonzero_rows.csv"

        cycle_recipes = recipes_csv
        if candidate_source == "region_generate":
            from .region_sampling import generate_region_candidates

            cycle_recipes = generate_region_candidates(
                region_dir=region_dir,
                out=cycle_dir / "region_generation",
                n=int(generate_n or max(4 * request, 12)),
                seed=int(generate_seed) + cycle_index,
                sampling_config=sampling_config,
            )

        cycle_out = run_global_chemistry_acquisition_cycle(
            db=db_path,
            out=cycle_dir,
            recipes_csv=cycle_recipes,
            candidate_table=candidate_table,
            max_candidates=request,
            dat_lst=dat_lst,
            use_mock=use_mock,
            run_batch=run_batch,
            refresh=refresh,
            train_surrogate=train_surrogate,
            coverage=False,
            fail_fast=fail_fast,
            retry_water_on_failure=retry_water_on_failure,
            priority_targets=[target_column],
            target_region_table=[region_table],
        )
        cycle_summary_path = Path(cycle_out) / "global_acquisition_cycle_summary.json"
        cycle_summary: dict[str, Any] = {}
        if cycle_summary_path.exists():
            import json

            cycle_summary = json.loads(cycle_summary_path.read_text(encoding="utf-8"))
        acquired = 0
        for stage in cycle_summary.get("stages") or []:
            if stage.get("name") == "acquire_candidates":
                acquired = int(stage.get("rows") or 0)
        candidates_used += acquired

        metric = read_target_metric(db_path, target)
        entry = {
            "cycle": cycle_index,
            "cycle_dir": str(cycle_out),
            "candidates_requested": request,
            "candidates_acquired": acquired,
            "r2": metric.get("r2"),
            "nonzero_count": metric.get("nonzero_count"),
            "nonzero_fraction": metric.get("nonzero_fraction"),
            "stage_statuses": {
                str(stage.get("name")): str(stage.get("status"))
                for stage in cycle_summary.get("stages") or []
            },
        }
        if trajectory:
            prev = trajectory[-1]
            if entry["r2"] is not None and prev.get("r2") is not None:
                entry["delta_r2"] = entry["r2"] - prev["r2"]
        elif baseline.get("r2") is not None and entry["r2"] is not None:
            entry["delta_r2"] = entry["r2"] - baseline["r2"]
        trajectory.append(entry)
        warnings.extend(str(item) for item in cycle_summary.get("warnings") or [])

        if acquired == 0:
            stop_reason = "acquisition_empty"
            break
        if stop_r2 is not None and entry["r2"] is not None and entry["r2"] >= float(stop_r2):
            stop_reason = "r2_goal_reached"
            break

    final = read_target_metric(db_path, target)
    report = {
        "target": target,
        "target_column": target_column,
        "db": str(db_path),
        "use_mock": use_mock,
        "cycles_run": len(trajectory),
        "candidates_used": candidates_used,
        "stop_reason": stop_reason,
        "baseline": baseline,
        "final": final,
        "improvement": {
            "delta_r2": (final.get("r2") - baseline["r2"])
            if final.get("r2") is not None and baseline.get("r2") is not None
            else None,
            "delta_nonzero_count": (final.get("nonzero_count") or 0)
            - (baseline.get("nonzero_count") or 0),
        },
        "trajectory": trajectory,
        "warnings": warnings,
    }
    write_json(out_dir / "campaign_report.json", report)
    _write_markdown(out_dir / "campaign_report.md", report)
    report["campaign_report"] = str(out_dir / "campaign_report.json")
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    baseline = report["baseline"]
    final = report["final"]

    def _fmt(value: Any) -> str:
        return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))

    lines = [
        f"# Coverage Campaign: `{report['target_column']}`",
        "",
        f"- DB: `{report['db']}`  ·  mock: `{report['use_mock']}`",
        f"- Cycles run: `{report['cycles_run']}`  ·  candidates used: `{report['candidates_used']}`",
        f"- Stop reason: `{report['stop_reason']}`",
        f"- R2: `{_fmt(baseline.get('r2'))}` → `{_fmt(final.get('r2'))}`",
        f"- Nonzero support: `{baseline.get('nonzero_count')}` → `{final.get('nonzero_count')}`",
        "",
        "| Cycle | Requested | Acquired | R2 | Nonzero | ΔR2 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in report["trajectory"]:
        lines.append(
            f"| {entry['cycle']} | {entry['candidates_requested']} | {entry['candidates_acquired']} "
            f"| {_fmt(entry.get('r2'))} | {entry.get('nonzero_count')} | {_fmt(entry.get('delta_r2'))} |"
        )
    if report["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in report["warnings"][:20]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
