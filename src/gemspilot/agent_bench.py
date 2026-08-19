"""GEMS-Agent-Bench: scenario benchmark for the agent tool layer (Phase 4).

Separate from the pipeline unit tests, this harness runs end-to-end
scenarios (natural request already parsed into structured inputs) through
the deterministic agent tools and checks expected behavior: outcome status,
recovery actions taken, guardrail effects, and rerun stability. Results are
written as a machine-readable report plus a Markdown summary.

Scenario kinds:

- ``diagnose``          feasibility diagnosis; expects feasibility/selection.
- ``design_recovery``   observe-replan loop; expects status and the exact
                        sequence of applied relaxation kinds.
- ``forward_mock``      forward run; expects status/row counts.
- ``budget_guardrail``  forward run under max_xgems_calls; expects completed
                        and skipped_budget row counts.
- ``session_recall``    run with session logging then recall; expects the
                        recorded event count.

- ``parse``             natural-language request -> task_query preview via
                        the LLM entrance; expects parse status and task type.

A scenario may declare ``requires: [path, ...]`` (files) and/or
``requires_env: [NAME, ...]`` (environment variables, also satisfied by a
local ``.env`` entry); unmet requirements mark the scenario ``skipped``
(with the reason) rather than failed, so the bench degrades gracefully on
machines without the chemistry DB or LLM credentials. ``repeat: N`` reruns the
scenario and additionally checks that repeated outcomes agree (rerun
stability, cache effects included by design).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import agent_tools
from inverse_gems.utils import load_yaml, write_json


def _check(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {"name": name, "expected": expected, "actual": actual, "ok": expected == actual}


def _check_ge(name: str, minimum: Any, actual: Any) -> dict[str, Any]:
    ok = actual is not None and actual >= minimum
    return {"name": name, "expected": f">={minimum}", "actual": actual, "ok": bool(ok)}


def _run_scenario_once(scenario: dict[str, Any], out_dir: Path, run_index: int) -> dict[str, Any]:
    kind = str(scenario["kind"])
    expect = dict(scenario.get("expect") or {})
    run_out = out_dir / f"run_{run_index}"
    checks: list[dict[str, Any]] = []
    signature: dict[str, Any] = {}

    if kind == "diagnose":
        result = agent_tools.diagnose_design_feasibility(
            scenario["design_query"],
            model_registry=scenario.get("model_registry"),
            target_policy=str(scenario.get("target_policy") or "recommended"),
        )
        summary = result["summary"]
        checks.append(_check("tool_ok", True, result["ok"]))
        if "feasible" in expect:
            checks.append(_check("feasible", expect["feasible"], summary.get("feasible")))
        if "selected_id" in expect:
            checks.append(_check("selected_id", expect["selected_id"], summary.get("selected_id")))
        if "blocker_kinds" in expect:
            checks.append(
                _check(
                    "blocker_kinds",
                    sorted(expect["blocker_kinds"]),
                    sorted((summary.get("blocker_histogram") or {}).keys()),
                )
            )
        signature = {"feasible": summary.get("feasible"), "selected_id": summary.get("selected_id")}

    elif kind == "design_recovery":
        result = agent_tools.run_design_with_recovery(
            scenario["design_query"],
            str(run_out),
            str(scenario.get("db") or (run_out / "db")),
            use_mock=bool(scenario.get("use_mock", True)),
            skip_validation=bool(scenario.get("skip_validation", True)),
            max_attempts=int(scenario.get("max_attempts", 3)),
            model_registry=scenario.get("model_registry"),
        )
        summary = result["summary"]
        applied_kinds = [change["kind"] for change in summary.get("changes_applied") or []]
        checks.append(_check("status", expect.get("status", "complete"), summary.get("status")))
        if "changes_applied_kinds" in expect:
            checks.append(_check("changes_applied_kinds", expect["changes_applied_kinds"], applied_kinds))
        if "max_attempt_count" in expect:
            ok = summary.get("attempt_count", 0) <= expect["max_attempt_count"]
            checks.append(
                {
                    "name": "attempt_count_within_bound",
                    "expected": f"<={expect['max_attempt_count']}",
                    "actual": summary.get("attempt_count"),
                    "ok": bool(ok),
                }
            )
        signature = {"status": summary.get("status"), "changes": applied_kinds}

    elif kind == "forward_mock":
        result = agent_tools.run_forward(
            scenario["forward_query"],
            str(run_out),
            str(scenario.get("db") or (run_out / "db")),
            use_mock=True,
        )
        summary = result["summary"]
        checks.append(_check("status", expect.get("status", "complete"), summary.get("status")))
        row_count = (summary.get("result_summary") or {}).get("row_count")
        if "row_count" in expect:
            checks.append(_check("row_count", expect["row_count"], row_count))
        signature = {"status": summary.get("status"), "row_count": row_count}

    elif kind == "budget_guardrail":
        result = agent_tools.run_forward(
            scenario["forward_query"],
            str(run_out),
            str(scenario.get("db") or (run_out / "db")),
            use_mock=True,
            max_xgems_calls=int(scenario["max_xgems_calls"]),
        )
        forward_dir = result["artifacts"].get("forward_dir")
        statuses: list[str] = []
        if forward_dir and (Path(forward_dir) / "time_series.csv").exists():
            frame = pd.read_csv(Path(forward_dir) / "time_series.csv")
            statuses = frame["chemistry_status"].astype(str).tolist()
        checks.append(_check("completed_rows", expect.get("completed_rows"), statuses.count("complete")))
        checks.append(
            _check("skipped_budget_rows", expect.get("skipped_budget_rows"), statuses.count("skipped_budget"))
        )
        signature = {"statuses": statuses}

    elif kind == "session_recall":
        session = str(run_out / "session")
        run_result = agent_tools.run_forward(
            scenario["forward_query"],
            str(run_out),
            str(scenario.get("db") or (run_out / "db")),
            use_mock=True,
            session=session,
        )
        checks.append(_check("run_ok", True, run_result["ok"]))
        recall = agent_tools.recall_session(session)
        event_count = recall["summary"].get("event_count")
        checks.append(_check_ge("event_count", int(expect.get("min_events", 1)), event_count))
        signature = {"event_count": event_count}

    elif kind == "propose":
        result = agent_tools.propose_constraint_relaxation(
            scenario["design_query"],
            model_registry=scenario.get("model_registry"),
            target_policy=str(scenario.get("target_policy") or "recommended"),
        )
        summary = result["summary"]
        kinds = [p["kind"] for p in summary.get("proposals") or []]
        checks.append(_check("tool_ok", True, result["ok"]))
        if "first_proposal_kind" in expect:
            checks.append(_check("first_proposal_kind", expect["first_proposal_kind"], kinds[0] if kinds else None))
        if "proposal_kinds_include" in expect:
            for wanted in expect["proposal_kinds_include"]:
                checks.append(
                    {"name": f"proposes_{wanted}", "expected": wanted, "actual": kinds, "ok": wanted in kinds}
                )
        signature = {"kinds": kinds}

    elif kind == "filter":
        source = run_out / "candidates.csv"
        run_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scenario["rows"]).to_csv(source, index=False)
        result = agent_tools.filter_candidates(str(source), list(scenario["where"]))
        summary = result["summary"]
        checks.append(_check("tool_ok", True, result["ok"]))
        checks.append(_check("matched_rows", expect.get("matched_rows"), summary.get("matched_rows")))
        signature = {"matched_rows": summary.get("matched_rows")}

    elif kind == "past_runs":
        db = str(scenario.get("db") or (run_out / "db"))
        run_result = agent_tools.run_forward(scenario["forward_query"], str(run_out), db, use_mock=True)
        checks.append(_check("run_ok", True, run_result["ok"]))
        result = agent_tools.query_past_runs(
            db,
            material_system=scenario.get("material_system"),
            binder_contains=scenario.get("binder_contains"),
        )
        matched = result["summary"].get("matched_rows")
        checks.append(_check_ge("matched_rows", int(expect.get("min_matched", 1)), matched))
        signature = {"matched_rows": matched}

    elif kind == "parse":
        result = agent_tools.parse_task_query(
            scenario["request"], str(run_out), model=scenario.get("model")
        )
        summary = result["summary"]
        task_type = summary.get("task_type")
        checks.append(_check("tool_ok", True, result["ok"]))
        if "task_type" in expect:
            checks.append(_check("task_type", expect["task_type"], task_type))
        if "task_type_in" in expect:
            allowed = list(expect["task_type_in"])
            checks.append(
                {
                    "name": "task_type_in",
                    "expected": allowed,
                    "actual": task_type,
                    "ok": task_type in allowed,
                }
            )
        signature = {"task_type": task_type, "status": summary.get("status")}

    else:
        checks.append({"name": "known_kind", "expected": "known scenario kind", "actual": kind, "ok": False})

    return {"checks": checks, "signature": signature}


def _env_available(name: str) -> bool:
    import os

    if os.environ.get(name):
        return True
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith(f"{name}="):
                return True
    return False


def _expand_env(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} references in scenario strings."""
    import os

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def run_agent_bench(
    config: str | Path | dict[str, Any],
    *,
    out: str | Path,
) -> dict[str, Any]:
    data = config if isinstance(config, dict) else load_yaml(config)
    scenarios = [_expand_env(s) for s in (data.get("scenarios") or [])]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario.get("id") or f"scenario_{len(results)}")
        scenario_dir = out_dir / scenario_id
        missing = [str(p) for p in scenario.get("requires") or [] if not Path(str(p)).exists()]
        missing_env = [str(n) for n in scenario.get("requires_env") or [] if not _env_available(str(n))]
        if missing or missing_env:
            reasons = []
            if missing:
                reasons.append(f"missing required path(s): {missing}")
            if missing_env:
                reasons.append(f"missing required env var(s): {missing_env}")
            results.append(
                {
                    "id": scenario_id,
                    "kind": scenario.get("kind"),
                    "status": "skipped",
                    "reason": "; ".join(reasons),
                    "checks": [],
                }
            )
            continue
        repeat = max(1, int(scenario.get("repeat", 1)))
        started = time.perf_counter()
        runs: list[dict[str, Any]] = []
        try:
            for run_index in range(repeat):
                runs.append(_run_scenario_once(scenario, scenario_dir, run_index))
        except Exception as exc:  # noqa: BLE001 - a crashing scenario is a failing scenario
            results.append(
                {
                    "id": scenario_id,
                    "kind": scenario.get("kind"),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "checks": [],
                }
            )
            continue
        duration = time.perf_counter() - started
        checks = [check for run in runs for check in run["checks"]]
        stable = all(run["signature"] == runs[0]["signature"] for run in runs)
        if repeat > 1:
            checks.append(
                {
                    "name": "rerun_stability",
                    "expected": "identical outcomes across repeats",
                    "actual": [run["signature"] for run in runs],
                    "ok": stable,
                }
            )
        passed = all(check["ok"] for check in checks)
        results.append(
            {
                "id": scenario_id,
                "kind": scenario.get("kind"),
                "status": "passed" if passed else "failed",
                "repeat": repeat,
                "duration_s": round(duration, 3),
                "checks": checks,
            }
        )

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "scenarios": results,
    }
    write_json(out_dir / "bench_report.json", summary)
    _write_markdown(out_dir / "bench_report.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GEMS-Agent-Bench Report",
        "",
        f"Total: {summary['total']}  ·  Passed: {summary['passed']}  ·  "
        f"Failed: {summary['failed']}  ·  Skipped: {summary['skipped']}",
        "",
        "| Scenario | Kind | Status | Checks | Duration (s) |",
        "|---|---|---|---|---|",
    ]
    for scenario in summary["scenarios"]:
        ok = sum(1 for c in scenario.get("checks") or [] if c["ok"])
        total = len(scenario.get("checks") or [])
        lines.append(
            f"| {scenario['id']} | {scenario.get('kind')} | {scenario['status']} "
            f"| {ok}/{total} | {scenario.get('duration_s', '')} |"
        )
    failed = [s for s in summary["scenarios"] if s["status"] == "failed"]
    if failed:
        lines += ["", "## Failures", ""]
        for scenario in failed:
            lines.append(f"### {scenario['id']}")
            if scenario.get("reason"):
                lines.append(f"- reason: {scenario['reason']}")
            for check in scenario.get("checks") or []:
                if not check["ok"]:
                    lines.append(
                        f"- `{check['name']}`: expected `{check['expected']}`, got `{check['actual']}`"
                    )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
