"""Framework-neutral agent tool layer (roadmap Phase 1).

Each tool is a plain function taking JSON-serializable arguments and returning
a standardized ToolResult dict:

    {
        "contract": "inverse-gems-tool/1.0",
        "tool": str,
        "ok": bool,
        "summary": dict,        # small, structured, safe to inline in context
        "artifacts": dict,      # name -> filesystem path (read via read_artifact)
        "warnings": list[str],
        "error": str | None,
    }

Large outputs never go into ``summary``; they stay on disk and are surfaced as
artifact paths, following the output-access protocol motivated by
PHREEQC-MCQ-200 (output navigation is the dominant agent failure mode).

The MCP wrapper in :mod:`inverse_gems.mcp_server` exposes these functions to
agent hosts; they are equally usable directly (function calling, tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

TOOL_CONTRACT = "inverse-gems-tool/1.0"

ARTIFACT_ROOTS_ENV = "INVERSE_GEMS_ARTIFACT_ROOTS"


def tool_result(
    tool: str,
    *,
    ok: bool,
    summary: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "contract": TOOL_CONTRACT,
        "tool": tool,
        "ok": ok,
        "summary": summary or {},
        "artifacts": artifacts or {},
        "warnings": warnings or [],
        "error": error,
    }


def _artifact_roots() -> list[Path]:
    roots = [Path.cwd().resolve()]
    raw = os.environ.get(ARTIFACT_ROOTS_ENV, "")
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(Path(part).resolve())
    return roots


def _resolve_artifact_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    for root in _artifact_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"Path {resolved} is outside the allowed artifact roots. "
        f"Allowed roots are the current working directory and {ARTIFACT_ROOTS_ENV} entries."
    )


def _optional_kernel_kwargs(func: Any, **candidates: Any) -> dict[str, Any]:
    """Return the non-None ``candidates`` as keyword arguments for ``func``.

    Newer kernel options (``reaction_model_config``, ``reaction_model_id``,
    ``materials_config``) are forwarded only when the installed InverseGems
    signature accepts them. Requesting an option the kernel does not support
    raises a clear TypeError instead of silently ignoring it.
    """
    import inspect

    selected = {name: value for name, value in candidates.items() if value is not None}
    if not selected:
        return {}
    parameters = inspect.signature(func).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    unsupported = sorted(name for name in selected if name not in parameters and not accepts_any)
    if unsupported:
        raise TypeError(
            f"{func.__module__}.{func.__name__} does not accept {unsupported}; "
            "upgrade the InverseGems kernel to use these options."
        )
    return selected


def _resolve_config_path(path: str | None) -> str | None:
    """Validate an optional config file path against the allowed artifact roots."""
    if path is None:
        return None
    return str(_resolve_artifact_path(path))


def _load_query_payload(query: Any) -> dict[str, Any]:
    """Accept a dict, a YAML string, or a path to a YAML file."""
    if isinstance(query, dict):
        return query
    text = str(query)
    candidate = Path(text)
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        text = candidate.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Query payload must parse to a mapping.")
    return data


def _compact_summary(summary: dict[str, Any], *, max_chars: int = 2000) -> dict[str, Any]:
    """Reduce a tool summary to session-log size: scalars and short values only."""
    compact: dict[str, Any] = {}
    for key, value in summary.items():
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            continue
        if len(encoded) <= 300:
            compact[key] = value
    encoded = json.dumps(compact, ensure_ascii=False, default=str)
    if len(encoded) > max_chars:
        compact = {key: compact[key] for key in list(compact)[:10]}
    return compact


def _log_session(session: str | None, result: dict[str, Any]) -> dict[str, Any]:
    if not session:
        return result
    from .agent_session import append_session_event

    try:
        append_session_event(
            session,
            tool=result["tool"],
            status="ok" if result["ok"] else "error",
            summary=_compact_summary(result.get("summary") or {}),
            artifacts=result.get("artifacts") or {},
        )
    except Exception as exc:  # noqa: BLE001 - session logging must never break the run
        result["warnings"] = [*result.get("warnings", []), f"session log failed: {exc}"]
    return result


def _request_result_to_tool(tool: str, result: Any) -> dict[str, Any]:
    data = result.to_dict()
    summary = {
        "status": data["status"],
        "task_type": data["task_type"],
        "run_dir": data["run_dir"],
        "answer_available": bool(data.get("answer_text")),
        "missing_outputs": data.get("missing_outputs") or {},
        "result_summary": data.get("summary") or {},
    }
    return tool_result(
        tool,
        ok=data["status"] == "complete",
        summary=summary,
        artifacts=dict(data.get("files") or {}),
        warnings=[str(w) for w in data.get("warnings") or []],
        error=data.get("error"),
    )


# ---------------------------------------------------------------------------
# Validation tools (deterministic, no LLM, no DB)
# ---------------------------------------------------------------------------


def validate_task_query(query: Any) -> dict[str, Any]:
    """Validate a task_query payload against the schema without executing it."""
    from inverse_gems.task_query import validate_task_query_data

    try:
        payload = _load_query_payload(query)
        validate_task_query_data(payload)
    except Exception as exc:  # noqa: BLE001 - validation errors go to the caller
        return tool_result("validate_task_query", ok=False, error=str(exc))
    return tool_result(
        "validate_task_query",
        ok=True,
        summary={
            "name": payload.get("name"),
            "task_type": payload.get("task_type"),
        },
    )


def validate_forward_query(query: Any) -> dict[str, Any]:
    """Validate a forward_query payload against the schema without executing it."""
    from inverse_gems.forward_query import validate_forward_query_data

    try:
        payload = _load_query_payload(query)
        validate_forward_query_data(payload)
    except Exception as exc:  # noqa: BLE001
        return tool_result("validate_forward_query", ok=False, error=str(exc))
    return tool_result(
        "validate_forward_query",
        ok=True,
        summary={
            "name": payload.get("name"),
            "task": payload.get("task"),
        },
    )


# ---------------------------------------------------------------------------
# Parse / preview (single LLM touchpoint)
# ---------------------------------------------------------------------------


def parse_task_query(
    request: str,
    out: str,
    *,
    model: str | None = None,
    model_registry: str | None = None,
) -> dict[str, Any]:
    """Parse a natural-language request into a task_query preview (LLM entrance).

    Requires OpenAI credentials. Execution stays deterministic: the returned
    preview must be confirmed via run_confirmed_query before anything runs.
    """
    from inverse_gems.api import parse_request_preview

    try:
        result = parse_request_preview(
            request=request,
            out=out,
            model=model,
            model_registry=model_registry,
        )
    except Exception as exc:  # noqa: BLE001 - surface as contract error
        return tool_result("parse_task_query", ok=False, error=str(exc))
    return _request_result_to_tool("parse_task_query", result)


# ---------------------------------------------------------------------------
# Deterministic execution tools
# ---------------------------------------------------------------------------


def run_forward(
    forward_query: Any,
    out: str,
    db: str,
    *,
    use_mock: bool = False,
    dat_lst: str | None = None,
    disable_plots: bool = True,
    retry_water_on_failure: bool = False,
    retry_water_policy: str = "diagnosis",
    max_xgems_calls: int | None = None,
    reaction_model_config: str | None = None,
    reaction_model_id: str | None = None,
    materials_config: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Run a forward query (single age or time series).

    ``retry_water_policy`` selects the solver recovery strategy when
    ``retry_water_on_failure`` is set: "diagnosis" (adaptive, default for
    agent use) or "ladder" (legacy fixed schedule). ``max_xgems_calls``
    caps non-cached solver invocations for this request; ``session`` logs
    the outcome to a session memory file for multi-turn refinement.
    ``reaction_model_config`` / ``reaction_model_id`` select the reaction
    parameter set and ``materials_config`` a materials YAML override; the
    config paths must lie inside the allowed artifact roots.
    """
    from inverse_gems.api import run_forward_request

    try:
        query_path = _materialize_query(forward_query, Path(out), "forward_query.yaml")
        result = run_forward_request(
            forward_query=query_path,
            out=out,
            db=db,
            dat_lst=dat_lst,
            use_mock=use_mock,
            disable_plots=disable_plots,
            retry_water_on_failure=retry_water_on_failure,
            retry_water_policy=retry_water_policy,
            max_xgems_calls=max_xgems_calls,
            reaction_model_id=reaction_model_id,
            reaction_model_config=_resolve_config_path(reaction_model_config),
            **_optional_kernel_kwargs(
                run_forward_request, materials_config=_resolve_config_path(materials_config)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _log_session(session, tool_result("run_forward", ok=False, error=str(exc)))
    return _log_session(session, _request_result_to_tool("run_forward", result))


def run_task(
    task_query: Any,
    out: str,
    db: str,
    *,
    use_mock: bool = False,
    skip_validation: bool = False,
    dat_lst: str | None = None,
    model_registry: str | None = None,
    reaction_model_config: str | None = None,
    reaction_model_id: str | None = None,
    materials_config: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Run a structured task_query (forward or inverse design)."""
    from inverse_gems.api import run_request

    try:
        query_path = _materialize_query(task_query, Path(out), "task_query.yaml")
        result = run_request(
            task_query=query_path,
            out=out,
            db=db,
            dat_lst=dat_lst,
            use_mock=use_mock,
            skip_validation=skip_validation,
            model_registry=model_registry,
            disable_plots=True,
            **_optional_kernel_kwargs(
                run_request,
                reaction_model_id=reaction_model_id,
                reaction_model_config=_resolve_config_path(reaction_model_config),
                materials_config=_resolve_config_path(materials_config),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _log_session(session, tool_result("run_task", ok=False, error=str(exc)))
    return _log_session(session, _request_result_to_tool("run_task", result))


def run_confirmed_query(
    confirmed_preview: str,
    out: str,
    db: str,
    *,
    confirm_preview: bool = True,
    use_mock: bool = False,
    skip_validation: bool = False,
    dat_lst: str | None = None,
    reaction_model_config: str | None = None,
    reaction_model_id: str | None = None,
    materials_config: str | None = None,
) -> dict[str, Any]:
    """Execute a previously previewed task_query (human-confirmed)."""
    from inverse_gems.api import run_confirmed_request

    try:
        result = run_confirmed_request(
            confirmed_preview=confirmed_preview,
            out=out,
            db=db,
            confirm_preview=confirm_preview,
            use_mock=use_mock,
            skip_validation=skip_validation,
            dat_lst=dat_lst,
            disable_plots=True,
            **_optional_kernel_kwargs(
                run_confirmed_request,
                reaction_model_id=reaction_model_id,
                reaction_model_config=_resolve_config_path(reaction_model_config),
                materials_config=_resolve_config_path(materials_config),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return tool_result("run_confirmed_query", ok=False, error=str(exc))
    return _request_result_to_tool("run_confirmed_query", result)


def _materialize_query(query: Any, out_dir: Path, filename: str) -> Path:
    """Persist an inline query payload so every run has an on-disk query artifact."""
    if not isinstance(query, dict):
        path = Path(str(query))
        if path.is_file():
            return path
        # YAML text passed inline
        query = yaml.safe_load(str(query))
        if not isinstance(query, dict):
            raise ValueError("Query must be a mapping, YAML text, or a path to a YAML file.")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(yaml.safe_dump(query, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Scenario (a): infeasible-design diagnosis and recovery
# ---------------------------------------------------------------------------


def diagnose_design_feasibility(
    design_query: Any,
    *,
    model_registry: str | None = None,
    target_policy: str = "recommended",
) -> dict[str, Any]:
    """Diagnose routing feasibility of a design query without running it."""
    from .design_recovery import diagnose_design_query

    try:
        payload = _load_query_payload(design_query)
        payload = payload.get("design_query", payload)
        diagnosis = diagnose_design_query(
            payload, model_registry=model_registry, target_policy=target_policy
        )
    except Exception as exc:  # noqa: BLE001
        return tool_result("diagnose_design_feasibility", ok=False, error=str(exc))
    return tool_result(
        "diagnose_design_feasibility",
        ok=True,
        summary=diagnosis,
        warnings=[str(w) for w in diagnosis.get("warnings") or []],
    )


def propose_constraint_relaxation(
    design_query: Any,
    *,
    model_registry: str | None = None,
    target_policy: str = "recommended",
    max_proposals: int = 5,
) -> dict[str, Any]:
    """Deterministically rank concrete relaxations for an infeasible design query."""
    from .design_recovery import diagnose_design_query, propose_relaxations

    try:
        payload = _load_query_payload(design_query)
        payload = payload.get("design_query", payload)
        diagnosis = diagnose_design_query(
            payload, model_registry=model_registry, target_policy=target_policy
        )
        proposals = propose_relaxations(payload, diagnosis, max_proposals=max_proposals)
    except Exception as exc:  # noqa: BLE001
        return tool_result("propose_constraint_relaxation", ok=False, error=str(exc))
    return tool_result(
        "propose_constraint_relaxation",
        ok=True,
        summary={
            "feasible": diagnosis["feasible"],
            "proposal_count": len(proposals),
            "proposals": proposals,
        },
    )


def run_design_with_recovery(
    design_query: Any,
    out: str,
    db: str,
    *,
    use_mock: bool = True,
    skip_validation: bool = True,
    max_attempts: int = 3,
    model_registry: str | None = None,
    target_policy: str = "recommended",
    dat_lst: str | None = None,
    reaction_model_config: str | None = None,
    reaction_model_id: str | None = None,
    materials_config: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Observe-replan loop: diagnose, apply top relaxation, re-run (bounded)."""
    from .design_recovery import run_design_with_recovery as _run

    try:
        payload = _load_query_payload(design_query)
        payload = payload.get("design_query", payload)
        outcome = _run(
            payload,
            out=out,
            db=db,
            use_mock=use_mock,
            skip_validation=skip_validation,
            max_attempts=max_attempts,
            model_registry=model_registry,
            target_policy=target_policy,
            dat_lst=dat_lst,
            reaction_model_id=reaction_model_id,
            reaction_model_config=_resolve_config_path(reaction_model_config),
            materials_config=_resolve_config_path(materials_config),
        )
    except Exception as exc:  # noqa: BLE001
        return _log_session(session, tool_result("run_design_with_recovery", ok=False, error=str(exc)))
    artifacts = {"recovery_log": outcome["recovery_log"]}
    final = outcome.get("final") or {}
    for name, path in (final.get("files") or {}).items():
        artifacts[name] = str(path)
    return _log_session(
        session,
        tool_result(
            "run_design_with_recovery",
            ok=outcome["status"] == "complete",
            summary={
                "status": outcome["status"],
                "reason": outcome.get("reason"),
                "attempt_count": len(outcome["attempts"]),
                "changes_applied": outcome["changes_applied"],
                "final_query": outcome["final_query"],
                "final_options": outcome["final_options"],
                "selected_id": final.get("selected_id"),
            },
            artifacts=artifacts,
            error=None if outcome["status"] == "complete" else str(outcome.get("reason")),
        ),
    )


# ---------------------------------------------------------------------------
# Phase 3: coverage-growth campaign
# ---------------------------------------------------------------------------


def run_coverage_campaign(
    target: str,
    db: str,
    out: str,
    *,
    cycles: int = 2,
    candidates_per_cycle: int = 10,
    max_total_candidates: int | None = None,
    stop_r2: float | None = None,
    recipes_csv: str | None = None,
    candidate_source: str = "pool",
    generate_n: int | None = None,
    use_mock: bool = True,
    dat_lst: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Grow global-DB coverage of a target over bounded acquisition cycles.

    Results (mock included) are written into ``db`` - point demos at a
    scratch copy, never the curated DB. ``candidate_source="pool"`` selects
    from ``recipes_csv``; ``"region_generate"`` generates fresh candidates
    inside the target region each cycle.
    """
    from .coverage_campaign import run_coverage_campaign as _run

    try:
        report = _run(
            target=target,
            db=db,
            out=out,
            cycles=cycles,
            candidates_per_cycle=candidates_per_cycle,
            max_total_candidates=max_total_candidates,
            stop_r2=stop_r2,
            recipes_csv=recipes_csv,
            candidate_source=candidate_source,
            generate_n=generate_n,
            use_mock=use_mock,
            dat_lst=dat_lst,
        )
    except Exception as exc:  # noqa: BLE001
        return _log_session(session, tool_result("run_coverage_campaign", ok=False, error=str(exc)))
    ok = report["cycles_run"] > 0 and report["stop_reason"] != "acquisition_empty"
    summary = {
        "target_column": report["target_column"],
        "cycles_run": report["cycles_run"],
        "candidates_used": report["candidates_used"],
        "stop_reason": report["stop_reason"],
        "baseline": report["baseline"],
        "final": report["final"],
        "improvement": report["improvement"],
        "trajectory": report["trajectory"],
    }
    return _log_session(
        session,
        tool_result(
            "run_coverage_campaign",
            ok=ok,
            summary=summary,
            artifacts={"campaign_report": report["campaign_report"], "db": str(db)},
            warnings=[str(w) for w in report.get("warnings") or []][:10],
        ),
    )


def calibrate_scm_kinetics(
    data_csv: str,
    out: str,
    *,
    model: str = "five_param_logistic",
    config_id: str | None = None,
    scms: list[str] | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Fit a registered kinetics model to user DoR data (CSV: scm, age_d, dor).

    Produces a reaction parameter config usable via reaction_model_config on
    any run; the new parameter set coexists with existing DB entries under
    its own reaction_model_signature.
    """
    from inverse_gems.kinetics_calibration import calibrate_scm_kinetics as _run

    try:
        resolved = _resolve_artifact_path(data_csv)
        report = _run(
            data_csv=resolved, out=out, model=model, config_id=config_id, scms=scms
        )
    except Exception as exc:  # noqa: BLE001
        return _log_session(session, tool_result("calibrate_scm_kinetics", ok=False, error=str(exc)))
    artifacts = {"reaction_model_config": report["config_path"]}
    if report.get("plot"):
        artifacts["calibration_plot"] = report["plot"]
    return _log_session(
        session,
        tool_result(
            "calibrate_scm_kinetics",
            ok=True,
            summary={
                "id": report["id"],
                "model": report["model"],
                "fits": {
                    name: {"r2": fit["r2"], "rmse": fit["rmse"], "n_points": fit["n_points"],
                           "age_range_d": [fit["age_min_d"], fit["age_max_d"]]}
                    for name, fit in report["fits"].items()
                },
                "usage": report["usage"],
            },
            artifacts=artifacts,
            warnings=[str(w) for w in report.get("warnings") or []],
        ),
    )


# ---------------------------------------------------------------------------
# Session memory and candidate refinement
# ---------------------------------------------------------------------------


def recall_session(
    session: str,
    *,
    tool: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Recall recent session events (past runs, their summaries and artifacts)."""
    from .agent_session import read_session_events

    try:
        events = read_session_events(session, tool=tool, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return tool_result("recall_session", ok=False, error=str(exc))
    return tool_result(
        "recall_session",
        ok=True,
        summary={"session": str(session), "event_count": len(events), "events": events},
    )


_FILTER_OPS = {
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}


def filter_candidates(
    candidates_csv: str,
    where: list[dict[str, Any]],
    *,
    limit: int = 20,
    session: str | None = None,
) -> dict[str, Any]:
    """Filter a candidates CSV by column conditions (multi-turn refinement).

    ``where`` is a list of ``{"column", "op", "value"}`` with op in
    <, <=, >, >=, ==, != or "contains". Bare column names also match their
    ``x__``/``meta__``/``y__`` prefixed variants. Writes the filtered rows
    next to the source CSV for chaining.
    """
    try:
        path = _resolve_artifact_path(candidates_csv)
        frame = pd.read_csv(path)
        applied: list[str] = []
        for condition in where:
            column = str(condition.get("column") or "")
            op = str(condition.get("op") or "")
            value = condition.get("value")
            resolved_column = next(
                (
                    name
                    for name in [column, f"x__{column}", f"meta__{column}", f"y__{column}"]
                    if name in frame.columns
                ),
                None,
            )
            if resolved_column is None:
                return tool_result(
                    "filter_candidates",
                    ok=False,
                    error=f"Column {column!r} not found. Available: {sorted(frame.columns)[:40]}",
                )
            series = frame[resolved_column]
            if op == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, na=False)
            elif op in _FILTER_OPS:
                mask = _FILTER_OPS[op](series, value)
            else:
                return tool_result(
                    "filter_candidates",
                    ok=False,
                    error=f"Unsupported op {op!r}; expected one of {sorted(_FILTER_OPS) + ['contains']}.",
                )
            frame = frame[mask]
            applied.append(f"{resolved_column} {op} {value!r}")
        filtered_path = path.with_name(f"{path.stem}_filtered_{len(applied)}cond.csv")
        frame.to_csv(filtered_path, index=False)
        result = tool_result(
            "filter_candidates",
            ok=True,
            summary={
                "source": str(path),
                "conditions": applied,
                "matched_rows": int(len(frame)),
                "rows": frame.head(max(0, int(limit))).to_dict(orient="records"),
            },
            artifacts={"filtered_csv": str(filtered_path)},
        )
    except Exception as exc:  # noqa: BLE001
        result = tool_result("filter_candidates", ok=False, error=str(exc))
    return _log_session(session, result)


# ---------------------------------------------------------------------------
# Memory / lookup tools
# ---------------------------------------------------------------------------


def query_past_runs(
    db: str,
    *,
    material_system: str | None = None,
    binder_contains: str | None = None,
    age_min: float | None = None,
    age_max: float | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Query the local run database (long-term memory) for past recipe runs."""
    from inverse_gems.database import InverseGemsDatabase

    db_dir = Path(db)
    if not (db_dir / "inverse_gems.sqlite").exists() and not any(db_dir.glob("*.sqlite*")):
        if not db_dir.exists():
            return tool_result(
                "query_past_runs",
                ok=False,
                error=f"Run database directory not found: {db_dir}",
            )
    try:
        store = InverseGemsDatabase(db_dir)
        rows = store.recipe_rows()
    except Exception as exc:  # noqa: BLE001
        return tool_result("query_past_runs", ok=False, error=str(exc))

    def _keep(row: dict[str, Any]) -> bool:
        if material_system and str(row.get("material_system") or "") != material_system:
            return False
        if binder_contains and binder_contains.lower() not in str(row.get("recipe_json") or "").lower():
            return False
        age = row.get("age_days")
        if age_min is not None and (age is None or float(age) < age_min):
            return False
        if age_max is not None and (age is None or float(age) > age_max):
            return False
        return True

    matched = [row for row in rows if _keep(row)]
    compact_keys = [
        "recipe_id",
        "material_system",
        "age_days",
        "w_b",
        "chem_hash",
        "created_at",
        "porosity",
        "solver_rescued",
        "primary_solver_status",
    ]
    compact = [
        {key: row.get(key) for key in compact_keys if key in row}
        for row in matched[: max(0, int(limit))]
    ]
    return tool_result(
        "query_past_runs",
        ok=True,
        summary={
            "total_rows": len(rows),
            "matched_rows": len(matched),
            "returned_rows": len(compact),
            "rows": compact,
        },
        artifacts={"db_dir": str(db_dir)},
    )


def query_model_registry(registry: str | None = None) -> dict[str, Any]:
    """List surrogate model registry entries and whether their artifacts exist.

    Degrades gracefully when the chemistry database has not been built yet:
    entries are still listed, with ``artifacts_present`` False.
    """
    registry_path = Path(registry) if registry else Path("configs/design_query_model_registry.yaml")
    if not registry_path.is_file():
        return tool_result(
            "query_model_registry",
            ok=False,
            error=f"Model registry not found: {registry_path}",
        )
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return tool_result("query_model_registry", ok=False, error=str(exc))

    entries = data.get("models") or data.get("registry") or []
    models: list[dict[str, Any]] = []
    missing_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        table = entry.get("model_table")
        bundle = entry.get("model_bundle")
        present = bool(table and Path(str(table)).exists()) and bool(bundle and Path(str(bundle)).exists())
        if not present:
            missing_count += 1
        models.append(
            {
                "id": entry.get("id"),
                "material_system": entry.get("material_system"),
                "age_days": entry.get("age_days"),
                "reaction_model_id": entry.get("reaction_model_id"),
                "artifacts_present": present,
            }
        )
    warnings = []
    if missing_count:
        warnings.append(
            f"{missing_count}/{len(models)} registry entries point to missing model artifacts "
            "(expected while the v2 chemistry DB rebuild is deferred)."
        )
    return tool_result(
        "query_model_registry",
        ok=True,
        summary={"registry": str(registry_path), "model_count": len(models), "models": models},
        artifacts={"registry": str(registry_path)},
        warnings=warnings,
    )


def check_coverage(global_db: str) -> dict[str, Any]:
    """Report what exists in a global chemistry DB directory (row counts, artifacts).

    When the DB has not been built (v2 defers this until the new SCM kinetics),
    the tool reports that explicitly instead of failing opaquely.
    """
    root = Path(global_db)
    if not root.exists():
        return tool_result(
            "check_coverage",
            ok=False,
            summary={"global_db": str(root), "built": False},
            error=(
                f"Global chemistry DB directory not found: {root}. "
                "The v2 database rebuild is deferred until the new SCM reaction kinetics land."
            ),
        )
    artifacts: dict[str, str] = {"global_db": str(root)}
    summary: dict[str, Any] = {"global_db": str(root), "built": True}
    model_table = root / "global_chemistry" / "global_model_table.csv"
    if model_table.exists():
        frame = pd.read_csv(model_table)
        summary["model_table_rows"] = int(len(frame))
        if "meta__material_system" in frame.columns:
            summary["material_systems"] = sorted(
                str(v) for v in frame["meta__material_system"].dropna().unique()
            )
        artifacts["global_model_table"] = str(model_table)
    else:
        summary["model_table_rows"] = 0
        summary["built"] = False
    for name in ["coverage_summary.json", "quality_summary.json", "quality_summary.md"]:
        for candidate in root.rglob(name):
            artifacts[name.replace(".", "_")] = str(candidate)
            break
    return tool_result("check_coverage", ok=True, summary=summary, artifacts=artifacts)


# ---------------------------------------------------------------------------
# Output-access protocol
# ---------------------------------------------------------------------------


def list_run_artifacts(run_dir: str, *, max_entries: int = 200) -> dict[str, Any]:
    """List files under a run directory with sizes, for output navigation."""
    try:
        root = _resolve_artifact_path(run_dir)
    except Exception as exc:  # noqa: BLE001
        return tool_result("list_run_artifacts", ok=False, error=str(exc))
    if not root.exists():
        return tool_result("list_run_artifacts", ok=False, error=f"Directory not found: {root}")
    entries: list[dict[str, Any]] = []
    truncated = False
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if len(entries) >= max_entries:
            truncated = True
            break
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
            }
        )
    return tool_result(
        "list_run_artifacts",
        ok=True,
        summary={"run_dir": str(root), "file_count": len(entries), "truncated": truncated, "files": entries},
        artifacts={"run_dir": str(root)},
    )


def read_artifact(
    path: str,
    *,
    offset: int = 0,
    limit: int = 100,
    columns: list[str] | None = None,
    json_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Read a slice of an artifact file (CSV rows, JSON keys, or text lines).

    - CSV: returns ``limit`` rows starting at ``offset``; optional column filter.
    - JSON: returns the parsed object, optionally reduced to ``json_keys``.
    - other: returns text lines ``offset``..``offset+limit``.
    """
    try:
        resolved = _resolve_artifact_path(path)
    except Exception as exc:  # noqa: BLE001
        return tool_result("read_artifact", ok=False, error=str(exc))
    if not resolved.is_file():
        return tool_result("read_artifact", ok=False, error=f"File not found: {resolved}")

    suffix = resolved.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(resolved)
            total = len(frame)
            if columns:
                keep = [c for c in columns if c in frame.columns]
                frame = frame[keep]
            window = frame.iloc[offset : offset + limit]
            return tool_result(
                "read_artifact",
                ok=True,
                summary={
                    "path": str(resolved),
                    "kind": "csv",
                    "total_rows": total,
                    "offset": offset,
                    "returned_rows": len(window),
                    "columns": list(window.columns),
                    "rows": window.to_dict(orient="records"),
                },
            )
        if suffix == ".json":
            import json as _json

            data = _json.loads(resolved.read_text(encoding="utf-8"))
            selected = data
            if json_keys and isinstance(data, dict):
                selected = {key: data.get(key) for key in json_keys}
            return tool_result(
                "read_artifact",
                ok=True,
                summary={
                    "path": str(resolved),
                    "kind": "json",
                    "top_level_keys": sorted(data.keys()) if isinstance(data, dict) else None,
                    "data": selected,
                },
            )
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        window = lines[offset : offset + limit]
        return tool_result(
            "read_artifact",
            ok=True,
            summary={
                "path": str(resolved),
                "kind": "text",
                "total_lines": len(lines),
                "offset": offset,
                "returned_lines": len(window),
                "text": "\n".join(window),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return tool_result("read_artifact", ok=False, error=str(exc))
