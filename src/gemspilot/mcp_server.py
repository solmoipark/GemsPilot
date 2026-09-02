"""MCP server exposing the GemsPilot agent tool layer over the InverseGems kernel.

Run with:

    gemspilot-mcp

or:

    py -m gemspilot.mcp_server

Requires the ``mcp`` dependency (installed with ``pip install gemspilot``).
The server is a thin wrapper: every tool delegates to
:mod:`gemspilot.agent_tools`, which returns the standardized ToolResult
contract. The LLM host never computes science; it only orchestrates these
deterministic tools.

Guardrail defaults (roadmap §3-(5)): tools that can trigger real xGEMS batches
default to mock/cached behavior; hosts must pass ``use_mock=False`` explicitly
and are expected to gate that behind human approval.
"""

from __future__ import annotations

from typing import Any

from . import agent_tools

try:
    from mcp.server.mcpserver import MCPServer as _ServerApp  # mcp >= 2.0
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as _ServerApp  # mcp 1.x
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "The 'mcp' package is required for the MCP server. "
            "Install it with: pip install inverse-gems[agent]"
        ) from exc

app = _ServerApp(
    "gemspilot",
    instructions=(
        "Deterministic thermodynamic modeling tools for blended cementitious binders. "
        "Typical flows: (1) validate_task_query -> run_task; "
        "(2) parse_task_query (LLM preview) -> human confirmation -> run_confirmed_query; "
        "(3) run_forward for direct forward calculations. "
        "Results return a small summary plus artifact paths; use list_run_artifacts "
        "and read_artifact to navigate outputs instead of guessing. "
        "Real xGEMS execution requires use_mock=False and should be human-approved."
    ),
)


@app.tool()
def validate_task_query(query: str) -> dict[str, Any]:
    """Validate a task_query (YAML text or file path) against the schema without running it."""
    return agent_tools.validate_task_query(query)


@app.tool()
def validate_forward_query(query: str) -> dict[str, Any]:
    """Validate a forward_query (YAML text or file path) against the schema without running it."""
    return agent_tools.validate_forward_query(query)


@app.tool()
def parse_task_query(request: str, out: str, model: str | None = None) -> dict[str, Any]:
    """Parse a natural-language request into a task_query preview (uses OpenAI; no execution)."""
    return agent_tools.parse_task_query(request, out, model=model)


@app.tool()
def run_forward(
    forward_query: str,
    out: str,
    db: str,
    use_mock: bool = True,
    dat_lst: str | None = None,
    retry_water_on_failure: bool = False,
    retry_water_policy: str = "diagnosis",
    max_xgems_calls: int | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Run a forward query (YAML text or file path). use_mock=False triggers real xGEMS.

    On solver failure with retry_water_on_failure=True, the water recovery
    policy ("diagnosis" adaptive or "ladder" fixed) retries with adjusted
    xGEMS water; every attempt and its diagnosis is recorded.
    max_xgems_calls caps non-cached solver invocations for this request;
    session (a directory or .jsonl path) logs the outcome for later recall.
    """
    return agent_tools.run_forward(
        forward_query,
        out,
        db,
        use_mock=use_mock,
        dat_lst=dat_lst,
        retry_water_on_failure=retry_water_on_failure,
        retry_water_policy=retry_water_policy,
        max_xgems_calls=max_xgems_calls,
        session=session,
    )


@app.tool()
def run_task(
    task_query: str,
    out: str,
    db: str,
    use_mock: bool = True,
    skip_validation: bool = False,
    dat_lst: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Run a structured task_query (forward or inverse design). use_mock=False triggers real xGEMS."""
    return agent_tools.run_task(
        task_query,
        out,
        db,
        use_mock=use_mock,
        skip_validation=skip_validation,
        dat_lst=dat_lst,
        session=session,
    )


@app.tool()
def run_confirmed_query(
    confirmed_preview: str,
    out: str,
    db: str,
    use_mock: bool = True,
    skip_validation: bool = False,
    dat_lst: str | None = None,
) -> dict[str, Any]:
    """Execute a human-confirmed task_query preview directory."""
    return agent_tools.run_confirmed_query(
        confirmed_preview,
        out,
        db,
        use_mock=use_mock,
        skip_validation=skip_validation,
        dat_lst=dat_lst,
    )


@app.tool()
def diagnose_design_feasibility(
    design_query: str,
    model_registry: str | None = None,
    target_policy: str = "recommended",
) -> dict[str, Any]:
    """Diagnose why a design query is (in)feasible for routing, without running it."""
    return agent_tools.diagnose_design_feasibility(
        design_query, model_registry=model_registry, target_policy=target_policy
    )


@app.tool()
def propose_constraint_relaxation(
    design_query: str,
    model_registry: str | None = None,
    target_policy: str = "recommended",
    max_proposals: int = 5,
) -> dict[str, Any]:
    """Rank deterministic relaxations (materials, targets, age, policy) for an infeasible design query."""
    return agent_tools.propose_constraint_relaxation(
        design_query,
        model_registry=model_registry,
        target_policy=target_policy,
        max_proposals=max_proposals,
    )


@app.tool()
def run_design_with_recovery(
    design_query: str,
    out: str,
    db: str,
    use_mock: bool = True,
    skip_validation: bool = True,
    max_attempts: int = 3,
    model_registry: str | None = None,
    target_policy: str = "recommended",
    session: str | None = None,
) -> dict[str, Any]:
    """Observe-replan loop: diagnose -> apply top relaxation -> re-run (bounded, fully logged)."""
    return agent_tools.run_design_with_recovery(
        design_query,
        out,
        db,
        use_mock=use_mock,
        skip_validation=skip_validation,
        max_attempts=max_attempts,
        model_registry=model_registry,
        target_policy=target_policy,
        session=session,
    )


@app.tool()
def run_coverage_campaign(
    target: str,
    db: str,
    out: str,
    cycles: int = 2,
    candidates_per_cycle: int = 10,
    max_total_candidates: int | None = None,
    stop_r2: float | None = None,
    recipes_csv: str | None = None,
    candidate_source: str = "pool",
    generate_n: int | None = None,
    use_mock: bool = True,
    session: str | None = None,
) -> dict[str, Any]:
    """Grow global-DB coverage of a target: region analysis -> acquisition -> batch -> retrain, per cycle.

    WARNING: results (mock included) are written into db - use a scratch copy
    of the global DB, never the curated one. use_mock=False needs real xGEMS
    and human approval. candidate_source "pool" selects from recipes_csv;
    "region_generate" generates fresh candidates inside the target region.
    """
    return agent_tools.run_coverage_campaign(
        target,
        db,
        out,
        cycles=cycles,
        candidates_per_cycle=candidates_per_cycle,
        max_total_candidates=max_total_candidates,
        stop_r2=stop_r2,
        recipes_csv=recipes_csv,
        candidate_source=candidate_source,
        generate_n=generate_n,
        use_mock=use_mock,
        session=session,
    )


@app.tool()
def calibrate_scm_kinetics(
    data_csv: str,
    out: str,
    model: str = "five_param_logistic",
    config_id: str | None = None,
    scms: list[str] | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Fit a registered SCM kinetics model to user DoR data (CSV: scm, age_d, dor columns).

    Writes a reaction parameter config with calibration provenance; pass it as
    reaction_model_config to any run to use the calibrated parameters (they
    coexist with existing DB entries under a new reaction_model_signature).
    """
    return agent_tools.calibrate_scm_kinetics(
        data_csv, out, model=model, config_id=config_id, scms=scms, session=session
    )


@app.tool()
def recall_session(session: str, tool: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Recall recent session events (past runs, summaries, artifact paths) for multi-turn refinement."""
    return agent_tools.recall_session(session, tool=tool, limit=limit)


@app.tool()
def filter_candidates(
    candidates_csv: str,
    where: list[dict[str, Any]],
    limit: int = 20,
    session: str | None = None,
) -> dict[str, Any]:
    """Filter a candidates CSV by conditions [{column, op, value}]; op: < <= > >= == != contains."""
    return agent_tools.filter_candidates(candidates_csv, where, limit=limit, session=session)


@app.tool()
def query_past_runs(
    db: str,
    material_system: str | None = None,
    binder_contains: str | None = None,
    age_min: float | None = None,
    age_max: float | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Query the local run database (long-term memory) for past recipe runs."""
    return agent_tools.query_past_runs(
        db,
        material_system=material_system,
        binder_contains=binder_contains,
        age_min=age_min,
        age_max=age_max,
        limit=limit,
    )


@app.tool()
def query_model_registry(registry: str | None = None) -> dict[str, Any]:
    """List surrogate model registry entries and whether their artifacts exist locally."""
    return agent_tools.query_model_registry(registry)


@app.tool()
def check_coverage(global_db: str) -> dict[str, Any]:
    """Report what exists in a global chemistry DB directory (row counts, coverage artifacts)."""
    return agent_tools.check_coverage(global_db)


@app.tool()
def list_run_artifacts(run_dir: str, max_entries: int = 200) -> dict[str, Any]:
    """List files under a run directory with sizes, for output navigation."""
    return agent_tools.list_run_artifacts(run_dir, max_entries=max_entries)


@app.tool()
def read_artifact(
    path: str,
    offset: int = 0,
    limit: int = 100,
    columns: list[str] | None = None,
    json_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Read a slice of an artifact (CSV rows, JSON keys, or text lines)."""
    return agent_tools.read_artifact(
        path, offset=offset, limit=limit, columns=columns, json_keys=json_keys
    )


def _registered_tool_names(server: Any) -> set[str]:
    manager = getattr(server, "_tool_manager", None)
    tools = getattr(manager, "_tools", None)
    if isinstance(tools, dict):
        return set(tools)
    try:
        return {tool.name for tool in manager.list_tools()}
    except Exception:  # noqa: BLE001 - unknown server API; fall back to no guard
        return set()


def register_extra_toolsets(server: Any) -> list[str]:
    """Register tools published under the ``gemspilot.toolsets`` entry-point group.

    Every entry point resolves to ToolSpec-like objects (``.name``, ``.func``,
    ``.policy``); each function is registered as ``server.tool(name=...)``.
    Broken entry points and names that are already registered are skipped, so
    a third-party toolset can never break or shadow the built-in tools. The
    names actually registered are returned (and kept in ``EXTRA_TOOLS``).
    """
    from .runner import discover_toolsets

    registered: list[str] = []
    existing = _registered_tool_names(server)
    for spec in discover_toolsets():
        if spec.name in existing:
            continue
        try:
            server.tool(name=spec.name)(spec.func)
        except Exception:  # noqa: BLE001 - a bad third-party tool must not break the server
            continue
        existing.add(spec.name)
        registered.append(spec.name)
    return registered


EXTRA_TOOLS = register_extra_toolsets(app)


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
