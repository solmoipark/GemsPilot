import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from gemspilot import agent_tools


FORWARD_QUERY = {
    "name": "agent_tools_mock_forward",
    "task": "forward_calculation",
    "recipe": {"binders": {"OPC": 60, "slag": 40}, "w_b": 0.45},
    "age_grid": {"values": [28.0]},
    "outputs": {
        "phase_masses": "all",
        "phase_volumes": "all",
        "phase_volumes_reconstructed": "all",
        "aqueous_species": "all",
        "scalars": "all",
    },
    "plots": [],
    "response_summary": {
        "phases": ["Mock C-S-H raw phase", "Mock Portlandite"],
        "scalars": ["pH", "porosity"],
    },
}


def _assert_contract(result: dict) -> None:
    assert result["contract"] == agent_tools.TOOL_CONTRACT
    assert set(result) == {"contract", "tool", "ok", "summary", "artifacts", "warnings", "error"}
    json.dumps(result)  # must stay JSON-serializable


def test_validate_forward_query_accepts_dict_yaml_and_path(tmp_path):
    ok = agent_tools.validate_forward_query(FORWARD_QUERY)
    _assert_contract(ok)
    assert ok["ok"] is True

    yaml_text = yaml.safe_dump(FORWARD_QUERY, sort_keys=False)
    assert agent_tools.validate_forward_query(yaml_text)["ok"] is True

    path = tmp_path / "q.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    assert agent_tools.validate_forward_query(str(path))["ok"] is True


def test_validate_task_query_reports_schema_errors():
    bad = agent_tools.validate_task_query({"task_type": "unknown_kind"})
    _assert_contract(bad)
    assert bad["ok"] is False
    assert bad["error"]


def test_run_forward_mock_returns_summary_and_artifacts(tmp_path):
    result = agent_tools.run_forward(
        FORWARD_QUERY, str(tmp_path / "run"), str(tmp_path / "db"), use_mock=True
    )
    _assert_contract(result)
    assert result["ok"] is True
    assert result["summary"]["status"] == "complete"
    assert result["summary"]["result_summary"]["completed_count"] >= 1
    assert Path(result["artifacts"]["forward_dir"]).exists()
    # inline dict queries must be materialized as an on-disk artifact
    assert (tmp_path / "run" / "forward_query.yaml").exists()


def test_query_past_runs_filters_after_mock_run(tmp_path):
    agent_tools.run_forward(FORWARD_QUERY, str(tmp_path / "run"), str(tmp_path / "db"), use_mock=True)
    result = agent_tools.query_past_runs(str(tmp_path / "db"))
    _assert_contract(result)
    assert result["ok"] is True
    assert result["summary"]["total_rows"] >= 1
    row = result["summary"]["rows"][0]
    assert "recipe_id" in row and "chem_hash" in row

    none = agent_tools.query_past_runs(str(tmp_path / "db"), material_system="no_such_system")
    assert none["summary"]["matched_rows"] == 0

    missing = agent_tools.query_past_runs(str(tmp_path / "nonexistent_db"))
    assert missing["ok"] is False


KERNEL_ROOT = Path(os.environ.get("INVERSE_GEMS_ROOT", "C:\\Users\\solmo\\InverseGems v2"))


def test_query_model_registry_flags_missing_artifacts():
    result = agent_tools.query_model_registry(str(KERNEL_ROOT / "configs" / "design_query_model_registry.yaml"))
    _assert_contract(result)
    assert result["ok"] is True
    assert result["summary"]["model_count"] > 0
    # v2 has no data/reports yet, so artifacts must be flagged missing, not fatal
    assert any(not m["artifacts_present"] for m in result["summary"]["models"])
    assert result["warnings"]


def test_check_coverage_reports_unbuilt_db(tmp_path):
    result = agent_tools.check_coverage(str(tmp_path / "no_global_db"))
    _assert_contract(result)
    assert result["ok"] is False
    assert result["summary"]["built"] is False
    assert "deferred" in result["error"]


def test_list_and_read_artifacts_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = agent_tools.run_forward(FORWARD_QUERY, "run", "db", use_mock=True)
    assert run["ok"] is True

    listing = agent_tools.list_run_artifacts("run")
    _assert_contract(listing)
    assert listing["ok"] is True
    names = [f["path"] for f in listing["summary"]["files"]]
    assert any(name.endswith("time_series.csv") for name in names)

    csv_path = next(p for k, p in run["artifacts"].items() if p.endswith("time_series.csv"))
    csv_read = agent_tools.read_artifact(csv_path, limit=5)
    assert csv_read["ok"] is True
    assert csv_read["summary"]["kind"] == "csv"
    assert csv_read["summary"]["returned_rows"] >= 1

    json_path = run["artifacts"]["forward_query_summary_json"]
    json_read = agent_tools.read_artifact(json_path, json_keys=["completed_count"])
    assert json_read["ok"] is True
    assert json_read["summary"]["data"]["completed_count"] >= 1


def test_read_artifact_rejects_paths_outside_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = Path(__file__).resolve()
    result = agent_tools.read_artifact(str(outside))
    _assert_contract(result)
    assert result["ok"] is False
    assert "outside the allowed artifact roots" in result["error"]


def test_mcp_server_registers_all_tools():
    pytest.importorskip("mcp")
    from gemspilot import mcp_server

    tools = asyncio.run(mcp_server.app.list_tools())
    names = {tool.name for tool in tools}
    builtin = {
        "validate_task_query",
        "validate_forward_query",
        "parse_task_query",
        "run_forward",
        "run_task",
        "run_confirmed_query",
        "diagnose_design_feasibility",
        "propose_constraint_relaxation",
        "run_design_with_recovery",
        "run_coverage_campaign",
        "calibrate_scm_kinetics",
        "query_past_runs",
        "query_model_registry",
        "check_coverage",
        "list_run_artifacts",
        "read_artifact",
        "recall_session",
        "filter_candidates",
    }
    assert builtin <= names
    # Anything beyond the built-ins must come from gemspilot.toolsets entry points.
    assert names - builtin == set(mcp_server.EXTRA_TOOLS)
    assert builtin.isdisjoint(mcp_server.EXTRA_TOOLS)


# --- calibration tool wrapper (moved from kernel test_kinetics_calibration) ---
import numpy as np
from inverse_gems.scm_reaction import SCMLogisticParameters, scm_alpha

_CAL_TRUE = {"slag": {"A": 0.0, "B": 0.9, "C": 15.0, "D": 0.72, "G": 1.0},
             "fly_ash": {"A": 0.0, "B": 0.6, "C": 45.0, "D": 0.48, "G": 1.0}}


def _synthetic_data(tmp_path, noise=0.015, seed=3):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for scm, params in _CAL_TRUE.items():
        for age in [1, 3, 7, 14, 28, 56, 90, 180, 365]:
            for _ in range(3):
                alpha = scm_alpha(float(age), SCMLogisticParameters(**params))
                rows.append({"scm": scm, "age_d": age,
                             "dor": float(np.clip(alpha + rng.normal(0, noise), 0, 1))})
    path = tmp_path / "user_dor.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_agent_tool_wrapper_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _synthetic_data(tmp_path)
    result = agent_tools.calibrate_scm_kinetics(str(data), str(tmp_path / "cal"), config_id="agent_cal")
    assert result["contract"] == agent_tools.TOOL_CONTRACT
    assert result["ok"] is True
    assert result["summary"]["id"] == "agent_cal"
    assert "slag" in result["summary"]["fits"]
    assert result["artifacts"]["reaction_model_config"].endswith("reaction_parameters.agent_cal.yaml")


# --- reaction_model_config / materials_config threading (mock-only) ---
PINNED_SLAG_CONFIG = {
    "id": "t",
    "scm_reaction": {"slag": {"A": 0.3, "B": 1, "C": 1, "D": 0.3, "G": 1}},
    "availability_modifier": {"enabled": False},
}


def test_run_forward_honours_reaction_model_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "pinned_slag.yaml"
    config.write_text(yaml.safe_dump(PINNED_SLAG_CONFIG), encoding="utf-8")

    result = agent_tools.run_forward(
        FORWARD_QUERY, "run", "db", use_mock=True, reaction_model_config=str(config)
    )
    _assert_contract(result)
    assert result["ok"] is True, result["error"]

    degrees = sorted((tmp_path / "db" / "recipe_runs").glob("*/reaction_degrees.json"))
    assert degrees
    for path in degrees:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["scm"]["slag"] == pytest.approx(0.3)


def test_run_forward_rejects_config_outside_artifact_roots(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.delenv(agent_tools.ARTIFACT_ROOTS_ENV, raising=False)
    outside = tmp_path / "outside.yaml"
    outside.write_text(yaml.safe_dump(PINNED_SLAG_CONFIG), encoding="utf-8")

    result = agent_tools.run_forward(
        FORWARD_QUERY, "run", "db", use_mock=True, reaction_model_config=str(outside)
    )
    _assert_contract(result)
    assert result["ok"] is False
    assert "outside the allowed artifact roots" in result["error"]


def test_optional_kernel_kwargs_follows_kernel_signature():
    def kernel(*, reaction_model_config=None, **_ignored):
        return None

    def strict_kernel(*, reaction_model_config=None):
        return None

    assert agent_tools._optional_kernel_kwargs(strict_kernel, reaction_model_config=None) == {}
    assert agent_tools._optional_kernel_kwargs(strict_kernel, reaction_model_config="x") == {
        "reaction_model_config": "x"
    }
    assert agent_tools._optional_kernel_kwargs(kernel, materials_config="m") == {"materials_config": "m"}
    with pytest.raises(TypeError, match="materials_config"):
        agent_tools._optional_kernel_kwargs(strict_kernel, materials_config="m")


def test_mcp_wrappers_expose_reaction_model_kwargs():
    pytest.importorskip("mcp")
    from gemspilot.mcp_server import app

    tools = {tool.name: tool for tool in asyncio.run(app.list_tools())}
    for name in ["run_forward", "run_task", "run_confirmed_query", "run_design_with_recovery"]:
        schema = getattr(tools[name], "input_schema", None) or tools[name].inputSchema  # mcp 2.x / 1.x
        properties = schema["properties"]
        assert {"reaction_model_config", "reaction_model_id", "materials_config"} <= set(properties), name
