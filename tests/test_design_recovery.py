import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from gemspilot import agent_tools
from gemspilot.design_recovery import (
    diagnose_design_query,
    propose_relaxations,
    run_design_with_recovery,
)


class ToyEstimator:
    def predict(self, x):
        porosity = 0.20 + 0.002 * x["x__OPC"].to_numpy()
        return pd.DataFrame({"y__porosity": porosity}).to_numpy()


@pytest.fixture()
def toy_registry(tmp_path):
    """Registry with one OPC+slag model at 28 d, backed by a toy table/bundle."""
    table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["high_opc", "low_opc"],
            "meta__material_system": ["OPC_slag", "OPC_slag"],
            "meta__age_bin": ["standard", "standard"],
            "x__OPC": [70.0, 30.0],
            "x__slag": [30.0, 70.0],
            "x__w_b": [0.40, 0.40],
            "x__age_days": [28.0, 28.0],
        }
    ).to_csv(table, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": ToyEstimator(),
            "inputs": ["x__OPC", "x__slag", "x__w_b", "x__age_days"],
            "targets": ["y__porosity"],
        },
        bundle,
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "default_age_tolerance": 1.0e-9,
                "models": [
                    {
                        "id": "toy_OPC_slag_age28",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "model_table": str(table),
                        "model_bundle": str(bundle),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry


def _query(**overrides):
    base = {
        "name": "recovery_toy",
        "design_space": {
            "allowed_materials": ["OPC"],
            "strict_materials": True,
            "age_days": 28,
        },
        "inputs": {
            "OPC": {"min": 20, "max": 80},
            "slag": {"min": 20, "max": 80},
            "w_b": {"min": 0.30, "max": 0.50},
        },
        "ranking": {"mode": "lexicographic"},
        "preferences": [{"input": "OPC", "direction": "minimize"}],
        "search_top_k": 1,
        "selection_top_k": 1,
    }
    base.update(overrides)
    return base


def test_diagnose_reports_material_blocker(toy_registry):
    diagnosis = diagnose_design_query(_query(), model_registry=toy_registry)
    assert diagnosis["feasible"] is False
    assert diagnosis["candidate_count"] == 1
    assert diagnosis["blocker_histogram"].get("unallowed_materials") == 1
    assert "slag" in str(diagnosis["candidates"][0]["blockers"])


def test_diagnose_feasible_when_materials_allowed(toy_registry):
    query = _query()
    query["design_space"]["allowed_materials"] = ["OPC", "slag"]
    diagnosis = diagnose_design_query(query, model_registry=toy_registry)
    assert diagnosis["feasible"] is True
    assert diagnosis["selected_id"] == "toy_OPC_slag_age28"


def test_propose_allows_missing_materials(toy_registry):
    query = _query()
    diagnosis = diagnose_design_query(query, model_registry=toy_registry)
    proposals = propose_relaxations(query, diagnosis)
    assert proposals
    top = proposals[0]
    assert top["kind"] == "allow_additional_materials"
    assert "slag" in top["query"]["design_space"]["allowed_materials"]
    assert top["unblocks"] == ["toy_OPC_slag_age28"]
    # the original query object is never mutated
    assert query["design_space"]["allowed_materials"] == ["OPC"]


def test_propose_drops_unsupported_targets(toy_registry):
    query = _query(
        preferences=[{"target": "C-A-S-H", "direction": "maximize"}],
    )
    query["design_space"]["allowed_materials"] = ["OPC", "slag"]
    diagnosis = diagnose_design_query(query, model_registry=toy_registry)
    assert diagnosis["feasible"] is False
    proposals = propose_relaxations(query, diagnosis)
    kinds = [p["kind"] for p in proposals]
    assert "drop_unsupported_targets" in kinds
    drop = next(p for p in proposals if p["kind"] == "drop_unsupported_targets")
    assert drop["query"]["preferences"] == []


def test_propose_nearest_age_when_no_candidates(toy_registry):
    query = _query()
    query["design_space"]["age_days"] = 90
    diagnosis = diagnose_design_query(query, model_registry=toy_registry)
    assert diagnosis["candidate_count"] == 0
    proposals = propose_relaxations(query, diagnosis)
    assert proposals[0]["kind"] == "use_nearest_supported_age"
    assert proposals[0]["query"]["design_space"]["age_days"] == 28.0


def test_recovery_loop_relaxes_and_completes(tmp_path, toy_registry):
    outcome = run_design_with_recovery(
        _query(),
        out=tmp_path / "recovery",
        db=tmp_path / "db",
        use_mock=True,
        skip_validation=True,
        model_registry=toy_registry,
    )
    assert outcome["status"] == "complete"
    assert [c["kind"] for c in outcome["changes_applied"]] == ["allow_additional_materials"]
    assert len(outcome["attempts"]) == 2
    assert outcome["attempts"][0]["feasible"] is False
    assert outcome["attempts"][1]["feasible"] is True

    log = json.loads(Path(outcome["recovery_log"]).read_text(encoding="utf-8"))
    assert log["original_query"]["design_space"]["allowed_materials"] == ["OPC"]
    assert "slag" in log["final_query"]["design_space"]["allowed_materials"]

    run_dir = Path(outcome["final"]["run_dir"])
    candidates = pd.read_csv(run_dir / "task_run" / "design" / "final_candidates.csv")
    assert candidates.iloc[0]["meta__recipe_id"] == "low_opc"


def test_recovery_loop_gives_up_without_proposals(tmp_path, toy_registry):
    query = _query()
    query["design_space"]["age_days"] = 28
    # unknown explicit system leaves zero candidates and clearing it is the
    # only proposal; after that materials remain blocked, so at most
    # max_attempts changes are applied and everything is logged.
    query["material_system"] = "no_such_system"
    outcome = run_design_with_recovery(
        query,
        out=tmp_path / "recovery2",
        db=tmp_path / "db2",
        use_mock=True,
        skip_validation=True,
        max_attempts=1,
        model_registry=toy_registry,
    )
    assert outcome["status"] == "failed"
    assert outcome["attempts"][0]["applied_proposal"]["kind"] == "clear_material_system_filter"


def test_agent_tool_wrappers_follow_contract(tmp_path, toy_registry):
    result = agent_tools.diagnose_design_feasibility(
        _query(), model_registry=str(toy_registry)
    )
    assert result["contract"] == agent_tools.TOOL_CONTRACT
    assert result["ok"] is True
    assert result["summary"]["feasible"] is False

    proposals = agent_tools.propose_constraint_relaxation(
        _query(), model_registry=str(toy_registry)
    )
    assert proposals["ok"] is True
    assert proposals["summary"]["proposal_count"] >= 1

    loop = agent_tools.run_design_with_recovery(
        _query(), str(tmp_path / "loop"), str(tmp_path / "loop_db"),
        model_registry=str(toy_registry),
    )
    assert loop["ok"] is True
    assert loop["summary"]["status"] == "complete"
    assert loop["artifacts"]["recovery_log"].endswith("recovery_log.json")
