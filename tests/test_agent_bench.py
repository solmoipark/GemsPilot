import json
from pathlib import Path

from gemspilot.agent_bench import run_agent_bench

FORWARD = {
    "name": "bench_fwd",
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
    "response_summary": {"phases": ["Mock Portlandite"], "scalars": ["pH"]},
}

FORWARD_3 = dict(FORWARD, age_grid={"values": [1.0, 7.0, 28.0]}, name="bench_fwd3", task=None)
FORWARD_3 = {k: v for k, v in FORWARD_3.items() if v is not None}


def test_bench_runs_mock_scenarios_and_reports(tmp_path):
    config = {
        "scenarios": [
            {
                "id": "fwd_ok",
                "kind": "forward_mock",
                "repeat": 2,
                "forward_query": FORWARD,
                "expect": {"status": "complete", "row_count": 1},
            },
            {
                "id": "budget",
                "kind": "budget_guardrail",
                "max_xgems_calls": 2,
                "forward_query": FORWARD_3,
                "expect": {"completed_rows": 2, "skipped_budget_rows": 1},
            },
            {
                "id": "session",
                "kind": "session_recall",
                "forward_query": FORWARD,
                "expect": {"min_events": 1},
            },
        ]
    }
    summary = run_agent_bench(config, out=tmp_path / "bench")
    assert summary["total"] == 3
    assert summary["failed"] == 0
    assert summary["passed"] == 3
    report = json.loads((tmp_path / "bench" / "bench_report.json").read_text(encoding="utf-8"))
    assert report["passed"] == 3
    fwd = next(s for s in report["scenarios"] if s["id"] == "fwd_ok")
    assert any(c["name"] == "rerun_stability" and c["ok"] for c in fwd["checks"])
    assert (tmp_path / "bench" / "bench_report.md").exists()


def test_bench_skips_when_requirements_missing(tmp_path):
    config = {
        "scenarios": [
            {
                "id": "needs_db",
                "kind": "diagnose",
                "requires": [str(tmp_path / "no_such_dir")],
                "design_query": {"name": "x"},
                "expect": {"feasible": True},
            }
        ]
    }
    summary = run_agent_bench(config, out=tmp_path / "bench")
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    assert "missing required path" in summary["scenarios"][0]["reason"]


# NOTE: keep this test's name and scenario ids short - the run writes
# db/chemistry/<64-char-hash>/... under pytest's tmp_path, and a long test
# name pushes file paths past the Windows 260-character limit.
def test_bench_bad_expect(tmp_path):
    config = {
        "scenarios": [
            {
                "id": "wrong",
                "kind": "forward_mock",
                "forward_query": FORWARD,
                "expect": {"status": "complete", "row_count": 99},
            },
            {"id": "unknown", "kind": "no_such_kind"},
        ]
    }
    summary = run_agent_bench(config, out=tmp_path / "bench")
    assert summary["failed"] == 2
    wrong = summary["scenarios"][0]
    bad_check = next(c for c in wrong["checks"] if not c["ok"])
    assert bad_check["name"] == "row_count"
    md = (tmp_path / "bench" / "bench_report.md").read_text(encoding="utf-8")
    assert "Failures" in md
