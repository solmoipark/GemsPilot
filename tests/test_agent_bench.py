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


def _qa_outcome(**overrides):
    outcome = {
        "final_text": "The porosity is 0.600 and the pH is 12.6.",
        "stop_reason": "final_answer",
        "steps": 3,
        "tool_calls": [
            {"step": 0, "tool": "run_forward", "ok": True},
            {"step": 1, "tool": "read_artifact", "ok": True},
        ],
        "usage": {"prompt_tokens": 5000, "completion_tokens": 400, "cost_usd": 0.004},
        "providers": ["TestProvider"],
    }
    outcome.update(overrides)
    return outcome


def test_grade_agent_qa_numeric_pass_and_fail():
    from gemspilot.agent_bench import grade_agent_qa

    scenario = {
        "grading": {"answer_kind": "numeric", "target": 0.5997, "rel_tol": 0.02,
                    "extract": "porosity"},
        "constraints": {"max_tool_calls": 6, "min_tool_calls": 2},
        "expect": {"tools_called_include": ["run_forward"]},
    }
    checks, metrics = grade_agent_qa(scenario, _qa_outcome())
    assert all(check["ok"] for check in checks)
    assert metrics["tool_calls"] == 2
    assert metrics["unnecessary_calls"] == 0

    wrong = _qa_outcome(final_text="The porosity is 0.30 and the pH is 9.")
    checks, _ = grade_agent_qa(scenario, wrong)
    failed = {check["name"] for check in checks if not check["ok"]}
    assert "numeric_answer" in failed


def test_grade_agent_qa_extract_hint_prefers_labelled_number():
    from gemspilot.agent_bench import grade_agent_qa

    scenario = {"grading": {"answer_kind": "numeric", "target": 12.6,
                            "rel_tol": 0.02, "extract": "pH"}}
    outcome = _qa_outcome(final_text="After 28 days (0.45 w/b), the pH is 12.61.")
    checks, _ = grade_agent_qa(scenario, outcome)
    numeric = next(check for check in checks if check["name"] == "numeric_answer")
    assert numeric["ok"]


def test_grade_agent_qa_refusal_requires_no_execution():
    from gemspilot.agent_bench import grade_agent_qa

    scenario = {"grading": {"answer_kind": "refusal"}}
    good = _qa_outcome(
        final_text="This request is infeasible with the available models; "
                   "please clarify the target strength.",
        tool_calls=[{"step": 0, "tool": "diagnose_design_feasibility", "ok": True}],
    )
    checks, _ = grade_agent_qa(scenario, good)
    assert all(check["ok"] for check in checks)

    bad = _qa_outcome(final_text="Sure, I ran it anyway; porosity 0.6.")
    checks, _ = grade_agent_qa(scenario, bad)
    failed = {check["name"] for check in checks if not check["ok"]}
    assert {"refusal_language", "no_execution_before_refusal"} <= failed


def test_grade_agent_qa_forbidden_tools_and_stop_reason():
    from gemspilot.agent_bench import grade_agent_qa

    scenario = {
        "grading": {"answer_kind": "behavior"},
        "constraints": {"forbidden_tools": ["run_task"]},
    }
    outcome = _qa_outcome(
        stop_reason="max_steps",
        tool_calls=[{"step": 0, "tool": "run_task", "ok": True}],
    )
    checks, _ = grade_agent_qa(scenario, outcome)
    failed = {check["name"] for check in checks if not check["ok"]}
    assert {"final_answer_reached", "never_calls_run_task"} <= failed


def test_grade_agent_qa_choice_word_boundary():
    from gemspilot.agent_bench import grade_agent_qa

    scenario = {
        "grading": {"answer_kind": "choice", "target": "feasible",
                    "must_not_contain": ["infeasible", "not feasible"]},
    }
    good = _qa_outcome(final_text="The design is feasible with the OPC_slag model.")
    checks, _ = grade_agent_qa(scenario, good)
    assert next(c for c in checks if c["name"] == "choice_answer")["ok"]

    bad = _qa_outcome(final_text="The design is infeasible: the target phase is unknown.")
    checks, _ = grade_agent_qa(scenario, bad)
    assert not next(c for c in checks if c["name"] == "choice_answer")["ok"]
