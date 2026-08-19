import json
from pathlib import Path

import pandas as pd

from gemspilot import agent_tools
from gemspilot.agent_session import append_session_event, read_session_events


FORWARD_3_AGES = {
    "name": "budget_series",
    "recipe": {"binders": {"OPC": 60, "slag": 40}, "w_b": 0.45},
    "age_grid": {"values": [1.0, 7.0, 28.0]},
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


def test_forward_budget_stops_remaining_ages(tmp_path):
    result = agent_tools.run_forward(
        FORWARD_3_AGES,
        str(tmp_path / "run"),
        str(tmp_path / "db"),
        use_mock=True,
        max_xgems_calls=2,
    )
    frame = pd.read_csv(Path(result["artifacts"]["forward_dir"]) / "time_series.csv")
    statuses = frame["chemistry_status"].tolist()
    assert statuses.count("complete") == 2
    assert statuses.count("skipped_budget") == 1
    assert "budget exhausted" in " ".join(result["warnings"])


def test_forward_budget_not_consumed_by_cache_hits(tmp_path):
    first = agent_tools.run_forward(
        FORWARD_3_AGES, str(tmp_path / "run1"), str(tmp_path / "db"), use_mock=True
    )
    assert first["ok"] is True
    # all three chemistries now cached; a budget of 1 must still complete all ages
    second = agent_tools.run_forward(
        FORWARD_3_AGES,
        str(tmp_path / "run2"),
        str(tmp_path / "db"),
        use_mock=True,
        max_xgems_calls=1,
    )
    assert second["ok"] is True
    frame = pd.read_csv(Path(second["artifacts"]["forward_dir"]) / "time_series.csv")
    assert frame["chemistry_status"].tolist() == ["complete"] * 3


def test_session_events_roundtrip(tmp_path):
    session = tmp_path / "session"
    append_session_event(session, tool="run_forward", status="ok", summary={"n": 1})
    append_session_event(session, tool="filter_candidates", status="ok", summary={"n": 2})
    events = read_session_events(session)
    assert [e["tool"] for e in events] == ["run_forward", "filter_candidates"]
    assert [e["event_index"] for e in events] == [0, 1]
    only = read_session_events(session, tool="run_forward")
    assert len(only) == 1


def test_run_forward_logs_to_session_and_recall(tmp_path):
    session = str(tmp_path / "session")
    result = agent_tools.run_forward(
        FORWARD_3_AGES, str(tmp_path / "run"), str(tmp_path / "db"),
        use_mock=True, session=session,
    )
    assert result["ok"] is True
    recall = agent_tools.recall_session(session)
    assert recall["ok"] is True
    assert recall["summary"]["event_count"] == 1
    event = recall["summary"]["events"][0]
    assert event["tool"] == "run_forward"
    assert event["status"] == "ok"
    assert "forward_dir" in event["artifacts"]


def test_filter_candidates_refines_and_chains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "final_candidates.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["a", "b", "c"],
            "x__OPC": [70.0, 40.0, 30.0],
            "x__slag": [30.0, 60.0, 70.0],
            "y__porosity": [0.34, 0.28, 0.26],
        }
    ).to_csv(csv_path, index=False)

    session = str(tmp_path / "session")
    result = agent_tools.filter_candidates(
        str(csv_path),
        [{"column": "slag", "op": "<", "value": 65}, {"column": "porosity", "op": "<=", "value": 0.30}],
        session=session,
    )
    assert result["ok"] is True
    assert result["summary"]["matched_rows"] == 1
    assert result["summary"]["rows"][0]["meta__recipe_id"] == "b"
    filtered = Path(result["artifacts"]["filtered_csv"])
    assert filtered.exists()
    # chaining: filter the filtered file again
    chained = agent_tools.filter_candidates(str(filtered), [{"column": "x__OPC", "op": ">", "value": 0}])
    assert chained["summary"]["matched_rows"] == 1
    # session recorded the refinement
    events = read_session_events(session)
    assert events[-1]["tool"] == "filter_candidates"


def test_filter_candidates_reports_bad_column_and_op(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "c.csv"
    pd.DataFrame({"x__OPC": [1.0]}).to_csv(csv_path, index=False)
    bad_col = agent_tools.filter_candidates(str(csv_path), [{"column": "nope", "op": "<", "value": 1}])
    assert bad_col["ok"] is False and "not found" in bad_col["error"]
    bad_op = agent_tools.filter_candidates(str(csv_path), [{"column": "OPC", "op": "~", "value": 1}])
    assert bad_op["ok"] is False and "Unsupported op" in bad_op["error"]
