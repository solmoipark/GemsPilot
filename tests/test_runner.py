from pathlib import Path

from gemspilot.runner import (
    Episode,
    ToolSpec,
    _apply_protocol,
    _policy_check,
    _remap_workspace_args,
    _sanitize_llm_args,
    build_tool_schema,
    default_toolset,
    toolset_discovery_warnings,
)


def test_sanitize_drops_filler_values_but_keeps_false():
    cleaned = _sanitize_llm_args({
        "forward_query": "name: x",
        "dat_lst": "",
        "retry_water_policy": " ",
        "session": None,
        "max_xgems_calls": 0,
        "use_mock": False,
        "disable_plots": True,
    })
    assert cleaned == {"forward_query": "name: x", "use_mock": False, "disable_plots": True}
    assert _sanitize_llm_args({"max_xgems_calls": 3})["max_xgems_calls"] == 3


def test_array_parameters_carry_items_schema():
    for spec in default_toolset():
        schema = build_tool_schema(spec)
        for prop in schema["function"]["parameters"]["properties"].values():
            if prop["type"] == "array":
                assert prop["items"]["type"] in {
                    "string", "integer", "number", "boolean", "object"
                }


def test_default_toolset_schemas_are_valid_function_specs():
    specs = default_toolset()
    names = {spec.name for spec in specs}
    assert {"run_forward", "read_artifact", "diagnose_design_feasibility"} <= names
    for spec in specs:
        schema = build_tool_schema(spec)
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] == spec.name
        assert function["description"]
        params = function["parameters"]
        assert params["type"] == "object"
        for required in params["required"]:
            assert required in params["properties"]


def test_policy_denies_real_execution_unless_allowed():
    spec = next(s for s in default_toolset() if s.name == "run_forward")
    assert _policy_check(spec, {"use_mock": True}, allow_real=False) is None
    denial = _policy_check(spec, {"use_mock": False}, allow_real=False)
    assert denial is not None and "DENIED" in denial
    assert _policy_check(spec, {"use_mock": False}, allow_real=True) is None
    read_spec = next(s for s in default_toolset() if s.name == "read_artifact")
    assert _policy_check(read_spec, {}, allow_real=False) is None


def test_workspace_remap_confines_writable_paths(tmp_path):
    workspace = tmp_path / "ep"
    remapped = _remap_workspace_args(
        "run_forward",
        {"out": "run1", "db": "C:/somewhere/else/db", "forward_query": "name: x"},
        workspace,
    )
    assert remapped["out"] == str(workspace / "run1")
    assert remapped["db"] == str(workspace / "db")
    assert remapped["forward_query"] == "name: x"


def test_toc_protocol_reduces_payload():
    result = {
        "contract": "inverse-gems-tool/1.0",
        "tool": "run_forward",
        "ok": True,
        "summary": {"status": "complete", "big": {"rows": list(range(200))}},
        "artifacts": {"forward_dir": "x"},
        "warnings": [],
        "error": None,
    }
    toc = _apply_protocol(result, "toc")
    assert toc["ok"] is True
    assert "summary" not in toc
    assert toc["summary_toc"]["status"] == '"complete"'
    assert toc["summary_toc"]["big"].endswith("...")
    assert _apply_protocol(result, "full") is result


def test_episode_defaults():
    episode = Episode(model="gpt-4.1-mini", workspace=Path("w"))
    assert episode.allow_real is False
    assert episode.protocol == "full"
    assert episode.max_steps == 12
    assert episode.completion_params == {}


# --- toolset extension points ---
BUILTIN_TOOL_NAMES = {
    "validate_task_query", "validate_forward_query", "run_forward", "run_task",
    "diagnose_design_feasibility", "propose_constraint_relaxation",
    "run_design_with_recovery", "query_past_runs", "query_model_registry",
    "check_coverage", "list_run_artifacts", "read_artifact", "recall_session",
    "filter_candidates", "calibrate_scm_kinetics",
}


def test_default_toolset_without_discovery_is_exactly_the_builtins():
    specs = default_toolset(discover=False)
    assert len(specs) == 15
    assert {spec.name for spec in specs} == BUILTIN_TOOL_NAMES


def test_default_toolset_appends_extra_and_skips_duplicates():
    extra = ToolSpec("x", lambda: {}, "read")
    shadow = ToolSpec("run_forward", lambda: {}, "read")
    specs = default_toolset(extra=[extra, shadow, extra], discover=False)
    names = [spec.name for spec in specs]
    assert names[:15] == [spec.name for spec in default_toolset(discover=False)]
    assert names[15:] == ["x"]
    assert next(spec for spec in specs if spec.name == "run_forward").policy == "mock_ok"


def test_default_toolset_accepts_toolspec_like_objects():
    class Duck:
        name = "duck"
        policy = "read"

        @staticmethod
        def func() -> dict:
            return {}

    specs = default_toolset(extra=[Duck()], discover=False)
    duck = next(spec for spec in specs if spec.name == "duck")
    assert isinstance(duck, ToolSpec)
    assert duck.policy == "read"


def test_default_toolset_discovery_never_raises_and_reports_warnings(monkeypatch):
    import importlib.metadata
    from types import SimpleNamespace

    import gemspilot.runner as runner_module

    def broken_load():
        raise RuntimeError("boom")

    fake_entry_points = [
        SimpleNamespace(name="broken", value="pkg.mod:BROKEN", load=broken_load),
        SimpleNamespace(
            name="good",
            value="pkg.mod:GOOD",
            load=lambda: [ToolSpec("y", lambda: {}, "read"), ToolSpec("run_forward", lambda: {}, "read")],
        ),
    ]
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda **kwargs: list(fake_entry_points) if kwargs.get("group") == runner_module.TOOLSET_ENTRY_POINT_GROUP else []
    )
    specs = default_toolset()
    names = [spec.name for spec in specs]
    assert names[:15] == [spec.name for spec in default_toolset(discover=False)]
    assert names[15:] == ["y"]
    warnings = toolset_discovery_warnings()
    assert any("'broken'" in message and "boom" in message for message in warnings)
    assert any("'run_forward'" in message for message in warnings)


def test_policy_check_semantics_unchanged_for_read_mock_ok_and_real_gated():
    read = ToolSpec("r", lambda: {}, "read")
    mock_ok = ToolSpec("m", lambda: {}, "mock_ok")
    real_gated = ToolSpec("g", lambda: {}, "real_gated")
    for allow_real in (False, True):
        assert _policy_check(read, {"use_mock": False}, allow_real=allow_real) is None
        assert _policy_check(mock_ok, {}, allow_real=allow_real) is None
        assert _policy_check(mock_ok, {"use_mock": True}, allow_real=allow_real) is None
        assert _policy_check(real_gated, {"use_mock": True}, allow_real=allow_real) is None
        assert _policy_check(real_gated, {"use_mock": False}, allow_real=allow_real) is not None
    assert _policy_check(mock_ok, {"use_mock": False}, allow_real=False) is not None
    assert _policy_check(mock_ok, {"use_mock": "false"}, allow_real=False) is not None
    assert _policy_check(mock_ok, {"use_mock": False}, allow_real=True) is None
