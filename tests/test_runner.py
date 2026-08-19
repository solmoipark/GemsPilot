from pathlib import Path

from gemspilot.runner import (
    Episode,
    ToolSpec,
    _apply_protocol,
    _policy_check,
    _remap_workspace_args,
    build_tool_schema,
    default_toolset,
)


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
