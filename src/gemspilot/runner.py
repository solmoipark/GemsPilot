"""LLM agent runner: a provider-agnostic tool-calling loop over the kernel tools.

The runner connects a language model (any provider supported by litellm) to
the GemsPilot tool layer. The model selects and parameterizes tools; every
tool executes deterministic kernel code. Governance is enforced outside the
model: an approval policy decides which calls run (read-only and mock calls
auto-approve; real xGEMS execution requires an explicit episode flag; tools
with the ``write`` policy only run as dry runs unless the episode allows
real execution), tool paths are remapped into an episode workspace, and
every step is recorded to a JSONL trajectory for failure analysis.

Two output-access protocols are supported, mirroring the ablation of
PHREEQC-MCQ-200 (arXiv:2607.00436):

- ``full``: the complete ToolResult JSON is returned to the model.
- ``toc``: only a table-of-contents view (ok flag, summary key listing with
  short previews, artifact names) is returned; details must be fetched with
  ``read_artifact`` / follow-up calls.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import agent_tools

DEFAULT_SYSTEM_PROMPT = """You are GemsPilot, an assistant that operates a deterministic \
thermodynamic-modeling kernel for blended cementitious binders through tools. \
You never compute chemistry yourself: every number you report must come from a tool result. \
Recipes use a 100 g binder basis (e.g. "OPC 60, slag 40, w/b 0.45, age 28"). \
Forward queries and task specifications are YAML documents. \
Use the workspace paths you are given for all outputs. Prefer mock/cached execution; \
real xGEMS runs are only possible when the task explicitly allows them. \
When you have the answer, reply with a concise final message and stop calling tools. \
If a request is infeasible or ambiguous, say so explicitly instead of guessing.

Forward-query YAML template (adapt binders/w_b/ages; keep the other keys):
name: my_run
task: forward_calculation
recipe: {binders: {OPC: 60, slag: 40}, w_b: 0.45}
age_grid: {values: [28.0]}
outputs: {phase_masses: all, phase_volumes: all, phase_volumes_reconstructed: all, aqueous_species: all, scalars: all}
plots: []
response_summary: {phases: [Portlandite], scalars: [pH, porosity]}

Numeric results live in the run artifacts: after run_forward, use read_artifact on the
time_series or response-summary artifact paths returned in the result to obtain values."""

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", dict: "object"}


@dataclass
class ToolSpec:
    name: str
    func: Callable[..., dict[str, Any]]
    # "read": always approved; "mock_ok": approved unless use_mock is false
    # and the episode disallows real execution; "real_gated": never approved
    # with use_mock false; "write": approved only with dry_run true (the
    # default) unless the episode allows real execution.
    policy: str  # "read" | "mock_ok" | "real_gated" | "write"
    description: str = ""


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return " ".join(doc.strip().split("\n\n")[0].split())


_INNER_JSON_TYPES = {"str": "string", "int": "integer", "float": "number",
                     "bool": "boolean", "dict": "object"}


def _annotation_type(annotation: Any) -> dict[str, Any]:
    if annotation in _JSON_TYPES:
        entry = {"type": _JSON_TYPES[annotation]}
    else:
        text = str(annotation)
        entry = {"type": "string"}
        for py, js in [("str", "string"), ("int", "integer"), ("float", "number"),
                       ("bool", "boolean"), ("list", "array"), ("dict", "object")]:
            if text.startswith(py) or f"{py} |" in text or f"| {py}" in text:
                entry = {"type": js}
                break
    if entry["type"] == "array":
        # Google's function-calling API rejects array parameters without an
        # explicit items schema; derive it from list[...] or default to string.
        match = re.search(r"list\[\s*(str|int|float|bool|dict)", str(annotation))
        entry["items"] = {"type": _INNER_JSON_TYPES.get(match.group(1), "string") if match else "string"}
    return entry


def build_tool_schema(spec: ToolSpec) -> dict[str, Any]:
    signature = inspect.signature(spec.func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name in {"self"}:
            continue
        entry = _annotation_type(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)
        properties[name] = entry
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description or _first_paragraph(spec.func.__doc__),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def default_toolset() -> list[ToolSpec]:
    tools = agent_tools
    return [
        ToolSpec("validate_task_query", tools.validate_task_query, "read"),
        ToolSpec("validate_forward_query", tools.validate_forward_query, "read"),
        ToolSpec("run_forward", tools.run_forward, "mock_ok"),
        ToolSpec("run_task", tools.run_task, "mock_ok"),
        ToolSpec("diagnose_design_feasibility", tools.diagnose_design_feasibility, "read"),
        ToolSpec("propose_constraint_relaxation", tools.propose_constraint_relaxation, "read"),
        ToolSpec("run_design_with_recovery", tools.run_design_with_recovery, "mock_ok"),
        ToolSpec("query_past_runs", tools.query_past_runs, "read"),
        ToolSpec("query_model_registry", tools.query_model_registry, "read"),
        ToolSpec("check_coverage", tools.check_coverage, "read"),
        ToolSpec("list_run_artifacts", tools.list_run_artifacts, "read"),
        ToolSpec("read_artifact", tools.read_artifact, "read"),
        ToolSpec("recall_session", tools.recall_session, "read"),
        ToolSpec("filter_candidates", tools.filter_candidates, "read"),
        ToolSpec("calibrate_scm_kinetics", tools.calibrate_scm_kinetics, "mock_ok"),
    ]


@dataclass
class Episode:
    model: str
    workspace: Path
    allow_real: bool = False
    protocol: str = "full"  # or "toc"
    max_steps: int = 12
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    toolset: list[ToolSpec] = field(default_factory=default_toolset)
    # Per-model litellm.completion overrides (e.g. {"temperature": None} for
    # reasoning models that reject the parameter). None values drop the key.
    completion_params: dict[str, Any] = field(default_factory=dict)


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _sanitize_llm_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Drop filler values some models emit for optional parameters.

    OpenAI-family models tend to populate every optional parameter with a
    type-default filler ("" for strings, 0 for max_xgems_calls). Passing
    those through crashes the kernel (e.g. dat_lst="" opens Path("") ==
    '.'), so treat them as "not provided". Boolean False is never dropped —
    it is load-bearing for use_mock gating.
    """
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key == "max_xgems_calls" and value in (0, "0"):
            continue
        cleaned[key] = value
    return cleaned


def _remap_workspace_args(name: str, arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Confine writable paths to the episode workspace."""
    remapped = dict(arguments)
    for key in ["out", "db", "session"]:
        if key in remapped and remapped[key] is not None:
            candidate = Path(str(remapped[key]))
            if not candidate.is_absolute():
                remapped[key] = str(workspace / candidate)
            else:
                try:
                    candidate.relative_to(workspace)
                except ValueError:
                    remapped[key] = str(workspace / candidate.name)
    return remapped


def _apply_protocol(result: dict[str, Any], protocol: str) -> dict[str, Any]:
    if protocol != "toc":
        return result
    summary = result.get("summary") or {}
    toc = {}
    for key, value in summary.items():
        preview = json.dumps(value, ensure_ascii=False, default=str)
        toc[key] = preview if len(preview) <= 80 else preview[:77] + "..."
    return {
        "contract": result.get("contract"),
        "tool": result.get("tool"),
        "ok": result.get("ok"),
        "summary_toc": toc,
        "artifacts": result.get("artifacts"),
        "warnings": result.get("warnings"),
        "error": result.get("error"),
        "note": "Table-of-contents view. Use read_artifact/list_run_artifacts for details.",
    }


def _policy_check(spec: ToolSpec, arguments: dict[str, Any], allow_real: bool) -> str | None:
    """Return a refusal message, or None if the call is approved.

    ``read`` tools always run. ``write`` tools run only as dry runs
    (``dry_run`` defaults to true) unless the episode allows real execution.
    ``mock_ok`` / ``real_gated`` tools are gated on ``use_mock`` as before.
    """
    if spec.policy == "read":
        return None
    if spec.policy == "write":
        dry_run = arguments.get("dry_run", True)
        if dry_run in (False, "false", "False", 0) and not allow_real:
            return (
                "DENIED by approval policy: writing outside dry-run is not allowed in this "
                "episode. Re-run with dry_run=true."
            )
        return None
    use_mock = arguments.get("use_mock", True)
    if use_mock in (False, "false", "False", 0):
        if spec.policy == "real_gated" or not allow_real:
            return (
                "DENIED by approval policy: real xGEMS execution is not allowed in this "
                "episode. Re-run with use_mock=true."
            )
    return None


def run_episode(task: str, episode: Episode) -> dict[str, Any]:
    """Run one agent episode; returns the outcome and writes a JSONL trajectory."""
    import litellm

    litellm.suppress_debug_info = True
    load_env_file()
    workspace = episode.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    # Whitelist this episode's workspace for artifact access, but restore the
    # variable afterwards: unbounded accumulation across a long sweep exceeds
    # the Windows 32767-character environment-variable limit and crashes
    # every subsequent episode in the process.
    base_roots = os.environ.get(agent_tools.ARTIFACT_ROOTS_ENV, "")
    if str(workspace) not in base_roots.split(os.pathsep):
        os.environ[agent_tools.ARTIFACT_ROOTS_ENV] = (
            f"{base_roots}{os.pathsep}{workspace}" if base_roots else str(workspace)
        )
    try:
        return _run_episode_inner(task, episode, workspace)
    finally:
        os.environ[agent_tools.ARTIFACT_ROOTS_ENV] = base_roots


def _run_episode_inner(task: str, episode: Episode, workspace: Path) -> dict[str, Any]:
    import litellm

    tool_specs = {spec.name: spec for spec in episode.toolset}
    tool_schemas = [build_tool_schema(spec) for spec in episode.toolset]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": episode.system_prompt
         + f"\nWorkspace directory for outputs: {workspace} (subpaths like out=run1, db=db)."},
        {"role": "user", "content": task},
    ]
    trajectory_path = workspace / "trajectory.jsonl"
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    tool_call_log: list[dict[str, Any]] = []
    providers_seen: list[str] = []
    final_text: str | None = None
    stop_reason = "max_steps"

    def record(event: dict[str, Any]) -> None:
        with trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    request_base: dict[str, Any] = {"temperature": episode.temperature}
    request_base.update(episode.completion_params)
    # A hung provider request must not stall a whole sweep: bound each call
    # and let litellm retry transient failures. models.yaml params can override.
    request_base.setdefault("timeout", 300)
    request_base.setdefault("num_retries", 2)
    # Bound per-step generation: unbounded no-tool reasoning can run for many
    # minutes on slow providers. 4096 tokens is ample for answers and replans.
    request_base.setdefault("max_tokens", 4096)
    if episode.model.startswith("openrouter/"):
        # Ask OpenRouter to report the billed cost in usage (credits == USD).
        extra_body = dict(request_base.get("extra_body") or {})
        extra_body.setdefault("usage", {"include": True})
        request_base["extra_body"] = extra_body
    request_base = {k: v for k, v in request_base.items() if v is not None}

    for step in range(episode.max_steps):
        started = time.perf_counter()
        response = litellm.completion(
            model=episode.model,
            messages=messages,
            # No-tool baseline episodes omit the tools parameter entirely;
            # an empty list is rejected by some providers.
            **({"tools": tool_schemas} if tool_schemas else {}),
            **request_base,
        )
        elapsed = time.perf_counter() - started
        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        step_cost = None
        if usage is not None:
            usage_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            usage_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            step_cost = getattr(usage, "cost", None)  # OpenRouter-native billed cost
        if step_cost:
            usage_totals["cost_usd"] += float(step_cost)
        else:
            try:
                usage_totals["cost_usd"] += (
                    litellm.completion_cost(completion_response=response) or 0.0
                )
            except Exception:  # noqa: BLE001 - cost lookup is best-effort
                pass

        provider = getattr(response, "provider", None)  # OpenRouter serving provider
        if provider and provider not in providers_seen:
            providers_seen.append(provider)
        tool_calls = getattr(message, "tool_calls", None) or []
        record({
            "step": step, "latency_s": round(elapsed, 2),
            "provider": provider,
            "assistant": message.content,
            "tool_calls": [
                {"name": c.function.name, "arguments": c.function.arguments} for c in tool_calls
            ],
        })

        if not tool_calls:
            final_text = message.content or ""
            stop_reason = "final_answer"
            break

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in tool_calls
            ],
        })
        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                payload = {"ok": False, "error": f"invalid tool arguments: {exc}"}
            else:
                arguments = _sanitize_llm_args(arguments)
                spec = tool_specs.get(name)
                if spec is None:
                    payload = {"ok": False, "error": f"unknown tool {name!r}"}
                else:
                    refusal = _policy_check(spec, arguments, episode.allow_real)
                    if refusal:
                        payload = {"ok": False, "error": refusal}
                    else:
                        arguments = _remap_workspace_args(name, arguments, workspace)
                        try:
                            payload = spec.func(**arguments)
                        except TypeError as exc:
                            payload = {"ok": False, "error": f"bad arguments: {exc}"}
                        except Exception as exc:  # noqa: BLE001
                            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        else:
                            payload = _apply_protocol(payload, episode.protocol)
            entry = {"step": step, "tool": name, "ok": payload.get("ok")}
            if name in ("run_forward", "run_task", "run_design_with_recovery"):
                entry["attempted_real"] = arguments.get("use_mock", True) in (
                    False, "false", "False", 0
                )
            tool_call_log.append(entry)
            record({"step": step, "tool_result": {"tool": name, "payload": payload}})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            })

    outcome = {
        "model": episode.model,
        "providers": providers_seen,
        "protocol": episode.protocol,
        "final_text": final_text,
        "stop_reason": stop_reason,
        "steps": step + 1,
        "tool_calls": tool_call_log,
        "usage": usage_totals,
        "trajectory": str(trajectory_path),
        "workspace": str(workspace),
    }
    (workspace / "episode_outcome.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return outcome
