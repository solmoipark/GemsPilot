"""Session memory for multi-turn agent refinement (roadmap Phase 2).

A session is a JSONL event log on disk. Execution tools append one compact
event per run (tool name, status, key summary fields, artifact paths), so a
later turn can resolve references like "the low-slag candidates from before"
by recalling recent events and re-reading their artifacts - no hidden state,
everything inspectable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSION_FILE_NAME = "agent_session.jsonl"


def _session_file(session: str | Path) -> Path:
    path = Path(session)
    if path.suffix == ".jsonl":
        return path
    return path / SESSION_FILE_NAME


def append_session_event(
    session: str | Path,
    *,
    tool: str,
    status: str,
    summary: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Append one event to the session log and return it."""
    path = _session_file(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_session_events(session)
    event = {
        "event_index": len(prior),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "status": status,
        "summary": summary or {},
        "artifacts": artifacts or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return event


def read_session_events(
    session: str | Path,
    *,
    tool: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read session events, newest last; optionally filter by tool name."""
    path = _session_file(session)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if tool and str(event.get("tool")) != tool:
            continue
        events.append(event)
    if limit is not None:
        events = events[-int(limit):]
    return events
