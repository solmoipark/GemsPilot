"""Scenario (a) of the agent roadmap: infeasible-design diagnosis and recovery.

When routing finds no eligible material-system model for an inverse design
query, this module (1) diagnoses why without raising, (2) proposes concrete,
deterministic constraint relaxations, and (3) optionally runs a bounded
observe-replan loop that applies one relaxation per attempt and records
exactly what was changed.

The LLM host never invents relaxations: proposals are generated and ranked by
deterministic code from the router's own blocker records; a host (or human)
may only choose among them.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from inverse_gems.model_router import (
    DEFAULT_AGE_DAYS,
    candidate_rows,
    load_model_registry,
    query_age_days,
    query_allowed_materials,
    query_design_space,
    query_explicit_systems,
)
from inverse_gems.utils import config_path, load_yaml, write_json

_MATERIAL_BLOCKER = re.compile(r"requires unallowed material\(s\): \[(?P<items>[^\]]*)\]")
_MISSING_TARGET_BLOCKER = re.compile(r"target '(?P<target>[^']+)' missing from diagnostics")
_TARGET_STATUS_BLOCKER = re.compile(r"target '(?P<target>[^']+)' is (?P<status>[a-zA-Z_]+)")


def diagnose_design_query(
    query_data: dict[str, Any],
    *,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    target_policy: str = "recommended",
    default_age_days: float = DEFAULT_AGE_DAYS,
) -> dict[str, Any]:
    """Non-raising feasibility diagnosis for an API-facing design query."""
    registry = load_model_registry(model_registry)
    profiles = load_yaml(material_systems_config or config_path("material_systems.yaml"))
    candidates, warnings = candidate_rows(
        query_data=query_data,
        registry=registry,
        profiles=profiles,
        target_policy=target_policy,  # type: ignore[arg-type]
        default_age_days=default_age_days,
        reaction_model_id=None,
        reaction_model_signature=None,
        reaction_model_config=None,
    )
    eligible = [row for row in candidates if row["eligible"]]
    blocker_histogram: dict[str, int] = {}
    for row in candidates:
        for blocker in row["blockers"]:
            kind = _blocker_kind(blocker)
            blocker_histogram[kind] = blocker_histogram.get(kind, 0) + 1

    registry_entries = registry.get("models") or registry.get("entries") or []
    registry_ages = sorted(
        {
            float(entry["age_days"])
            for entry in registry_entries
            if isinstance(entry, dict) and entry.get("age_days") is not None
        }
    )
    requested_age = query_age_days(query_data)

    compact = [
        {
            "id": row.get("id"),
            "material_system": row.get("material_system"),
            "age_days": row.get("age_days"),
            "score": row.get("score"),
            "eligible": row.get("eligible"),
            "blockers": row.get("blockers"),
            "profile_allowed_materials": row.get("profile_allowed_materials"),
            "matched_targets": row.get("matched_targets"),
        }
        for row in candidates[:10]
    ]
    return {
        "feasible": bool(eligible),
        "eligible_count": len(eligible),
        "candidate_count": len(candidates),
        "selected_id": eligible[0].get("id") if eligible else None,
        "requested_age_days": requested_age,
        "requested_allowed_materials": query_allowed_materials(query_data),
        "explicit_material_systems": query_explicit_systems(query_data),
        "target_policy": target_policy,
        "blocker_histogram": blocker_histogram,
        "registry_age_options": registry_ages,
        "candidates": compact,
        "warnings": warnings,
    }


def _blocker_kind(blocker: str) -> str:
    if _MATERIAL_BLOCKER.search(blocker):
        return "unallowed_materials"
    if _MISSING_TARGET_BLOCKER.search(blocker):
        return "target_missing"
    if _TARGET_STATUS_BLOCKER.search(blocker):
        return "target_not_recommended"
    return "other"


def _parse_material_blocker(blocker: str) -> list[str]:
    match = _MATERIAL_BLOCKER.search(blocker)
    if not match:
        return []
    return [item.strip().strip("'\"") for item in match.group("items").split(",") if item.strip()]


def _add_allowed_materials(query_data: dict[str, Any], materials: list[str]) -> dict[str, Any]:
    modified = copy.deepcopy(query_data)
    design_space = dict(modified.get("design_space") or {})
    allowed = list(design_space.get("allowed_materials") or [])
    for material in materials:
        if material not in allowed:
            allowed.append(material)
    design_space["allowed_materials"] = allowed
    modified["design_space"] = design_space
    return modified


def _remove_targets(query_data: dict[str, Any], names: list[str]) -> dict[str, Any]:
    modified = copy.deepcopy(query_data)
    lowered = {name.lower() for name in names}

    targets = modified.get("targets")
    if isinstance(targets, dict):
        modified["targets"] = {
            key: value for key, value in targets.items() if key.lower() not in lowered
        }
    errors = modified.get("prediction_errors")
    if isinstance(errors, dict):
        modified["prediction_errors"] = {
            key: value for key, value in errors.items() if key.lower() not in lowered
        }
    for section in ["preferences", "predicted_targets", "validated_targets"]:
        entries = modified.get(section)
        if isinstance(entries, list):
            modified[section] = [
                entry
                for entry in entries
                if not (
                    isinstance(entry, dict)
                    and str(entry.get("target") or "").lower() in lowered
                )
            ]
    return modified


def _set_age(query_data: dict[str, Any], age_days: float) -> dict[str, Any]:
    modified = copy.deepcopy(query_data)
    if "design_space" in modified or "age_days" not in modified:
        design_space = dict(modified.get("design_space") or {})
        design_space["age_days"] = age_days
        modified["design_space"] = design_space
    else:
        modified["age_days"] = age_days
    return modified


def _clear_explicit_systems(query_data: dict[str, Any]) -> dict[str, Any]:
    modified = copy.deepcopy(query_data)
    for key in ["material_system", "material_systems"]:
        modified.pop(key, None)
        design_space = modified.get("design_space")
        if isinstance(design_space, dict):
            design_space.pop(key, None)
    return modified


def propose_relaxations(
    query_data: dict[str, Any],
    diagnosis: dict[str, Any],
    *,
    max_proposals: int = 5,
) -> list[dict[str, Any]]:
    """Rank deterministic recovery proposals for an infeasible design query.

    Each proposal carries the fully modified query (and/or run-option changes)
    plus a human-readable rationale, so the change applied is always explicit.
    """
    proposals: list[dict[str, Any]] = []
    seen_material_sets: set[tuple[str, ...]] = set()
    caution_proposed = False
    missing_targets_proposed = False

    if diagnosis["candidate_count"] == 0:
        if diagnosis["explicit_material_systems"]:
            proposals.append(
                {
                    "kind": "clear_material_system_filter",
                    "description": (
                        "No registry model matches the explicitly requested material system(s) "
                        f"{diagnosis['explicit_material_systems']}; remove the explicit filter "
                        "and let the router choose."
                    ),
                    "query": _clear_explicit_systems(query_data),
                    "options": {},
                    "unblocks": [],
                }
            )
        ages = diagnosis.get("registry_age_options") or []
        requested_age = diagnosis.get("requested_age_days")
        if ages and requested_age is not None:
            nearest = min(ages, key=lambda age: abs(age - float(requested_age)))
            if abs(nearest - float(requested_age)) > 1.0e-9:
                proposals.append(
                    {
                        "kind": "use_nearest_supported_age",
                        "description": (
                            f"No registry model covers age {requested_age} d; "
                            f"the nearest supported age is {nearest} d."
                        ),
                        "query": _set_age(query_data, nearest),
                        "options": {},
                        "unblocks": [],
                    }
                )
        return proposals[:max_proposals]

    for row in diagnosis["candidates"]:
        if row["eligible"]:
            continue
        material_missing: list[str] = []
        missing_targets: list[str] = []
        caution_targets: list[str] = []
        for blocker in row["blockers"]:
            material_missing.extend(_parse_material_blocker(blocker))
            match = _MISSING_TARGET_BLOCKER.search(blocker)
            if match:
                missing_targets.append(match.group("target"))
            match = _TARGET_STATUS_BLOCKER.search(blocker)
            if match and match.group("status") == "usable_with_caution":
                caution_targets.append(match.group("target"))

        if material_missing and not missing_targets:
            key = tuple(sorted(material_missing))
            if key not in seen_material_sets:
                seen_material_sets.add(key)
                proposals.append(
                    {
                        "kind": "allow_additional_materials",
                        "description": (
                            f"Allow material(s) {sorted(material_missing)} so the "
                            f"'{row['material_system']}' model (score {row['score']}) becomes eligible."
                        ),
                        "query": _add_allowed_materials(query_data, sorted(material_missing)),
                        "options": {},
                        "unblocks": [row.get("id")],
                    }
                )
        if caution_targets and not caution_proposed and diagnosis["target_policy"] == "recommended":
            caution_proposed = True
            proposals.append(
                {
                    "kind": "allow_caution_targets",
                    "description": (
                        f"Target(s) {sorted(set(caution_targets))} are usable with caution; "
                        "relax the routing target policy from 'recommended' to 'allow_caution'."
                    ),
                    "query": copy.deepcopy(query_data),
                    "options": {"route_target_policy": "allow_caution"},
                    "unblocks": [row.get("id")],
                }
            )
        if missing_targets and not missing_targets_proposed:
            missing_targets_proposed = True
            unique = sorted(set(missing_targets))
            proposals.append(
                {
                    "kind": "drop_unsupported_targets",
                    "description": (
                        f"Target(s) {unique} are not available in any candidate model's "
                        "diagnostics; drop them from the query (they can be checked later "
                        "with real xGEMS validation)."
                    ),
                    "query": _remove_targets(query_data, unique),
                    "options": {},
                    "unblocks": [row.get("id")],
                }
            )
        if len(proposals) >= max_proposals:
            break
    return proposals[:max_proposals]


def run_design_with_recovery(
    design_query: dict[str, Any],
    *,
    out: str | Path,
    db: str | Path,
    use_mock: bool = True,
    skip_validation: bool = True,
    max_attempts: int = 3,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    target_policy: str = "recommended",
    dat_lst: str | Path | None = None,
) -> dict[str, Any]:
    """Bounded observe-replan loop for scenario (a).

    Diagnose -> (if infeasible) apply the top-ranked relaxation -> repeat, then
    execute the first feasible query. Every applied change is recorded in the
    returned ``attempts`` log and in ``recovery_log.json`` under ``out``.
    """
    from inverse_gems.api import run_request

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    query = copy.deepcopy(design_query)
    options: dict[str, Any] = {"route_target_policy": target_policy}
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] = {"status": "failed", "reason": "max_attempts_exhausted"}

    for attempt in range(max_attempts):
        diagnosis = diagnose_design_query(
            query,
            model_registry=model_registry,
            material_systems_config=material_systems_config,
            target_policy=options["route_target_policy"],
        )
        record: dict[str, Any] = {
            "attempt": attempt,
            "feasible": diagnosis["feasible"],
            "eligible_count": diagnosis["eligible_count"],
            "blocker_histogram": diagnosis["blocker_histogram"],
        }
        if diagnosis["feasible"]:
            record["selected_id"] = diagnosis["selected_id"]
            run_out = out_dir / f"attempt_{attempt}"
            task_query = {
                "name": str(design_query.get("name") or "design_recovery"),
                "task_type": "inverse_design",
                "design_query": query,
            }
            result = run_request(
                task_query=_write_task_query(task_query, run_out),
                out=run_out,
                db=db,
                use_mock=use_mock,
                skip_validation=skip_validation,
                model_registry=model_registry,
                material_systems_config=material_systems_config,
                route_target_policy=options["route_target_policy"],
                dat_lst=dat_lst,
                disable_plots=True,
            )
            record["run_status"] = result.status
            record["run_dir"] = str(run_out)
            attempts.append(record)
            final = {
                "status": result.status,
                "run_dir": str(run_out),
                "selected_id": diagnosis["selected_id"],
                "files": result.files,
                "result_summary": result.summary,
            }
            break

        proposals = propose_relaxations(query, diagnosis)
        if not proposals:
            record["applied_proposal"] = None
            attempts.append(record)
            final = {"status": "failed", "reason": "infeasible_no_proposals", "diagnosis": diagnosis}
            break
        applied = proposals[0]
        record["applied_proposal"] = {
            "kind": applied["kind"],
            "description": applied["description"],
            "options": applied["options"],
            "alternatives_considered": [p["kind"] for p in proposals[1:]],
        }
        attempts.append(record)
        query = applied["query"]
        options.update(applied["options"])

    log = {
        "original_query": design_query,
        "final_query": query,
        "final_options": options,
        "attempts": attempts,
        "final": {key: value for key, value in final.items() if key != "result_summary"},
    }
    write_json(out_dir / "recovery_log.json", log)
    return {
        "status": final.get("status", "failed"),
        "reason": final.get("reason"),
        "attempts": attempts,
        "changes_applied": [
            record["applied_proposal"]
            for record in attempts
            if record.get("applied_proposal")
        ],
        "final_query": query,
        "final_options": options,
        "final": final,
        "recovery_log": str(out_dir / "recovery_log.json"),
    }


def _write_task_query(task_query: dict[str, Any], out_dir: Path) -> Path:
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "task_query.yaml"
    path.write_text(yaml.safe_dump(task_query, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
