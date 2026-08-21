"""Precompute kernel ground-truth targets for numeric agent-QA scenarios.

Reads configs/qa_targets.yaml, runs each item through the kernel
(agent_tools.run_forward against a scratch DB), extracts the requested
scalar from response_summary.csv, and writes a ready-to-run agent_qa
scenario file with the frozen target plus provenance.

Mock items run in any environment. Items marked ``grounding: real`` are
computed only with --grounding real, which must be invoked inside the
py313-xgems conda env (real xGEMS execution); they are skipped otherwise.

Usage (from the GemsPilot repo root):
    python scripts/ground_truth.py [--grounding mock|real]
        [--out configs/agent_qa_generated.yaml] [--kernel-root PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

QUANTITY_PHRASES = {
    "porosity": "the porosity",
    "pH": "the pore-solution pH",
    "system_volume": "the total system volume",
}
EXTRACT_HINTS = {"porosity": "porosity", "pH": "pH", "system_volume": "volume"}

# Phase quantities: quantity "phase_mass:<RawName>" reads the
# phase_mass__<RawName> column of time_series.csv. display/extract control
# the task wording and the grading label.
PHASE_DISPLAY = {
    "Portlandite": ("the mass of the Portlandite phase", "Portlandite"),
    "ettringite": ("the mass of the ettringite phase", "ettringite"),
    "CNASH": ("the mass of the C-A-S-H phase (kernel phase name CNASH)", "CNASH"),
    "OH-hydrotalcite": ("the mass of the OH-hydrotalcite phase", "hydrotalcite"),
    "C4AcH11": ("the mass of the monocarbonate phase (kernel phase name C4AcH11)", "C4AcH11"),
}


def _recipe_phrase(recipe: dict) -> str:
    binders = ", ".join(
        f"{name.replace('_', ' ')} {amount}" for name, amount in recipe["binders"].items()
    )
    return f"{binders}, w/b {recipe['w_b']}"


def _forward_query(item: dict) -> str:
    recipe = item["recipe"]
    binders = ", ".join(f"{k}: {v}" for k, v in recipe["binders"].items())
    ages = ", ".join(str(a) for a in item["ages"])
    return (
        f"name: gt_{item['id']}\n"
        "task: forward_calculation\n"
        f"recipe: {{binders: {{{binders}}}, w_b: {recipe['w_b']}}}\n"
        f"age_grid: {{values: [{ages}]}}\n"
        "outputs: {phase_masses: all, phase_volumes: all, "
        "phase_volumes_reconstructed: all, aqueous_species: all, scalars: all}\n"
        "plots: []\n"
        + (
            f"response_summary: {{phases: [{item['quantity'].split(':', 1)[1]}], scalars: [porosity]}}\n"
            if item["quantity"].startswith("phase_mass:")
            else f"response_summary: {{scalars: [{item['quantity']}]}}\n"
        )
    )


def _task_text(item: dict, grounding: str) -> str:
    if item["quantity"].startswith("phase_mass:"):
        phase = item["quantity"].split(":", 1)[1]
        quantity = PHASE_DISPLAY[phase][0] + " as reported in the run artifacts"
    else:
        quantity = QUANTITY_PHRASES.get(item["quantity"], item["quantity"])
    if grounding == "real":
        mode = (
            "a forward calculation with real xGEMS execution (allowed for this "
            'task; pass use_mock=false and dat_lst="${INVERSE_GEMS_ROOT}/Test-dat.lst")'
        )
    else:
        mode = "a mock forward calculation"
    ask_age = item.get("ask_age")
    if ask_age is not None and len(item["ages"]) > 1:
        ages = ", ".join(f"{a:g}" for a in item["ages"])
        return (
            f"Run {mode} for a paste of {_recipe_phrase(item['recipe'])} "
            f"over ages {ages} days and report {quantity} at age {ask_age:g} days."
        )
    age = item["ages"][0]
    return (
        f"Run {mode} for a paste of {_recipe_phrase(item['recipe'])} "
        f"at age {age:g} days and report {quantity}."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(REPO_ROOT / "configs" / "qa_targets.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "configs" / "agent_qa_generated.yaml"))
    parser.add_argument("--work", default=str(REPO_ROOT / "runs" / "ground_truth"))
    parser.add_argument("--grounding", choices=["mock", "real"], default="mock")
    parser.add_argument(
        "--dat-lst", default=None,
        help="GEMS input file list for real runs (default: <kernel-root>/Test-dat.lst).",
    )
    parser.add_argument(
        "--kernel-root",
        default=os.environ.get("INVERSE_GEMS_ROOT", r"C:\Users\solmo\InverseGems v2"),
    )
    args = parser.parse_args()

    kernel_root = Path(args.kernel_root).resolve()
    os.environ["INVERSE_GEMS_ROOT"] = str(kernel_root)
    work = Path(args.work).resolve()
    os.environ["INVERSE_GEMS_ARTIFACT_ROOTS"] = (
        os.environ.get("INVERSE_GEMS_ARTIFACT_ROOTS", "") + os.pathsep + str(work)
    ).strip(os.pathsep)
    os.chdir(kernel_root)

    from gemspilot import agent_tools

    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    scenarios = []
    skipped = []
    for item in spec["items"]:
        grounding = str(item.get("grounding", "mock"))
        if grounding != args.grounding:
            skipped.append((item["id"], f"grounding={grounding}"))
            continue
        use_mock = grounding == "mock"
        out_dir = work / item["id"]
        kwargs: dict = {"use_mock": use_mock}
        if not use_mock:
            kwargs["dat_lst"] = args.dat_lst or str(kernel_root / "Test-dat.lst")
        result = agent_tools.run_forward(
            _forward_query(item), str(out_dir / "run"), str(out_dir / "db"), **kwargs
        )
        if not result["ok"] or result["summary"].get("status") != "complete":
            reason = result["summary"].get("status") or result.get("error")
            skipped.append((item["id"], f"kernel run failed: {reason}"))
            print(f"[fail] {item['id']}: {reason}")
            continue
        if item["quantity"].startswith("phase_mass:"):
            phase = item["quantity"].split(":", 1)[1]
            frame = pd.read_csv(Path(result["artifacts"]["forward_dir"]) / "time_series.csv")
            column = f"phase_mass__{phase}"
        else:
            summary_csv = Path(result["summary"]["result_summary"]["response_summary"]["out"]).parent \
                / "forward" / "response_summary.csv"
            if not summary_csv.exists():
                summary_csv = Path(result["summary"]["result_summary"]["csv"])
            frame = pd.read_csv(summary_csv)
            column = f"scalar__{item['quantity']}"
        if column not in frame.columns:
            skipped.append((item["id"], f"column {column} missing"))
            print(f"[fail] {item['id']}: {column} not in columns")
            continue
        ask_age = item.get("ask_age", item["ages"][0])
        row = frame.loc[(frame["age_days"] - float(ask_age)).abs() < 1e-9]
        if row.empty:
            skipped.append((item["id"], f"age {ask_age} not in results"))
            continue
        target = float(row.iloc[0][column])
        recipe_id = str(row.iloc[0].get("recipe_id", ""))
        scenario = {
            "id": item["id"],
            "kind": "agent_qa",
            "family": item["family"],
            "task": _task_text(item, grounding),
            "grading": (
                # pH is logarithmic: grade with an absolute tolerance.
                {
                    "answer_kind": "numeric",
                    "target": round(target, 6),
                    "abs_tol": 0.1,
                    "extract": "pH",
                }
                if item["quantity"] == "pH"
                else {
                    "answer_kind": "numeric",
                    "target": round(target, 6),
                    **(
                        {"abs_tol": float(item["abs_tol"])}
                        if item.get("abs_tol") is not None
                        else {"rel_tol": float(item.get("rel_tol", 0.02))}
                    ),
                    "extract": (
                        PHASE_DISPLAY[item["quantity"].split(":", 1)[1]][1]
                        if item["quantity"].startswith("phase_mass:")
                        else EXTRACT_HINTS.get(item["quantity"], item["quantity"])
                    ),
                }
            ),
            "constraints": {"max_tool_calls": 8, "min_tool_calls": 2},
            "allow_real": grounding == "real",
            "provenance": {
                "grounding": grounding,
                "recipe_id": recipe_id,
                "quantity": item["quantity"],
                "ask_age_days": float(ask_age),
            },
        }
        scenarios.append(scenario)
        print(f"[ok  ] {item['id']}: {item['quantity']} = {target:.6g}")

    out_path = Path(args.out)
    existing = []
    if out_path.exists():
        existing_data = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
        new_ids = {s["id"] for s in scenarios}
        existing = [s for s in (existing_data.get("scenarios") or []) if s["id"] not in new_ids]
    payload = {"scenarios": existing + scenarios}
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(f"\n{len(scenarios)} scenario(s) written -> {out_path}")
    for item_id, reason in skipped:
        print(f"  skipped {item_id}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
