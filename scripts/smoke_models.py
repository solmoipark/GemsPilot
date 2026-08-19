"""Smoke-test every model in configs/models.yaml with one tool-grounded episode.

Each model gets the validated forward-lookup task (mock OPC 60 / slag 40 /
w/b 0.45 / 28 d) and is graded against the kernel ground truth
(porosity 0.5997, pH 12.6). This verifies, per model: tool-calling works
through OpenRouter/litellm, the approval policy and workspace remap hold,
and the final answer carries kernel numbers rather than priors.

Usage (from the GemsPilot repo root):
    python scripts/smoke_models.py [--only LABEL ...] [--out DIR]
        [--kernel-root PATH] [--max-steps N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from gemspilot.runner import Episode, load_env_file, run_episode  # noqa: E402

TASK = (
    "Run a mock forward calculation for a paste of OPC 60, slag 40, w/b 0.45 "
    "at age 28 days and report the porosity and the pore-solution pH."
)
TARGETS = {"porosity": 0.5997, "pH": 12.6}
REL_TOL = 0.02

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _matches(text: str, target: float) -> bool:
    for token in _NUMBER.findall(text or ""):
        try:
            value = float(token)
        except ValueError:
            continue
        if abs(value - target) <= REL_TOL * abs(target):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "models.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "smoke"))
    parser.add_argument(
        "--kernel-root",
        default=os.environ.get("INVERSE_GEMS_ROOT", r"C:\Users\solmo\InverseGems v2"),
    )
    parser.add_argument("--only", nargs="*", default=None, help="labels to run")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    load_env_file(REPO_ROOT / ".env")
    kernel_root = Path(args.kernel_root).resolve()
    os.environ["INVERSE_GEMS_ROOT"] = str(kernel_root)
    os.chdir(kernel_root)

    roster = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(args.out).resolve()
    results = []
    for entry in roster["models"]:
        label = entry["label"]
        if args.only and label not in args.only:
            continue
        key_env = entry.get("api_key_env") or roster.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            results.append({"label": label, "status": "skipped", "reason": f"{key_env} unset"})
            print(f"[skip] {label}: {key_env} unset")
            continue
        workspace = out_root / label
        episode = Episode(
            model=entry["id"],
            workspace=workspace,
            max_steps=args.max_steps,
            completion_params=dict(entry.get("params") or {}),
        )
        print(f"[run ] {label} ({entry['id']}) ...", flush=True)
        try:
            outcome = run_episode(TASK, episode)
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            results.append({"label": label, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[err ] {label}: {type(exc).__name__}: {exc}")
            continue
        final = outcome.get("final_text") or ""
        grades = {name: _matches(final, target) for name, target in TARGETS.items()}
        ok = all(grades.values()) and outcome.get("stop_reason") == "final_answer"
        results.append({
            "label": label,
            "model": entry["id"],
            "status": "pass" if ok else "fail",
            "grades": grades,
            "stop_reason": outcome.get("stop_reason"),
            "steps": outcome.get("steps"),
            "tool_calls": len(outcome.get("tool_calls") or []),
            "providers": outcome.get("providers"),
            "cost_usd": round(outcome.get("usage", {}).get("cost_usd", 0.0), 5),
            "final_text": final[:400],
        })
        print(
            f"[{'pass' if ok else 'FAIL'}] {label}: steps={outcome.get('steps')} "
            f"cost=${results[-1]['cost_usd']} grades={grades} "
            f"providers={outcome.get('providers')}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "smoke_report.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    passed = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{passed}/{len(results)} passed -> {report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
