"""GemsPilot command-line interface.

Commands:
    gemspilot bench --out DIR [--config configs/agent_bench.yaml] [--kernel-root PATH]
    gemspilot experiment --out DIR [--models configs/models.yaml]
        [--scenarios FILE ...] [--conditions tf tt nt] [--repeats N]
        [--only-models LABEL ...] [--only-items ID ...] [--kernel-root PATH]

``--kernel-root`` points at an InverseGems working tree that holds the
chemistry database and model registries. It is exported as
``INVERSE_GEMS_ROOT`` (used by ``${INVERSE_GEMS_ROOT}`` references in
scenario files) and used as the working directory during the run, so the
kernel's relative-path conventions keep working.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemspilot")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Run GEMS-Agent-Bench scenarios.")
    bench.add_argument("--config", default="configs/agent_bench.yaml")
    bench.add_argument("--out", required=True)
    bench.add_argument(
        "--kernel-root",
        default=os.environ.get("INVERSE_GEMS_ROOT"),
        help="InverseGems working tree with data/ and configs/ (default: $INVERSE_GEMS_ROOT).",
    )

    experiment = sub.add_parser(
        "experiment", help="Run the model x condition x scenario matrix."
    )
    experiment.add_argument("--models", default="configs/models.yaml")
    experiment.add_argument(
        "--scenarios",
        nargs="+",
        default=["configs/agent_qa_generated.yaml", "configs/agent_qa_manual.yaml"],
    )
    experiment.add_argument("--out", required=True)
    experiment.add_argument("--conditions", nargs="+", default=None,
                            help="Condition codes: tf (tools+full), tt (tools+toc), nt (no tools).")
    experiment.add_argument("--repeats", type=int, default=1)
    experiment.add_argument("--only-models", nargs="*", default=None)
    experiment.add_argument("--only-items", nargs="*", default=None)
    experiment.add_argument("--max-steps", type=int, default=12)
    experiment.add_argument(
        "--kernel-root",
        default=os.environ.get("INVERSE_GEMS_ROOT"),
        help="InverseGems working tree with data/ and configs/ (default: $INVERSE_GEMS_ROOT).",
    )
    return parser


def _enter_kernel_root(kernel_root: str | None, out: Path) -> None:
    """Export INVERSE_GEMS_ROOT, chdir to it, and whitelist the out tree.

    Also exports GEMSPILOT_ROOT (for ${GEMSPILOT_ROOT} references in
    scenario files) and whitelists the repo's fixtures directory so tools
    like calibrate_scm_kinetics can read bundled input data.
    """
    repo_root = Path.cwd().resolve()
    os.environ.setdefault("GEMSPILOT_ROOT", str(repo_root))
    fixtures = Path(os.environ["GEMSPILOT_ROOT"]) / "configs" / "fixtures"
    if kernel_root:
        root = Path(kernel_root).resolve()
        os.environ["INVERSE_GEMS_ROOT"] = str(root)
        os.chdir(root)
    roots = [
        p for p in os.environ.get("INVERSE_GEMS_ARTIFACT_ROOTS", "").split(os.pathsep) if p
    ]
    for extra in [str(out), str(fixtures)]:
        if extra not in roots:
            roots.append(extra)
    os.environ["INVERSE_GEMS_ARTIFACT_ROOTS"] = os.pathsep.join(roots)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bench":
        from .agent_bench import run_agent_bench

        config = Path(args.config).resolve()
        out = Path(args.out).resolve()
        _enter_kernel_root(args.kernel_root, out)
        summary = run_agent_bench(config, out=out)
        print(
            json.dumps(
                {
                    "total": summary["total"],
                    "passed": summary["passed"],
                    "failed": summary["failed"],
                    "skipped": summary["skipped"],
                    "report": str(out / "bench_report.json"),
                },
                indent=2,
            )
        )
        return 0 if summary["failed"] == 0 else 1
    if args.command == "experiment":
        from .experiment import run_experiment
        from .runner import load_env_file

        # Resolve inputs and credentials before entering the kernel root.
        load_env_file(Path.cwd() / ".env")
        models = Path(args.models).resolve()
        scenarios = [Path(p).resolve() for p in args.scenarios]
        out = Path(args.out).resolve()
        _enter_kernel_root(args.kernel_root, out)
        summary = run_experiment(
            models,
            scenarios,
            out,
            repeats=args.repeats,
            conditions=args.conditions,
            only_models=args.only_models,
            only_items=args.only_items,
            max_steps=args.max_steps,
        )
        print(json.dumps(summary, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
