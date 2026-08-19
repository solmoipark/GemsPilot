"""GemsPilot command-line interface.

Commands:
    gemspilot bench --out DIR [--config configs/agent_bench.yaml] [--kernel-root PATH]

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bench":
        from .agent_bench import run_agent_bench

        config = Path(args.config).resolve()
        out = Path(args.out).resolve()
        if args.kernel_root:
            kernel_root = Path(args.kernel_root).resolve()
            os.environ["INVERSE_GEMS_ROOT"] = str(kernel_root)
            os.chdir(kernel_root)
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
