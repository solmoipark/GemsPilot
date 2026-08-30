# GemsPilot

GemsPilot is the LLM agent layer for [InverseGems](https://github.com/solmoipark/InverseGems), a deterministic thermodynamic-modeling kernel for blended cementitious binders (xGEMS/GEMS). InverseGems computes; GemsPilot orchestrates.

The split mirrors the two-paper architecture:

- **InverseGems (kernel)** — reaction models, xGEMS equilibrium runs, the reactive-chemistry database, surrogates, calibration, sensitivity. Every scientific step is deterministic.
- **GemsPilot (agent layer, this repository)** — the framework-neutral tool layer over the kernel, an MCP server, observe–replan recovery loops, the autonomous coverage-growth campaign, session memory, guardrails, and **GEMS-Agent-Bench**: an end-to-end scenario benchmark for LLM agents that drive a deterministic scientific simulator, in the spirit of PHREEQC-MCQ-200 (arXiv:2607.00436).

The language model never produces a scientific conclusion. It selects and parameterizes tools; all computation, screening, ranking, and validation run in the deterministic kernel, under per-request solver-call budgets and approval gates.

## Components

- `gemspilot.agent_tools` — 18 tools with a standardized ToolResult contract (compact summary + artifact references)
- `gemspilot.mcp_server` — MCP server (`gemspilot-mcp`) exposing the tools to any agent host
- `gemspilot.design_recovery` / `solver recovery via kernel` — observe–replan loops with deterministic relaxation proposals
- `gemspilot.coverage_campaign` — bounded autonomous database-growth campaigns
- `gemspilot.agent_bench` — GEMS-Agent-Bench scenario harness and graders
- `gemspilot.cli` — `gemspilot bench` and related commands

## Installation

```bash
pip install -e .[test,llm]
```

Real xGEMS execution additionally requires an xGEMS-enabled Python environment and a GEMS system-definition file set.

GemsPilot ships no GEMS system-definition files of its own. The canonical set (`*-dat.lst`, `*-dch.json`, `*-ipm.json`, `*-fun.json`, `*-dbr*.json`) lives in the InverseGems kernel, alongside the chemistry database. Both are located through the `INVERSE_GEMS_ROOT` environment variable, and benchmark items refer to them as e.g. `${INVERSE_GEMS_ROOT}/Test-dat.lst`:

```bash
export INVERSE_GEMS_ROOT=/path/to/InverseGems
```

GemsPilot is built on xGEMS/GEMS3K and Cemdata via the InverseGems kernel. It is an independent project, not affiliated with or endorsed by the GEMS development team.

**License**: BSD 3-Clause (see [LICENSE](LICENSE)).
