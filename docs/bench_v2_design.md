# GEMS-Agent-Bench v2 — Design Note

Goal: grow the 15-scenario behavioral suite into a publishable benchmark
(60–80 items) for LLM agents driving a deterministic thermodynamic-modeling
kernel, in the spirit of PHREEQC-MCQ-200 (arXiv:2607.00436), and run a
multi-model experiment matrix over it.

## Item kinds

1. **Behavioral scenarios** (existing kinds: diagnose, design_recovery,
   forward_mock, budget_guardrail, session_recall, propose, filter,
   past_runs, parse) — check *what the tools do* deterministically, without
   an LLM in the loop. These stay as harness regression items.
2. **Agent QA items** (new kind `agent_qa`) — a natural-language task given
   to `runner.run_episode`; graded on the agent's *final answer* against a
   kernel-derived unique target value, plus trajectory metrics.

## agent_qa schema (draft)

```yaml
- id: qa_forward_porosity_opc60_slag40_28d
  kind: agent_qa
  family: forward_lookup            # see families below
  task: >-
    Run a mock forward calculation for a paste of OPC 60, slag 40,
    w/b 0.45 at age 28 days and report the porosity.
  grading:
    answer_kind: numeric            # numeric | choice | refusal | behavior
    target: 0.5997                  # precomputed by the kernel (see grounding)
    rel_tol: 0.02
    extract: porosity               # regex/labelled-number extraction hint
  constraints:
    forbidden_tools: []             # e.g. run_forward with use_mock=false
    max_tool_calls: 6               # unnecessary-call metric threshold
  allow_real: false
  grounding: mock                   # mock | cached | real (real → precomputed in xgems env)
```

## Families (target counts, total ≈ 70)

| Family | n | Grading |
|---|---|---|
| forward lookup QA (porosity/pH/phase amounts, single + time series) | 18 | numeric |
| inverse feasibility & diagnosis (feasible/infeasible/selected model) | 10 | choice/behavior |
| constraint-relaxation recovery (must reach complete + right relaxation) | 8 | behavior |
| budget & approval compliance (must not exceed / must refuse real runs) | 8 | behavior |
| session memory & multi-turn refinement | 6 | behavior/numeric |
| calibration workflow (fit user CSV, report fitted D) | 6 | numeric |
| ambiguous / infeasible requests (correct answer = ask or refuse) | 8 | refusal |
| real-xGEMS grounded subset | 8 (~11%) | numeric |

Real-grounded targets are precomputed once in the `py313-xgems` env and
frozen into the scenario file with provenance (chemistry fingerprint id).

## Metrics per (item × condition × repeat)

- answer correctness (primary), with numeric tolerance per item
- tool-selection correctness (called the expected tool family at least once)
- unnecessary calls (# tool calls − minimal path), budget/approval violations
- steps, tokens, cost; stop_reason distribution
- failure-stage label (input construction / execution / output navigation /
  final-answer mapping), auto-derived from the trajectory where possible

Aggregations follow PHREEQC-MCQ-200: per-model accuracy, gain/loss/retention
vs the no-tool baseline, and the output-access ablation (protocol full vs toc).

## Conditions

- models: from `configs/models.yaml` (user-provided keys; tier labels)
- tools on vs **no-tool baseline** (same prompt, no tool schemas; answer from
  priors — expected to fail numeric items, quantifying grounding gains)
- output protocol: `full` vs `toc` (runner already implements both)
- ablation: router-only copilot (single parse→run, no replan) vs full agent
- repeats ≥ 3 (temperature 0; repeats capture provider nondeterminism)

## Implementation steps (next session)

1. `agent_bench.py`: add `agent_qa` kind → calls `runner.run_episode`,
   grades with a deterministic extractor (labelled-number regex + tolerance;
   choice matching; refusal detection by rubric keywords + no-tool-call check).
2. `scripts/ground_truth.py`: batch-precompute numeric targets (mock via
   default env; real via py313-xgems) and emit scenario YAML fragments.
3. `configs/models.yaml` + experiment driver `gemspilot experiment`
   (matrix loop, per-model cost caps, resumable by item id).
4. Analysis notebook/script → aggregate CSVs and paper figures.

## Cost estimate

Smoke episode (GPT-4.1-mini, 3 steps): ~$0.003. 70 items × 4 conditions ×
3 repeats ≈ 840 episodes ≈ $3–10 per mid-tier model; frontier models
dominate the budget — cap via per-model `max_cost_usd` in models.yaml.
