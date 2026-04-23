# TFG — Multi-Agent LLM Orchestration for Automated Code Generation

## Project summary

This repository contains the implementation and evaluation harness for a Bachelor's
thesis (TFG) comparing three LangGraph-based configurations for automated Python code
generation, evaluated on HumanEval and MBPP.

## Configurations

| # | Name | Description |
|---|------|-------------|
| 1 | `baseline` | Single-node monolith — one LLM call produces the final code |
| 2 | `sequential` | Five-role pipeline: PM → Architect → Developer → Tester → Reviewer |
| 3 | `self_reflection` | Sequential pipeline with an iterative self-correction loop |

## Stack

- **Python 3.11+**
- **LangGraph** — graph orchestration
- **LangChain / langchain-groq** — LLM interface
- **Groq API** — inference backend (Llama 3.3 70B, Qwen 2.5 Coder)
- **HuggingFace datasets** — benchmark loading
- **scipy / numpy / pandas** — metrics and analysis

## Repository layout

```
TFG_MultiAgente/
├── src/
│   ├── agents/
│   │   ├── base_agent.py               # Abstract base with _call_llm + retry
│   │   ├── baseline_agent.py           # Config 1: monolithic solver
│   │   └── roles/                      # Config 2 & 3 role agents
│   │       ├── product_manager.py      # PM: problem → PRD
│   │       ├── architect.py            # Architect: PRD → design_doc
│   │       ├── developer.py            # Developer: design_doc → code_artifact
│   │       ├── qa_tester.py            # QA: sandbox executor (no LLM)
│   │       └── code_reviewer.py        # Reviewer: verdict + structured review
│   ├── evaluation/
│   │   ├── humaneval_loader.py         # HumanEval loader + local cache
│   │   ├── sandbox.py                  # Subprocess-isolated code execution
│   │   ├── metrics.py                  # pass@k, avg_test_pass_rate, compute_all_metrics
│   │   └── runner.py                   # Full evaluation loop → CSV
│   ├── graph/
│   │   ├── baseline_graph.py           # build_baseline_graph, run_baseline
│   │   └── sequential_graph.py         # build_sequential_graph, run_sequential
│   ├── state/
│   │   └── schema.py                   # AgentState TypedDict (locked)
│   └── tools/                          # (reserved for S3 tool integrations)
├── experiments/
│   └── cache/                          # Cached benchmark JSON files
├── figures/                            # Generated plots (not committed)
├── doc/                                # Thesis document
├── tests/                              # pytest test suite
├── pyproject.toml
└── CONTEXT.md                          # This file
```

## Key design decisions

- **Shared state**: `AgentState` (TypedDict) flows through every LangGraph node; token
  counts and latency accumulate in-place so telemetry is always available at graph exit.
- **Sandbox isolation**: generated code runs in a subprocess (not `exec` in the main
  process) with builtins restricted to prevent filesystem/network side-effects.
- **Reproducibility**: every evaluation run is seeded with `random.seed(seed)` before
  graph invocation; seeds are logged in the output CSV.
- **pass@k estimator**: uses the unbiased formula from Chen et al. (2021),
  computed in log-space to avoid overflow for large n.
- **QA agent**: deterministic — runs the sandbox, never calls the LLM, so it adds
  zero token cost and is fast.
- **Reviewer verdict**: parser raises `ValueError` if the first non-empty line is not
  `VERDICT: APPROVE` or `VERDICT: REQUEST_CHANGES`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq inference API key (required) |

## Running an evaluation

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install / update dependencies
pip install -e .

# Quick smoke test — sequential pipeline, 1 problem
python - <<'EOF'
from src.evaluation.humaneval_loader import get_problem
from src.graph.sequential_graph import run_sequential
state = run_sequential(get_problem("HumanEval/1"), "llama-3.3-70b-versatile")
print(state["review_comments"].split("\n")[0])
EOF
```

## Sprint plan

| Sprint | Status   | Deliverable                                       |
|--------|----------|---------------------------------------------------|
| S1     | Done     | Shared state, baseline graph, evaluation harness  |
| S2     | Done     | Sequential multi-agent pipeline (5 roles)         |
| S3     | Next     | Self-reflection loop + comparative analysis       |

## S2 — What was delivered

- `src/agents/roles/product_manager.py` — PM agent: problem → PRD
- `src/agents/roles/architect.py` — Architect agent: PRD → design_doc
- `src/agents/roles/developer.py` — Developer agent: design_doc → code_artifact
- `src/agents/roles/qa_tester.py` — QA agent: sandbox execution only, no LLM
- `src/agents/roles/code_reviewer.py` — Reviewer agent: verdict parser + structured review
- `src/graph/sequential_graph.py` — `build_sequential_graph` + `run_sequential`
- `src/evaluation/runner.py` — sequential config wired in; QA results reused (no double sandbox)

## Next sprint (S3)

- `src/graph/self_reflection_graph.py` — add conditional edge reviewer → developer
  when verdict is `REQUEST_CHANGES`
- `max_revisions` parameter controlling the loop (test values: 1, 2, 3)
- Temperature 0.4 for the Developer agent in the self-reflection configuration
- Wire `config="self_reflection"` in `runner.py` (currently raises `NotImplementedError`)
