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
│   │   ├── base_agent.py          # Abstract base with _call_llm + retry
│   │   └── baseline_agent.py      # Config 1: monolithic solver
│   ├── evaluation/
│   │   ├── humaneval_loader.py    # HumanEval loader + local cache
│   │   ├── sandbox.py             # Subprocess-isolated code execution
│   │   ├── metrics.py             # pass@k, avg_test_pass_rate, compute_all_metrics
│   │   └── runner.py              # Full evaluation loop → CSV
│   ├── graph/
│   │   └── baseline_graph.py      # build_baseline_graph, run_baseline
│   ├── state/
│   │   └── schema.py              # AgentState TypedDict
│   └── tools/                     # (reserved for S2/S3 tool integrations)
├── experiments/
│   └── cache/                     # Cached benchmark JSON files
├── figures/                       # Generated plots (not committed)
├── doc/                           # Thesis document
├── tests/                         # pytest test suite
├── pyproject.toml
└── CONTEXT.md                     # This file
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

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq inference API key (required) |

## Running an evaluation

```bash
# Install dependencies
pip install -e .

# Quick smoke test (1 problem, 1 seed)
python -m src.evaluation.runner   # see runner.py for programmatic API

# Full HumanEval baseline run
python experiments/run_baseline.py   # to be added in S2
```

## Sprint plan

| Sprint | Deliverable |
|--------|------------|
| S1 (current) | Shared state, baseline graph, evaluation harness |
| S2 | Sequential multi-agent pipeline (5 roles) |
| S3 | Self-reflection loop + comparative analysis |
