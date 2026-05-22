# Changelog

Sprint-by-sprint history of the TFG. Each sprint corresponds to a single
commit (or a small set of commits) on the `main` branch. See `git log` for
the exact SHAs and `doc/decisiones.md` for the rationale behind major
decisions.

## S10 — Autonomy, ClassEval real, defensa, pyright clean (2026-05-22)

- Watchdog script (`scripts/experiment_watchdog.sh`) keeps Ollama + runner
  alive and chains main → ablations → optional MBPP phases automatically.
- ClassEval real integration: `src/graph/classeval_graph.py` wired into the
  runner with new config `classeval_sequential` and benchmark `classeval`.
- Defensa outline (`doc/defensa_outline.md`): 14-slide structure plus 7
  anticipated tribunal questions with prepared responses.
- Pyright errors reduced from 28 to 0 across `src/`.
- Tests for the analysis pipeline (`tests/test_analyze_results.py`): 18
  additional pytest cases covering bootstrap CIs, McNemar (chi² and exact
  branches), aggregation, and row coercion edge cases.
- Auto-narrative generator (`doc/tables/findings_narrative.md`): Markdown
  summary of findings written automatically from the analysis pipeline.
- CHANGELOG.md (this file).
- Resumen ejecutivo converted to `.docx`.

## S9 — Production layer for the entrega final (2026-05-22)

- Anexos B (decisiones) and C (comandos de reproducción) in the memoria.
- Resumen ejecutivo in Markdown following the format of Entrega 1 and 2.
- `scripts/build_memoria.sh`: pandoc pipeline that assembles 8 chapters,
  3 anexos and the global bibliography into a single `.docx`.
- Class-aware Developer agent (`src/agents/roles/developer_classeval.py`).
- System architecture diagram (`figures/arquitectura_sistema.png`) and
  PNGs of the three ablation graphs.
- Resolved inconsistencies flagged during the resumen draft: CONTEXT.md
  S4 run count (9 100 → 2 460) and cap 8 baseline sample size (390 → 492).

## S8 — Memoria chapters 5 to 8 (2026-05-22)

- Chapter 5 — Desarrollo (343 lines).
- Chapter 6 — Experimentos (249 lines).
- Chapter 7 — Resultados refined with partial-data framing (273 lines).
- Chapter 8 — Conclusiones (192 lines).
- Anexo A — Prompts verbatim of every agent (250 lines).
- Bibliography centralised in `doc/referencias/bibliografia.md` (single
  global list per tutor's feedback).
- Chapter-level "Referencias" blocks removed from chapter 2 (7 sections).
- Capítulo 1 contributions section expanded to include S7 deliverables;
  section 1.4 references Anexo A.
- Smoke test of clean install in `/tmp` with Python 3.12: pip install,
  pytest, module imports — all green.

## S7 — Role ablations, analysis pipeline, adherence metric (2026-05-22)

- Three role-ablation graphs (`src/graph/ablation_graphs.py`):
  `no_pm`, `no_architect`, `no_reviewer`.
- Runner config registry extended; ablation configs resumable.
- MBPP cache pre-built (`experiments/cache/mbpp.json`, 200 problems).
- Analysis pipeline (`experiments/analyze_results.py`): pass@1 with 95%
  bootstrap CI, pass@3 (Chen 2021), Pareto, latency boxplot, revision
  distribution, problem-difficulty top-20, paired McNemar.
- Adherence metric (`experiments/adherence_metric.py`).
- New decisions entry in `doc/decisiones.md` with SWE-bench scope decision.

## S6 — Ollama backend + HumanEval prompt prepend (2026-05-13)

- Ollama as default backend (qwen2.5-coder:7b-instruct-q4_K_M).
- Reviewer verdict derived deterministically from `test_results`.
- HumanEval evaluation uses canonical `prompt + completion` concatenation.
- Mermaid graphs for the 3 main configurations rendered to `figures/`.

## S5 — Cerebras backend + evalplus harness fix (2026-04-30)

- Cerebras Inference API as alternative backend (qwen-3-235b).
- Sandbox harness updated for evalplus test format.
- Scope reduced where measurements required excessive compute.

## S4 — Experiment runner, dashboard, quick-check (2026-04-25)

- Resumable runner with atomic `progress.json`.
- Live Rich-based dashboard (`experiments/dashboard.py`).
- 10-problem smoke test (`experiments/quick_check.py`).
- Per-config CSVs in `experiments/results/`.

## S3 — Self-reflection loop + comparative analysis (2026-04-24)

- `src/graph/self_reflection_graph.py` with conditional edge
  reviewer → developer.
- `ReflectiveDeveloperAgent` operating at temperature 0.4.
- `max_revisions` ∈ {1, 2, 3} as the hyperparameter to sweep.

## S2 — Sequential multi-agent pipeline (2026-04-23)

- Five role agents implemented:
  - `src/agents/roles/product_manager.py`
  - `src/agents/roles/architect.py`
  - `src/agents/roles/developer.py`
  - `src/agents/roles/qa_tester.py` (sandbox-only, no LLM)
  - `src/agents/roles/code_reviewer.py`
- `src/graph/sequential_graph.py` linking the five roles.

## S1 — Shared state, baseline graph, evaluation harness (2026-04-22)

- `src/state/schema.py` with the `AgentState` TypedDict.
- `src/agents/baseline_agent.py` for the monolithic config.
- `src/graph/baseline_graph.py`.
- `src/evaluation/sandbox.py` with subprocess isolation.
- `src/evaluation/metrics.py` with the Chen et al. (2021) pass@k estimator.
- `src/evaluation/humaneval_loader.py`.

---

For decision rationale see `doc/decisiones.md`; for the chapter-level
view of the project's contributions see
`doc/capitulos/01_introduccion.md` and `doc/capitulos/08_conclusiones.md`.
