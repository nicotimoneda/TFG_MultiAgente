# Orquestación de Equipos de Agentes LLM con Roles Especializados para Resolución Colaborativa de Tareas Complejas de Ingeniería de Software

Trabajo Fin de Grado — Grado en Computación e IA — Universidad Alfonso X el Sabio. Curso 2025–2026.

**Stack:** Python · LangGraph · LangChain · Ollama · Qwen 2.5 Coder 7B Instruct.

## Resumen

Comparativa empírica controlada de tres configuraciones LangGraph para generación automática de código Python sobre HumanEval (164 problemas):

1. `baseline` — un único LLM monolítico produce el código directamente.
2. `sequential` — pipeline de cinco roles especializados (Product Manager → Arquitecto → Developer → QA Tester → Code Reviewer) con estado compartido tipado y comunicación por artefactos.
3. `self_reflection` — pipeline secuencial con bucle iterativo Reviewer → Developer (`max_revisions = 1` en la corrida principal).

Tres variantes de ablación (`no_pm`, `no_architect`, `no_reviewer`) quedan implementadas y disponibles en el runner para una segunda pasada.

## Resultado principal

El experimento contrasta empíricamente la hipótesis intuitiva de que la especialización por roles mejora la corrección. **No la sostiene.** Sobre HumanEval con Qwen 2.5 Coder 7B Q4_K_M servido en local:

| Configuración | n | pass@1 | IC 95 % | Tokens medios | Latencia (s) |
|---|---:|---:|---|---:|---:|
| Baseline | 492 | **80,08 %** | [76,42 ; 83,54] | 283 | 5,1 |
| Sequential | 492 | 58,33 % | [53,86 ; 62,60] | 11 614 | 396,4 |
| SR (r=1) | 492 | 64,84 % | [60,57 ; 69,31] | 14 135 | 411,6 |

McNemar pareado: Baseline vs Sequential p < 0,0001 (b=128, c=21); Baseline vs SR_r1 p < 0,0001 (b=101, c=26); Sequential vs SR_r1 p = 0,0066 (b=49, c=81). H2 queda respaldada para `r = 1`.

Las configuraciones multi-agente cuestan ≈40 × más tokens y ≈77 × más latencia para producir peor pass@1; la frontera de Pareto coste-calidad la ocupa por completo el baseline. La memoria discute tres causas plausibles (propagación de errores entre roles, sobrecarga del prompt de rol en un modelo de 7 B, HumanEval inadecuado para evaluar pipelines multi-agente) y posiciona el resultado dentro de una línea crítica reciente (Chen et al., 2024; Olausson et al., 2024; Huang et al., 2024). Detalle completo en los capítulos 7 y 8 de la memoria.

## Estructura del repositorio

```
TFG_MultiAgente/
├── doc/                       Documentación escrita del TFG
│   ├── capitulos/             Capítulos en Markdown
│   ├── tables/                Tablas autogeneradas por el pipeline de análisis
│   ├── referencias/           Bibliografía global (BibTeX + Markdown)
│   ├── entregas/              Entregables (resumen ejecutivo, entregas previas)
│   └── decisiones.md          Documento canónico de decisiones técnicas
├── src/                       Código fuente del sistema
│   ├── state/                 AgentState compartido (TypedDict)
│   ├── agents/                Agente base, baseline y los 5 agentes de rol
│   ├── graph/                 Grafos LangGraph + variantes de ablación
│   ├── llm/                   Factory de clientes LLM (Ollama / Cerebras)
│   └── evaluation/            Sandbox, métricas y cargadores de benchmarks
├── experiments/               Runner, análisis y telemetría
│   ├── run_experiments.py     Runner principal (resumible, atómico)
│   ├── analyze_results.py     Post-procesador → figuras + tablas
│   ├── adherence_metric.py    Métrica de adherencia estructural
│   ├── dashboard.py           Dashboard en tiempo real para la corrida
│   ├── quick_check.py         Smoke test corto
│   ├── cache/                 Cache local de benchmarks (HumanEval, MBPP)
│   └── results/               CSVs por configuración y progress.json
├── figures/                   Figuras generadas + diagramas de los grafos
├── scripts/                   Scripts auxiliares (build memoria, watchdog)
└── tests/                     Tests unitarios e integración
```

## Configuración del entorno

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments]"
```

Backend de inferencia (Ollama local, opción canónica del TFG):

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama serve
```

El backend Cerebras se evaluó durante la planificación y se descartó por incompatibilidad del rate limit del tier público con la cardinalidad de la matriz. El factory de clientes sigue soportándolo para subsets puntuales con plan de pago (`export CEREBRAS_API_KEY=...`, `LLM_BACKEND=cerebras`).

## Reproducir el experimento

Smoke test (~15 min, 10 problemas × 5 configs):

```bash
python experiments/quick_check.py
```

**Corrida canónica del TFG** (≈9 días en MacBook Air M2, resumible):

```bash
LLM_BACKEND=ollama \
MODEL=qwen2.5-coder:7b-instruct-q4_K_M \
bash scripts/experiment_watchdog.sh
```

El watchdog ejecuta las tres configuraciones del barrido principal (`baseline`, `sequential`, `self_reflection_r1`) sobre HumanEval con tres semillas. Si la corrida se interrumpe, reejecutar el mismo comando: las ejecuciones completadas se omiten gracias a la propiedad de resumibilidad del runner.

**Matriz completa** (8 configs × 2 benchmarks × 3 semillas, no ejecutada en la corrida principal por presupuesto de cómputo — entrada S8 del anexo de decisiones):

```bash
LLM_BACKEND=ollama python experiments/run_experiments.py \
    --model qwen2.5-coder:7b-instruct-q4_K_M \
    --benchmarks humaneval,mbpp \
    --configs baseline,sequential,self_reflection_r1,self_reflection_r2,self_reflection_r3,ablation_no_pm,ablation_no_architect,ablation_no_reviewer
```

Dashboard de seguimiento en tiempo real (terminal aparte):

```bash
python experiments/dashboard.py
```

## Reproducir el análisis

```bash
python experiments/analyze_results.py    # figuras + tablas + narrativa
python experiments/adherence_metric.py   # adherencia estructural
```

Productos del análisis:

- `figures/pass_at_1.png` — pass@1 por configuración con IC 95 % bootstrap.
- `figures/cost_quality_pareto.png` — tokens vs. pass@1 (escala log).
- `figures/latency_box.png` — distribución de latencias por configuración.
- `figures/revision_distribution.png` — uso del ciclo en self-reflection.
- `doc/tables/summary.md` (y `.tex`) — tabla principal de resultados.
- `doc/tables/pairwise_mcnemar.md` — contrastes pareados de McNemar.
- `doc/tables/per_benchmark.md` — cross-tab configuración × benchmark.
- `doc/tables/problem_difficulty.md` — top-20 problemas con mayor desacuerdo.
- `doc/tables/adherence.md` — adherencia estructural por configuración.
- `doc/tables/findings_narrative.md` — borrador automático de hallazgos.

## Generar la memoria

```bash
bash scripts/build_memoria.sh
```

Produce `build/2526_TFG_GCIA_NP147254_Memoria.docx` (y `.pdf` si hay xelatex instalado) concatenando los 17 ficheros Markdown en orden canónico (resumen, declaración de IA, caps 1-8, anexos A-F, bibliografía).

## Documentación

- [Capítulos de la memoria](doc/capitulos/) — del resumen a la reflexión final.
- [Bibliografía global](doc/referencias/bibliografia.md) (BibTeX en `referencias.bib`).
- [Anexo de decisiones técnicas (documento canónico)](doc/decisiones.md) — sprint a sprint, con alternativas evaluadas y descartadas.
- [Resumen ejecutivo](doc/entregas/ResumenEjecutivo_EntregaFinal.md).

## Licencia

MIT. Véase `LICENSE`.
