# Orquestación de Equipos de Agentes LLM con Roles Especializados para Resolución Colaborativa de Tareas Complejas de Ingeniería de Software

Trabajo Fin de Grado — Computación e IA — Universidad Alfonso X el Sabio.

**Stack tecnológico:** Python · LangGraph · LangChain · Ollama · HuggingFace
datasets.

**Fecha de inicio:** 2026-04-22.

## Resumen

Comparativa empírica de tres configuraciones LangGraph para generación
automática de código Python evaluadas sobre HumanEval y MBPP:

1. `baseline` — un único LLM monolítico produce el código directamente.
2. `sequential` — pipeline de cinco roles (Product Manager → Arquitecto →
   Developer → QA Tester → Code Reviewer).
3. `self_reflection` — pipeline secuencial con bucle iterativo Reviewer →
   Developer hasta `max_revisions ∈ {1, 2, 3}`.

Se añaden tres variantes de ablación que suprimen un rol del pipeline
secuencial (`no_pm`, `no_architect`, `no_reviewer`) para cuantificar la
contribución individual de cada rol.

El detalle de la arquitectura, el banco experimental y los resultados se
recoge en la memoria bajo `doc/capitulos/`.

## Estructura del repositorio

```
TFG_MultiAgente/
├── doc/                       Documentación escrita del TFG
│   ├── capitulos/             Capítulos de la memoria en Markdown
│   ├── tables/                Tablas autogeneradas por el pipeline de análisis
│   ├── referencias/           Bibliografía global (BibTeX + Markdown)
│   ├── entregas/              PDFs de las entregas previas
│   └── decisiones.md          Anexo de decisiones técnicas por sprint
├── src/                       Código fuente del sistema
│   ├── state/                 AgentState compartido (TypedDict)
│   ├── agents/                Agente base, baseline y agentes de rol
│   ├── graph/                 Grafos LangGraph + variantes de ablación
│   ├── llm/                   Factory de clientes LLM (Ollama / Cerebras)
│   └── evaluation/            Sandbox, métricas y cargadores de benchmarks
├── experiments/               Runner, análisis y telemetría
│   ├── run_experiments.py     Runner principal (resumible, atómico)
│   ├── analyze_results.py     Post-procesador → figuras + tablas
│   ├── adherence_metric.py    Métrica de adherencia estructural
│   ├── dashboard.py           Dashboard en tiempo real para la corrida
│   ├── quick_check.py         Smoke test 10 problemas × 5 configs
│   ├── cache/                 Cache local de benchmarks (HumanEval, MBPP, …)
│   └── results/               CSVs por configuración y progress.json
├── figures/                   Figuras generadas + diagramas de los grafos
└── README.md
```

## Configuración del entorno

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments]"
```

Backend local (recomendado para reproducibilidad):

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama serve
```

Backend remoto (Cerebras, para verificaciones con un modelo mayor):

```bash
export CEREBRAS_API_KEY=...
```

## Reproducir el experimento

Smoke test (~15 min):

```bash
python experiments/quick_check.py
```

Corrida completa, resumible:

```bash
LLM_BACKEND=ollama python experiments/run_experiments.py \
    --model qwen2.5-coder:7b-instruct-q4_K_M \
    --benchmarks humaneval,mbpp \
    --configs baseline,sequential,self_reflection_r1,self_reflection_r2,self_reflection_r3,ablation_no_pm,ablation_no_architect,ablation_no_reviewer
```

Dashboard en tiempo real (terminal aparte):

```bash
python experiments/dashboard.py
```

Si se interrumpe la corrida, volver a ejecutar el mismo comando del runner:
las ejecuciones ya completadas se omiten y el proceso retoma donde quedó.

## Reproducir el análisis

```bash
python experiments/analyze_results.py    # figuras + tablas
python experiments/adherence_metric.py   # adherencia estructural
```

Productos del análisis:

- `figures/pass_at_1.png` — pass@1 por configuración con IC 95% bootstrap.
- `figures/cost_quality_pareto.png` — tokens vs. pass@1 (escala log).
- `figures/latency_box.png` — distribución de latencias por configuración.
- `figures/revision_distribution.png` — uso del ciclo en self-reflection.
- `doc/tables/summary.md` — tabla principal de resultados.
- `doc/tables/summary.tex` — versión LaTeX para incrustar en la memoria.
- `doc/tables/per_benchmark.md` — cross-tab configuración × benchmark.
- `doc/tables/problem_difficulty.md` — top-20 problemas con mayor desacuerdo.
- `doc/tables/adherence.md` — adherencia estructural por configuración.

## Documentación

- [Resumen del proyecto y estado por sprint](CONTEXT.md)
- [Anexo de decisiones técnicas](doc/decisiones.md)
- Capítulos de la memoria: [01](doc/capitulos/01_introduccion.md) ·
  [02](doc/capitulos/02_estado_del_arte.md) ·
  [03](doc/capitulos/03_objetivos.md) ·
  [04](doc/capitulos/04_metodologia.md) ·
  [05](doc/capitulos/05_desarrollo.md) ·
  [06](doc/capitulos/06_experimentos.md) ·
  [07](doc/capitulos/07_resultados.md) ·
  [08](doc/capitulos/08_conclusiones.md)
- [Bibliografía global](doc/referencias/bibliografia.md)
