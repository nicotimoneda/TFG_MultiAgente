# Anexo C: Comandos de reproducción

Este anexo consolida en un único listado los comandos necesarios para
reproducir el experimento, regenerar las figuras y tablas y verificar la
integridad del sistema. La fuente canónica de cada comando es el
`README.md` del repositorio; este anexo es la versión completa de
referencia para el tribunal evaluador.

## C.1. Preparación del entorno

```bash
git clone https://github.com/nicotimoneda/TFG_MultiAgente.git
cd TFG_MultiAgente
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments]"
```

Las dependencias declaradas en `pyproject.toml` son LangGraph (≥0.2),
LangChain (≥0.3), `langchain-openai` (≥0.2), HuggingFace `datasets`
(≥2.20), SciPy (≥1.13), NumPy (≥1.26), Pandas (≥2.2) y, dentro del
extra `experiments`, `rich`, `tqdm` y `matplotlib`. Pytest se instala
adicionalmente para ejecutar la suite de pruebas.

## C.2. Backend de inferencia

Configuración local recomendada (utilizada en la corrida principal):

```bash
ollama serve   # en una terminal
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

Configuración remota para verificaciones cruzadas:

```bash
export CEREBRAS_API_KEY="..."   # cuenta gratuita en cloud.cerebras.ai
export LLM_BACKEND=cerebras
```

## C.3. Verificación rápida

Antes de lanzar la corrida completa, el smoke test ejecuta 10 problemas
sobre las 5 configuraciones principales en aproximadamente 15 minutos:

```bash
LLM_BACKEND=ollama python experiments/quick_check.py
```

La suite de pruebas unitarias se ejecuta con:

```bash
LLM_BACKEND=ollama pytest -q
```

Cubre el sandbox de subproceso (timeouts, builtins bloqueados,
errores de sintaxis), las funciones de pass@k y la topología de las
tres variantes de ablación.

## C.4. Corrida completa

Recomendado en una sesión separable (terminal con `nohup` o `tmux`):

```bash
LLM_BACKEND=ollama nohup python experiments/run_experiments.py \
    --model qwen2.5-coder:7b-instruct-q4_K_M \
    --benchmarks humaneval \
    --configs baseline,sequential,self_reflection_r1,self_reflection_r2,self_reflection_r3 \
    > experiments/logs/run.out 2>&1 &
```

Las ablaciones se incorporan a la corrida añadiendo configs:

```bash
LLM_BACKEND=ollama python experiments/run_experiments.py \
    --model qwen2.5-coder:7b-instruct-q4_K_M \
    --configs ablation_no_pm,ablation_no_architect,ablation_no_reviewer
```

MBPP se incluye con:

```bash
LLM_BACKEND=ollama python experiments/run_experiments.py \
    --model qwen2.5-coder:7b-instruct-q4_K_M \
    --benchmarks humaneval,mbpp
```

Todas las invocaciones son resumibles: si la corrida se interrumpe,
basta con volver a ejecutar el mismo comando para que retome donde
quedó. Las ejecuciones ya completadas se omiten.

## C.5. Seguimiento en vivo

```bash
python experiments/dashboard.py     # tabla interactiva con Rich
cat experiments/results/progress.json    # JSON puntual
tail -f experiments/logs/run.out         # log textual
```

## C.6. Análisis y regeneración de artefactos

Tras (o durante) la corrida:

```bash
python experiments/analyze_results.py    # figuras + tablas + McNemar pareado
python experiments/adherence_metric.py   # adherencia estructural
```

Productos en `figures/` y `doc/tables/`:

| Fichero | Contenido |
|---|---|
| `figures/arquitectura_sistema.png` | Diagrama del sistema |
| `figures/graph_baseline.png` | Topología del grafo baseline |
| `figures/graph_sequential.png` | Topología del grafo secuencial |
| `figures/graph_self_reflection_r1.png` (r2, r3) | Grafos con bucle de revisión |
| `figures/graph_ablation_no_pm.png` (no_architect, no_reviewer) | Grafos de las ablaciones |
| `figures/pass_at_1.png` | Barras de pass@1 con IC 95% bootstrap |
| `figures/cost_quality_pareto.png` | Frontera de Pareto coste–calidad |
| `figures/latency_box.png` | Distribución de latencias por configuración |
| `figures/revision_distribution.png` | Uso del ciclo en self-reflection |
| `doc/tables/summary.md` | Tabla principal de resultados (Markdown) |
| `doc/tables/summary.tex` | Versión LaTeX |
| `doc/tables/per_benchmark.md` | Cross-tab configuración × benchmark |
| `doc/tables/problem_difficulty.md` | Top-20 problemas con mayor desacuerdo |
| `doc/tables/pairwise_mcnemar.md` | Tests pareados de McNemar |
| `doc/tables/adherence.md` | Adherencia estructural por configuración |

## C.7. Acceso a los benchmarks adicionales

Los cargadores de ClassEval y SWE-bench Lite están disponibles y
cachean localmente:

```bash
python -c "from src.evaluation.classeval_loader import load_classeval; load_classeval()"
python -c "from src.evaluation.swebench_loader import load_swebench_lite; load_swebench_lite()"
```

La ejecución real de SWE-bench requiere el harness Docker oficial,
referenciado en `src/evaluation/swebench_loader.py`. La ejecución de
ClassEval requiere adaptar los prompts del Developer a generación de
clases (sección 8.6).

## C.8. Hardware de referencia

Las latencias y el coste reportados en el capítulo 7 se obtienen sobre
un MacBook Air con Apple M2, 8 núcleos de CPU, 10 núcleos de GPU y
16 GB de memoria unificada. Otros equipos pueden reproducir las tasas
de éxito (pass@1, adherencia) pero las latencias absolutas variarán.
