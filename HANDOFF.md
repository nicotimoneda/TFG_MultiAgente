# HANDOFF — TFG Multi-Agente (Nico Timoneda, UAX)

Documento de transferencia para arrancar una nueva sesión de Claude sin contexto previo. Pega este fichero entero en la primera conversación.

---

## Identidad del proyecto

- **Título:** *Orquestación de Equipos de Agentes LLM con Roles Especializados para Resolución Colaborativa de Tareas Complejas de Ingeniería de Software.*
- **Autor:** Nicolás Timoneda Martínez (NP 147254). Grado en Computación e IA, UAX. Curso 2025-26.
- **Repo local:** `~/UNI/TFG/TFG_MultiAgente`
- **Repo remoto:** `https://github.com/nicotimoneda/TFG_MultiAgente`
- **Entrega institucional:** **8 de junio de 2026**.
- **Hoy:** 26-mayo-2026 (S23 cerrada).

## Resumen ejecutivo del trabajo

Comparativa empírica controlada de **3 configuraciones LangGraph** sobre **HumanEval (164 problemas)** con **Qwen 2.5 Coder 7B Instruct Q4_K_M** en Ollama local sobre MacBook Air M2:

1. `baseline` — un único LLM monolítico.
2. `sequential` — pipeline de 5 roles (PM → Arquitecto → Developer → QA → Reviewer).
3. `self_reflection_r1` — pipeline secuencial + bucle Reviewer → Developer con `max_revisions = 1`.

3 semillas (42, 123, 456). **Matriz principal: 1 476 ejecuciones**.

Tres ablaciones (`no_pm`, `no_architect`, `no_reviewer`) y variantes SR_r2/r3 quedan **implementadas y testadas pero no evaluadas** (decisión de scope S8, ver `doc/decisiones.md`).

## Hallazgo principal del TFG (LA NARRATIVA ESTABLECIDA — NO ROMPER)

El experimento **no respalda** la hipótesis intuitiva de que la especialización multi-agente mejora pass@1. Cifras al 80,9 % de la matriz (1 195 / 1 476 runs):

| Configuración | n | pass@1 | IC 95 % | Tokens | Latencia (s) |
|---|---:|---:|---|---:|---:|
| Baseline | 492 | **80,08 %** | [76,4 ; 83,5] | 283 | 5,1 |
| Sequential | 492 | 58,33 % | [53,9 ; 62,6] | 11 614 | 396,4 |
| SR (r=1) | 211* | 67,30 % | [60,7 ; 73,5] | 13 719 | 386,8 |

McNemar baseline vs sequential: **p < 0,0001** (b=128, c=21). Baseline vs SR_r1: **p < 0,0001** (b=51, c=4). Sequential vs SR_r1: p = 0,19 (no concluyente todavía).

**Veredicto por hipótesis:**
- **H1 (especialización mejora pass@1)** → **rechazada con dirección invertida**.
- **H2 (SR mejora secuencial)** → **no concluyente** con tendencia favorable (esperando cierre).
- **H3 (trade-off coste-calidad)** → **rechazada con dirección invertida**; el multi-agente cuesta 40× más tokens y 77× más latencia para peor pass@1.

**Tres causas plausibles del hallazgo negativo** (ya argumentadas en cap 8.3.1):
1. Propagación de errores entre roles del pipeline.
2. Sobrecarga del prompt de rol en un modelo de 7 B con ventana limitada.
3. HumanEval (funciones aisladas) es inadecuado para evaluar pipelines diseñados para coordinación entre fases.

**Posicionamiento en la literatura crítica:** Chen et al. 2024 (*Are More LLM Calls All You Need?*, ICML), Olausson et al. 2024 (*Is Self-Repair a Silver Bullet?*, ICLR), Huang et al. 2024 (AgentCoder). El TFG se inscribe en esa línea.

## Estado del experimento

**Última lectura (26-may-2026 14:00 UTC):**

```
Total:     1 476
Completed: 1 239 (83,9 %)
Failed:    0
Current:   self_reflection_r1, HumanEval/85, seed 42
```

Ritmo ~10 runs/h. Quedan ~237 runs ≈ 24 h. **ETA cierre: 27-28 mayo.**

Procesos en marcha:
- `nohup bash scripts/experiment_watchdog.sh` (background, supervivencia al cierre de terminal).
- 2 LaunchAgents activos: `com.nico.tfg.watchdog` + `com.nico.tfg.caffeinate`.
- `sudo pmset -c disablesleep 1` para Mac con tapa cerrada.

Para ver estado:
```bash
cat ~/UNI/TFG/TFG_MultiAgente/experiments/results/progress.json
python ~/UNI/TFG/TFG_MultiAgente/experiments/dashboard.py
```

## Reglas de estilo de la memoria (feedback del tutor — OBLIGATORIO RESPETAR)

1. **Sin "Referencias" al final de cada capítulo.** Toda la bibliografía va en `doc/referencias/bibliografia.md` (humano) + `referencias.bib` (BibTeX). Las citas en cuerpo se hacen "Autor (año)" o "(Autor, año)".
2. **Densidad visual alta.** Cada capítulo de desarrollo/experimentos/resultados tiene ≥1 figura o tabla.
3. **Estructura UAX:** ver `Documentación - UAX/` y `doc/entregas/` para las entregas previas y la guía oficial.

## Memoria del TFG (qué hay y dónde)

```
doc/capitulos/
├── 00_resumen.md              # Resumen + abstract bilingüe (con "Resultado principal")
├── declaracion_ia.md          # Declaración de uso de IA (HONESTA — revisada hoy)
├── 01_introduccion.md         # Motivación + pregunta (anticipa hallazgo)
├── 02_estado_del_arte.md      # SOTA + literatura crítica reciente
├── 03_objetivos.md            # OE1-OE7 + H1-H3 + alcance/limitaciones
├── 04_metodologia.md          # Enfoque empírico-comparativo
├── 05_desarrollo.md           # Implementación + recorrido HumanEval/1
├── 06_experimentos.md         # Banco experimental + métricas
├── 07_resultados.md           # Datos + §7.6 contraste H1/H2/H3 + §7.7 análisis cualitativo
├── 08_conclusiones.md         # OE + por hipótesis + §8.3.1 discusión + §8.5 desviaciones
├── anexo_A_prompts.md         # Prompts verbatim (verificados vs código)
├── anexo_B_decisiones.md      # S1..S9 (S8 = recorte de scope, MUY referenciada)
├── anexo_C_reproducir.md      # Comandos de reproducción
├── anexo_D_glosario.md        # Siglas y términos
├── anexo_E_etica.md           # Ética + sostenibilidad (CO2e calculado con datos reales)
└── anexo_F_agradecimientos.md

doc/referencias/
├── bibliografia.md            # Versión legible
└── referencias.bib            # BibTeX canónico

doc/entregas/
└── ResumenEjecutivo_EntregaFinal.md
doc/decisiones.md              # Documento canónico de decisiones (D1..D6, S7, S8)
```

## Código y experimento

```
src/state/                     # AgentState (TypedDict)
src/agents/                    # base_agent, baseline_agent
src/agents/roles/              # PM, architect, developer, qa_tester, code_reviewer
src/graph/                     # baseline_graph, sequential_graph, self_reflection_graph + ablaciones
src/llm/client_factory.py      # Factory Ollama/Cerebras
src/evaluation/                # Sandbox + cargadores benchmark

experiments/
├── run_experiments.py         # Runner principal (resumible, atómico)
├── analyze_results.py         # Genera figuras + tablas + narrativa
├── adherence_metric.py        # Adherencia estructural
├── dashboard.py               # Dashboard tiempo real
├── cache/humaneval.json       # 164 problemas con prompts y tests canónicos
└── results/
    ├── baseline_results.csv          # 492 runs cerrados
    ├── sequential_results.csv        # 492 runs cerrados
    ├── self_reflection_r1_results.csv # ~255 runs (en curso)
    └── progress.json                  # Estado del runner

scripts/
├── build_memoria.sh           # Concatena .md → .docx con pandoc
└── experiment_watchdog.sh     # Supervisa runner, reinicia si falla

figures/                       # PNG + Mermaid (regenerados automáticamente)
```

## Lo crítico: cuando cierre el experimento (27-28 may)

**Comando único de cierre:**
```bash
cd ~/UNI/TFG/TFG_MultiAgente
python experiments/analyze_results.py    # regenera figuras + tablas
python experiments/adherence_metric.py   # regenera adherencia
bash scripts/build_memoria.sh            # .docx final
```

**Lo que se actualizará automáticamente:**
- `doc/tables/summary.md`, `pairwise_mcnemar.md`, `problem_difficulty.md`, `adherence.md`, `findings_narrative.md`.
- `figures/pass_at_1.png`, `cost_quality_pareto.png`, `latency_box.png`, `revision_distribution.png`.

**Lo que hay que actualizar manualmente con los números finales** (búsqueda y reemplazo de cifras concretas):
- Cap 7: tabla 7.2 (cifras de SR_r1 cambian al cerrar), texto de §7.3.1 (interpretación), §7.6.2 (verificar si H2 cruza a p < 0,05).
- Cap 8.3 H2 (puede pasar de "no concluyente" a "respaldada").
- 00_resumen.md (Resultado principal ES y EN).
- doc/entregas/ResumenEjecutivo_EntregaFinal.md (tabla).
- README.md (tabla).
- anexo E §E.5 (recalcular CO2e con latencias finales).

**Para H2:** vigilar si la diferencia sequential vs SR_r1 cruza significativa cuando SR_r1 esté completo (al cierre tiene n=492 pares en vez de 211). Si lo hace, H2 pasa de "no concluyente" a "respaldada"; si no, queda como está.

## Historial de commits relevantes (recientes)

```
491560a (S23) Declaración IA precisa + CO2e real
852eb75 (S22) Análisis cualitativo §7.7 + anexo B S8 + README
a1cf5d1 (S21) Cap 4/5 coherencia + humanizer + build .docx
c677fbe (S20) Literatura crítica + cap 1/resumen + §8.5 desviaciones
c3f24ae (S19) Cap 7 + 8 reescritos con datos parciales (hallazgo negativo)
e9a32d6 (S18) Coherencia caps 1/3/6/7/8 con scope final
```

## Cosas a NO ROMPER

- **No revertir el hallazgo negativo.** Está bien sustentado (McNemar p<0,0001) y la narrativa actual lo enmarca como contribución. Cualquier reescritura optimista contradice los datos.
- **No reintroducir SWE-bench, MBPP, ClassEval, DevBench, comparativa de LLMs por rol o Dynamic Task Decomposition como "entregados".** Están en §8.5 (Desviaciones) y §8.7 (Líneas futuras) por una razón.
- **No quitar la referencia "entrada S8 del anexo de decisiones"**: está en cap 3, cap 4, cap 5, cap 6, cap 8 y la entrada está en `anexo_B_decisiones.md §B.8` y `doc/decisiones.md`.
- **No commitear `experiments/results/*.csv` ni `progress.json`** mientras el experimento está corriendo (cambian solos).
- **Nunca usar emojis en archivos** salvo petición explícita del usuario.
- **No re-añadir bibliografía al final de los capítulos** (regla del tutor). Solo en `doc/referencias/`.

## Pendiente fuera de scope hoy

- **Defensa oral** (slides, guion). El usuario lo dejó para cuando cierre el experimento.
- **Verificar el .docx en Word** (formato, TOC, paginación) manualmente.
- **Eventual segunda pasada del runner** con SR_r2/r3 y ablaciones si sobra tiempo de cómputo antes del 8 de junio (decisión del autor, no automática).

## Cómo retomar (primera frase de la nueva sesión)

> Lee `HANDOFF.md` en `~/UNI/TFG/TFG_MultiAgente/`. Comprueba el estado del experimento con `cat experiments/results/progress.json`. Si ya está cerrado (completed == total), ejecuta el pipeline de cierre y actualiza los números finales en las secciones marcadas en el handoff. Si sigue corriendo, dime cuánto queda y propón qué hacer en el tiempo restante.

---

*Última actualización del handoff: 2026-05-26, sesión S23, commit `491560a`.*
