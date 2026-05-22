# Capítulo 6: Experimentos

## 6.1. Matriz experimental

La matriz experimental cruza tres dimensiones independientes: la configuración
del sistema, el benchmark sobre el que se evalúa y la semilla pseudoaleatoria
que indexa la réplica. La tabla 6.1 enumera los puntos de cada dimensión.

| Dimensión | Valores | Cardinalidad |
|---|---|---|
| Configuración | baseline, sequential, self_reflection_r1, self_reflection_r2, self_reflection_r3, ablation_no_pm, ablation_no_architect, ablation_no_reviewer | 8 |
| Benchmark | HumanEval (164 problemas), MBPP (subconjunto de 200 problemas) | 364 problemas |
| Semilla | 42, 123, 456 | 3 réplicas |
| **Total** | | **8 736 ejecuciones** |

Tabla 6.1. Dimensiones de la matriz experimental completa.

El total efectivo de la corrida principal corresponde a la versión reducida
sobre HumanEval únicamente y las cinco configuraciones originales (baseline,
sequential, self_reflection_r1/r2/r3): 164 × 3 × 5 = 2 460 ejecuciones. Las
ablaciones y la inclusión de MBPP se contemplan como segunda pasada del runner,
aprovechando la propiedad de resumibilidad descrita en la sección 5.8: tanto
el cache local de MBPP (`experiments/cache/mbpp.json`) como el registro de
configuraciones del runner están ya preparados para esa pasada, y reincorporar
benchmarks o configuraciones no obliga a re-ejecutar nada de lo ya completado.

La figura 6.1 reproduce, a nivel del banco experimental, el diagrama
de flujo de la sección 4.4 (figura 4.1). En este capítulo la figura
sirve de referencia visual mientras se describen las decisiones de
configuración que rellenan cada caja del flujo: qué benchmark se
carga, qué modelo se invoca, qué métricas se computan y dónde aterriza
cada artefacto.

![Flujo experimental del TFG](figures/flujo_experimental.png)

Figura 6.1. Flujo experimental visto desde el banco: cada caja se
describe en detalle en las secciones 6.2 a 6.7
(`figures/flujo_experimental.png`).

## 6.2. Modelo y backend de inferencia

El experimento final utiliza el modelo Qwen 2.5 Coder 7B Instruct con
cuantización Q4_K_M, servido por Ollama en local sobre el equipo MacBook Air
M2 descrito en la sección 4.5. La elección del modelo se justifica en el
capítulo 4 por la restricción de memoria unificada del equipo de ejecución.

La temperatura de generación se fija en 0.2 para todos los agentes excepto
el Developer reflexivo del grafo de self-reflection, que opera con
temperatura 0.4. Esta asimetría —deliberada, descrita en la sección 5.4.3—
favorece la diversidad de salidas durante la fase de revisión iterativa
para que el ciclo Reviewer → Developer pueda explorar soluciones
alternativas y no quede atrapado en revisiones idénticas a la propuesta
inicial.

El parámetro `seed` de la API de inferencia de Ollama no se fija. La
inferencia es muestral por naturaleza y forzar determinismo violaría la
definición del estimador pass@k de Chen et al. (2021), que asume `n`
generaciones independientes por problema. Las tres semillas del experimento
indexan **réplicas del experimento**, no semillas del modelo: cada réplica
constituye una muestra independiente de la distribución de salidas del
pipeline para ese problema. La justificación completa de esta decisión
aparece como S5 en el anexo de decisiones técnicas.

## 6.3. Benchmarks utilizados

### 6.3.1. HumanEval

HumanEval (Chen et al., 2021) es el benchmark central del estudio. Aporta
164 problemas de programación en Python, cada uno definido por una firma de
función con docstring descriptivo y un conjunto de aserciones unitarias. La
versión utilizada es `evalplus/humanevalplus`, equivalente al HumanEval
original pero con suite de tests extendida que aumenta la sensibilidad del
benchmark a errores sutiles, especialmente en casos límite.

El cache local del benchmark (`experiments/cache/humaneval.json`) elimina la
dependencia de HuggingFace durante la ejecución: una vez descargado en la
primera invocación, las réplicas siguientes operan completamente offline.

### 6.3.2. MBPP

MBPP (Mostly Basic Python Problems, Austin et al., 2021) proporciona un
banco complementario a HumanEval. Sus problemas, redactados como
descripciones cortas en lenguaje natural y con tests asociados, cubren
ejercicios introductorios y de complejidad media. La versión utilizada se
limita a los 200 primeros problemas del split de entrenamiento, una muestra
suficiente para detectar diferencias entre configuraciones sin saturar el
presupuesto de cómputo.

El cache local (`experiments/cache/mbpp.json`) replica la lógica de
HumanEval: se descarga una sola vez desde
`google-research-datasets/mbpp`, se normaliza al schema interno del
proyecto y se vuelca a disco para usos posteriores.

## 6.4. Métricas

### 6.4.1. pass@k

La métrica principal es pass@1 con su estimador insesgado de Chen et al.
(2021):

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

donde `n` es el número de muestras por problema (en este experimento `n=3`,
las tres réplicas indexadas por semilla), `c` el número de réplicas que
superan todos los tests y `k` el parámetro del estimador. El cálculo se
realiza en espacio logarítmico mediante `math.lgamma` para evitar
desbordamiento numérico cuando `n` y `c` crecen, según la implementación de
referencia del paper original.

pass@1 se reporta como proporción simple de ejecuciones que superan todos
los tests: pass@1 = (#ejecuciones con éxito) / (#ejecuciones totales). pass@3
se reporta como promedio del estimador por problema con `k=3`, lo que
equivale a la probabilidad de que al menos una de tres muestras sea
correcta.

### 6.4.2. Tasa media de superación de pruebas

Para soluciones que no superan todos los tests del problema, la tasa media
de superación de pruebas (`average_test_pass_rate`) cuantifica el grado de
corrección parcial: para cada ejecución se calcula la fracción de tests
que pasan y la métrica final es el promedio por configuración. Su
utilidad es discriminar entre soluciones completamente incorrectas y
soluciones próximas a la corrección, especialmente relevante en el ciclo
de self-reflection donde una mejora iterativa puede aumentar la fracción
de tests aprobados sin alcanzar todavía el éxito completo.

### 6.4.3. Coste en tokens y latencia

El coste en tokens se desglosa en tokens de entrada y de salida, agregados
durante todas las llamadas al LLM dentro de una misma ejecución. Los
contadores se acumulan en el `AgentState` por cada agente; al final de la
ejecución representan el coste total del pipeline para resolver el
problema. La latencia se mide como wall-clock total desde la inicialización
del grafo hasta la finalización del último nodo.

Ambas métricas son centrales para responder la H3 de trade-off
calidad-coste: la mejora en pass@1 que aporta una configuración compleja
sólo es justificable si el incremento en tokens y latencia es proporcional
al beneficio observado.

### 6.4.4. Adherencia estructural al protocolo

La propuesta del trabajo postula que el protocolo de comunicación
estructurada por artefactos reduce las alucinaciones respecto a la
conversación libre. Para operacionalizar la afirmación se introduce la
métrica de **adherencia estructural**: la fracción de ejecuciones en las
que ningún agente del pipeline emitió un artefacto malformado.

La señal concreta es la ausencia de bloques ```python``` en la salida del
agente Developer (o del agente Baseline en su configuración monolítica).
Cuando el modelo no enmarca el código en el fence esperado el agente
registra una advertencia de tipo `WARNING` en el log; el script
`experiments/adherence_metric.py` correlaciona esas advertencias contra el
CSV de cada configuración y produce el ratio:

$$\text{adherencia} = 1 - \frac{\#\text{ejecuciones con aviso estructural}}{\#\text{ejecuciones totales}}$$

La métrica es post-hoc, read-only y se calcula sin re-invocar al LLM. Su
resultado se reporta como porcentaje en la tabla 7.4 del capítulo de
resultados.

### 6.4.5. Intervalos de confianza

Las métricas continuas se acompañan de intervalos de confianza al 95%
mediante bootstrap percentil con 2 000 remuestras por configuración. La
implementación, en `experiments/analyze_results.py`, utiliza una semilla
fija para que los intervalos sean reproducibles entre ejecuciones del
análisis. Los intervalos se incluyen en la columna principal de pass@1 de
la tabla de resumen para que el lector pueda juzgar de un vistazo la
solidez estadística de cada estimación.

## 6.5. Pipeline de análisis

El pipeline de análisis (`experiments/analyze_results.py`) consume los CSV
crudos del directorio `experiments/results/` y produce los artefactos
listos para la memoria. Funciona de forma incremental: puede ejecutarse
mientras el experimento principal todavía está en curso y regenera todas
las figuras y tablas con los datos disponibles en ese momento.

| Artefacto | Fichero |
|---|---|
| pass@1 por configuración (barras + IC 95%) | `figures/pass_at_1.png` |
| Frontera de Pareto coste–calidad (log-tokens vs pass@1) | `figures/cost_quality_pareto.png` |
| Distribución de latencias (boxplot por configuración) | `figures/latency_box.png` |
| Distribución del contador de revisiones (SR r1/r2/r3) | `figures/revision_distribution.png` |
| Tabla de resumen (Markdown) | `doc/tables/summary.md` |
| Tabla de resumen (LaTeX) | `doc/tables/summary.tex` |
| Cross-tab configuración × benchmark | `doc/tables/per_benchmark.md` |
| Top-20 problemas con mayor desacuerdo entre configs | `doc/tables/problem_difficulty.md` |

Tabla 6.2. Artefactos producidos por el pipeline de análisis.

El informe de problemas con mayor desacuerdo merece detalle. Para cada
problema del benchmark se computa el spread = max(pass_rate) − min(pass_rate)
a través de las configuraciones que lo evaluaron. Los problemas con spread
alto son aquellos donde la elección de arquitectura cambia el resultado:
constituyen el subconjunto más informativo para la discusión cualitativa
del capítulo 7. Los problemas con spread cercano a cero —resueltos por
todas las configuraciones o fallidos por todas— no aportan información
discriminativa entre arquitecturas.

## 6.6. Protocolo de ejecución y trazabilidad

Cada ejecución individual se persiste como una fila del CSV correspondiente
a su configuración. Los campos del CSV son: `benchmark`, `problem_id`,
`config`, `seed`, `pass_all_tests`, `test_pass_rate`, `tokens_input`,
`tokens_output`, `latency_seconds`, `revision_count`, `timestamp`, `model`
y `error`. El campo `model` registra el identificador exacto del modelo
servido (`qwen2.5-coder:7b-instruct-q4_K_M` para la corrida principal), lo
que permite distinguir corridas con backend diferente al fusionar
resultados de varias sesiones.

El campo `timestamp` se rellena con la marca UTC al iniciar la ejecución, no
al guardarla, para preservar el orden temporal de generación. El campo
`error` aloja la primera línea del traceback cuando la ejecución falla; el
traceback completo se vuelca a `experiments/logs/errors.log`. El runner
ignora las filas con `error` no vacío al construir el conjunto de
ejecuciones completadas, lo que permite reintentar fallos transitorios sin
modificación manual del CSV.

## 6.7. Estado de la corrida

La corrida principal se lanza el 22 de mayo de 2026 con el comando:

```bash
LLM_BACKEND=ollama nohup python experiments/run_experiments.py \
    --model qwen2.5-coder:7b-instruct-q4_K_M \
    > experiments/logs/run.out 2>&1 &
```

El proceso, con PID registrado en la consola, se ejecuta en background con
`nohup` para sobrevivir al cierre del terminal. El dashboard de seguimiento
(`experiments/dashboard.py`) se puede invocar en cualquier momento sobre el
mismo `progress.json` sin interferir con el runner.

La cardinalidad efectiva al cierre del documento se reporta en el inicio
del capítulo 7 junto con los resultados parciales disponibles.

## 6.8. Reproducibilidad

Todas las dependencias se declaran en `pyproject.toml` con versiones
fijadas. El entorno se reproduce con:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[experiments]"
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

Los prompts completos de los seis agentes se recogen en el Anexo A. Los
caches locales de los benchmarks (`experiments/cache/humaneval.json` y
`mbpp.json`) se publican junto con el repositorio para que la
reproducción no dependa de la disponibilidad de HuggingFace. El sandbox
de ejecución es un proceso Python aislado con timeout y builtins
restringidos, descrito en la sección 5.5 y configurable mediante
parámetros del módulo `src/evaluation/sandbox.py`.

La corrida completa sobre el equipo de referencia (MacBook Air M2, 16 GB)
y el modelo seleccionado se completa en cuestión de horas para baseline,
extendiéndose a varios días para las configuraciones con self-reflection.
El runner es resumible, por lo que la ejecución puede interrumpirse y
retomarse sin pérdida de progreso.
