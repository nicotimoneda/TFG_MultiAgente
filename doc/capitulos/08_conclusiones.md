# Capítulo 8: Conclusiones y líneas futuras

## 8.1. Conclusiones generales

El TFG ha producido tres cosas que se pueden señalar concretamente: un
sistema multi-agente funcional construido sobre LangGraph, un banco
experimental que lo evalúa de forma reproducible, y los datos crudos
de esa evaluación tal como están al cerrar el documento. Los tres
elementos arquitecturales que se propusieron originalmente —baseline
monolítico, pipeline secuencial de cinco roles y pipeline con bucle
de auto-revisión— están implementados y probados. Sobre esa base se
han añadido tres ablaciones de rol que no estaban en la propuesta
original pero que resultaron útiles para responder con más finura la
pregunta de qué contribuye cada agente.

La corrida experimental seguía en marcha al cerrar la memoria, así
que algunos números del capítulo 7 son parciales por construcción.
El pipeline de análisis está pensado precisamente para eso: cada vez
que se ejecuta sobre los CSV actualizados, regenera figuras, tablas
y test pareados. La interpretación final por hipótesis depende de
cuándo se vuelva a tirar el análisis.

## 8.2. Cumplimiento de los objetivos específicos

| Objetivo | Estado | Evidencia |
|---|---|---|
| OE1 — Estado del arte | Cumplido | Capítulo 2 |
| OE2 — Arquitectura multi-agente | Cumplido | Capítulo 5, `src/graph/sequential_graph.py` |
| OE3 — Baseline LLM monolítico | Cumplido | Sección 5.4.1, `src/graph/baseline_graph.py` |
| OE4 — Mecanismo de auto-revisión | Cumplido | Sección 5.4.3, `src/graph/self_reflection_graph.py` |
| OE5 — Evaluación empírica | En ejecución | Banco listo, baseline completado, secuencial y SR en cola |
| OE6 — Análisis cualitativo | Parcial | Estudio piloto en 7.2; análisis comparativo pendiente de datos adicionales |
| OE7 — Reproducibilidad documentada | Cumplido | `pyproject.toml`, `CONTEXT.md`, Anexo A, caches en repo |

Tabla 8.1. Estado de cumplimiento de los objetivos específicos al cierre
del documento.

Dos objetivos quedan en estado distinto a "cumplido". El OE5 está
ejecutándose en el momento de cerrar el documento: la infraestructura del
banco es completa y resumible, y la cardinalidad del experimento avanza
hacia las 2 460 ejecuciones de la matriz principal. El OE6 depende del
OE5: el análisis cualitativo comparativo entre configuraciones —dónde gana
el pipeline complejo y dónde no— sólo se puede ejecutar sobre los CSV de
todas las configuraciones. La parte cualitativa del OE6 que es
independiente del experimento, el estudio piloto que motivó el diseño,
está cumplida en la sección 7.2.

## 8.3. Conclusiones por hipótesis

Las tres hipótesis del capítulo 3 se contrastarán con el cierre del
barrido experimental. El estado actual permite ya formular dos
observaciones preliminares:

**Sobre H1 (especialización).** El baseline alcanza pass@1 cercano al 83%
sobre HumanEval con 280 tokens medios por problema. Cualquier mejora del
pipeline secuencial deberá superar ese listón con un margen
estadísticamente distinguible. La amplitud del intervalo de confianza
bootstrap del baseline, calculado sobre las 492 ejecuciones que completan
la matriz baseline × HumanEval × 3 semillas, indica que el
test de McNemar dispondrá de potencia suficiente para detectar
diferencias del orden del 5–10% en pass@1 entre configuraciones cuando
todas las configuraciones acumulen un volumen similar de ejecuciones.

**Sobre la adherencia al protocolo estructurado.** El baseline registra
adherencia estructural del 100% sobre las ejecuciones completadas. Este
resultado, antes de comparar con configuraciones más complejas, no
permite concluir nada sobre la afirmación de que el protocolo
estructurado reduce alucinaciones; sí establece el listón superior con
el que el resto de configuraciones se contrastarán. El modelo Qwen 2.5
Coder de 7 B parece suficientemente competente con instrucciones de
formato simples como para emitir bloques ```python``` consistentes; el
test interesante será observar si esa adherencia se mantiene en los
pipelines más complejos.

## 8.4. Contribuciones del trabajo

Aparte de los objetivos formales, el trabajo aporta dos contribuciones
operativas que pueden servir de base para investigación futura:

**Un banco experimental reproducible y resumible.** El runner principal
(`experiments/run_experiments.py`) implementa una matriz cartesiana de
configuración × benchmark × seed cuyo estado se persiste de forma
atómica. Cualquier interrupción —fallo de inferencia, cierre del
portátil, agotamiento de cuota— se recupera reejecutando el mismo
comando, sin necesidad de manipular CSV. La adición de nuevas
configuraciones —como las tres ablaciones añadidas durante el sprint
S7— se integra sin invalidar las ejecuciones ya realizadas.

**Un pipeline de análisis incremental con figuras y tablas listas para
publicación.** El script `experiments/analyze_results.py` consume los
CSV y produce el conjunto completo de artefactos del capítulo 7,
incluyendo intervalos de confianza bootstrap y un informe automático de
problemas con mayor desacuerdo entre configuraciones. La métrica de
adherencia (`experiments/adherence_metric.py`) operacionaliza una
afirmación de la literatura que con frecuencia se sustenta sólo de
forma cualitativa.

## 8.5. Limitaciones del estudio

El alcance del trabajo se limita por las restricciones siguientes:

**Modelo de 7 B parámetros.** La elección del modelo viene determinada
por el hardware de ejecución (MacBook Air M2, 16 GB de memoria
unificada). Las conclusiones del estudio son válidas para Qwen 2.5
Coder 7B Instruct con cuantización Q4_K_M y no se extrapolan
automáticamente a modelos de mayor capacidad. Trabajos previos como
ChatDev (Qian et al., 2024) y MetaGPT (Hong et al., 2024) reportan
diferencias significativas en el comportamiento de los pipelines
multi-agente entre modelos pequeños y modelos frontera.

**Benchmark de funciones aisladas.** HumanEval y MBPP evalúan generación
de funciones independientes con pruebas unitarias asociadas. No miden
generación de proyectos multi-fichero, manejo de dependencias entre
módulos o resolución de issues sobre código existente. Las afirmaciones
sobre la utilidad del pipeline multi-agente para ingeniería de software
realista exceden la capacidad de evidencia del banco utilizado.

**SWE-bench, DevBench, ClassEval y APPS fuera del alcance.** La
propuesta inicial contemplaba estos benchmarks. SWE-bench se descartó
por requerir reproducción de entornos Docker con dependencias variables
por repositorio, un coste de infraestructura que excede los recursos
disponibles. ClassEval, DevBench y APPS quedan también fuera del alcance
por motivos similares de coste computacional con un modelo local de 7 B
servido en un único equipo.

**Comparativa heterogénea de backends por rol.** La pregunta de la
propuesta sobre qué LLM funciona mejor en cada rol (¿CodeLlama como
developer y Mixtral como reviewer?) requeriría un espacio de
configuraciones combinatoriamente grande sobre múltiples modelos. El
estudio actual usa el mismo modelo en todos los roles, lo que permite
atribuir las diferencias observadas exclusivamente al diseño del
pipeline, pero deja sin responder la pregunta de optimización por rol.

## 8.6. Líneas de trabajo futuro

Las líneas siguientes se derivan directamente de las limitaciones
anteriores y de las preguntas abiertas que el experimento principal
deja planteadas:

**Escalado del modelo.** Repetir el experimento con un modelo de mayor
capacidad —Qwen 2.5 Coder 32B sobre Cerebras, o cualquier modelo
frontera mediante API— permitiría contrastar si las diferencias entre
configuraciones se mantienen, se atenúan o se invierten al cambiar la
capacidad base. La hipótesis intuitiva es que pipelines multi-agente
aportan más con modelos más débiles, pero la literatura no la sustenta
de forma robusta y este trabajo deja la pregunta empíricamente abierta
para el modelo de 7 B.

**Integración de SWE-bench Lite.** El subconjunto Lite de SWE-bench
contiene 300 issues con menor variabilidad de dependencias que el
benchmark completo y puede acometerse con infraestructura Docker
controlada. Su incorporación al banco extendería las conclusiones a
tareas de ingeniería de software realistas. El proyecto incluye un
módulo de scaffolding (`src/evaluation/swebench_loader.py`) preparado
para esa extensión, con un caveat explícito sobre los requisitos de
Docker que la corrida real exige.

**Dynamic Task Decomposition.** El modelo 3 de la propuesta —un agente
Planner que descompone la tarea, asigna a roles y replanifica cuando
una tarea falla— no se ha implementado en este trabajo. Su incorporación
abriría preguntas nuevas sobre estabilidad, convergencia y coste
adicional, especialmente sobre problemas donde el grafo fijo del
pipeline secuencial no captura bien la estructura del problema.

**Comparativa heterogénea de backends por rol.** Asignar distintos
modelos a distintos roles —por ejemplo un modelo pequeño y rápido como
Code Reviewer y un modelo grande como Developer— permitiría explorar
el espacio coste-calidad de forma más granular que el ajuste de
`max_revisions`. La factory de clientes está preparada para esta
extensión: basta con instanciar varios clientes con configuraciones
diferentes y asignarlos a cada agente.

**Evaluación de proyectos multi-fichero.** La extensión natural del
trabajo, en línea con DevBench y ClassEval, es evaluar la capacidad del
pipeline para producir módulos completos o clases con varias funciones
relacionadas. El `AgentState` actual modela un único `code_artifact`
como cadena; la extensión exigiría modificar el estado y los prompts
para gestionar múltiples artefactos correlacionados.

## 8.7. Reflexión final

La pregunta de la propuesta era si la especialización por roles aporta
una ventaja medible frente a un LLM monolítico, y bajo qué condiciones.
El sistema y el banco experimental ahora permiten formularla como un
experimento controlado: los números finales dependen de cuándo se
consolide la matriz, pero el aparato para responderla está montado.

Hay algo que el experimento no termina de capturar y que vale la pena
nombrar. Trabajar con cinco agentes hablando entre sí a través de un
estado tipado se siente, en la práctica, más cercano a coordinar un
equipo humano que a invocar un modelo. Cuando una solución falla, se
puede mirar qué dijo el Product Manager, qué interpretó el Arquitecto,
qué implementó el Developer y qué señaló el Reviewer; cada paso queda
explícito y auditable. Esa propiedad —la trazabilidad de la decisión—
me parece más interesante a largo plazo que cualquier puntito de
pass@1 que aporte el sistema. Si los pipelines multi-agente terminan
imponiéndose, no creo que sea sólo por su precisión: creo que será
también porque permiten razonar sobre por qué el sistema hizo lo que
hizo, algo que un LLM monolítico de prompt largo simplemente no
ofrece.

Independientemente del veredicto cuantitativo final, el trabajo deja
tres piezas que sobreviven al ciclo del TFG: el código del sistema
multi-agente, el banco experimental resumible y el pipeline de
análisis automático. Las líneas futuras de la sección 8.6 pueden
empezar desde ahí sin reconstruir la infraestructura.
