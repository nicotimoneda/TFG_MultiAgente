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
| OE5 — Evaluación empírica | Cumplido en lo esencial, en cierre para SR_r1 | Baseline y sequential completos (492 runs cada uno); SR_r1 al 43 % (211/492) y avanzando |
| OE6 — Análisis cualitativo | Parcial | Estudio piloto en 7.2; análisis por dificultad del problema (H3) pendiente del cierre de SR_r1 |
| OE7 — Reproducibilidad documentada | Cumplido | `pyproject.toml`, `CONTEXT.md`, Anexo A, caches en repo |

Tabla 8.1. Estado de cumplimiento de los objetivos específicos al cierre
del documento.

Dos objetivos quedan en estado distinto a "cumplido". El OE5 está
ejecutándose en el momento de cerrar el documento: la infraestructura del
banco es completa y resumible, y la cardinalidad del experimento avanza
hacia las 1 476 ejecuciones de la matriz principal (3 configuraciones ×
164 problemas × 3 semillas). El OE6 depende del
OE5: el análisis cualitativo comparativo entre configuraciones —dónde gana
el pipeline complejo y dónde no— sólo se puede ejecutar sobre los CSV de
todas las configuraciones. La parte cualitativa del OE6 que es
independiente del experimento, el estudio piloto que motivó el diseño,
está cumplida en la sección 7.2.

## 8.3. Conclusiones por hipótesis

El análisis del capítulo 7 sobre el 80,9 % de la matriz principal
(1 195 de 1 476 ejecuciones) permite ya emitir conclusiones firmes para
dos de las tres hipótesis y dejar la tercera en estado no concluyente
a la espera del cierre.

**Sobre H1 (especialización).** **Rechazada con dirección invertida.**
El pipeline secuencial obtiene 58,33 % de pass@1 frente al 80,08 % del
baseline (diferencia de 21,8 puntos, McNemar p < 0,0001 sobre 492 pares
con b = 128 y c = 21). La especialización por roles no mejora la
corrección sobre HumanEval con este modelo: la empeora. La hipótesis
de partida —que distribuir el trabajo entre cinco agentes especializados
produciría soluciones más correctas que un único agente bien
promptado— no se sostiene en este experimento.

**Sobre H2 (auto-revisión).** **No concluyente con tendencia en el
sentido esperado.** La configuración con self-reflection
(`max_revisions = 1`) alcanza 67,30 % de pass@1, casi 9 puntos por
encima del pipeline secuencial sin ciclo, pero la diferencia no es
significativa con los 211 pares disponibles al cierre (p = 0,194). Si
la tendencia se mantiene cuando SR_r1 complete la pasada, la hipótesis
quedará respaldada para `r = 1`. Para `r > 1` el contraste no se puede
hacer con los datos actuales: SR_r2 y SR_r3 quedaron fuera del scope
por restricciones de cómputo.

**Sobre H3 (trade-off coste-calidad).** **Rechazada con dirección
invertida.** La hipótesis postulaba un trade-off cuantificable según
la dificultad del problema, con el beneficio del pipeline complejo
disminuyendo en problemas simples. Los datos contradicen el supuesto
sobre el que se construyó la hipótesis: el pipeline complejo no aporta
beneficio en ningún rango. El multi-agente paga 40 veces más tokens y
77 veces más latencia que el baseline para obtener peor pass@1, y la
frontera de Pareto coste-calidad la ocupa por completo el baseline. No
hay trade-off que analizar; hay una dominancia clara de la
configuración monolítica.

**Sobre la adherencia al protocolo estructurado.** El protocolo de
comunicación por artefactos se cumple sin excepciones en baseline y
secuencial (100 %) y con una sola desviación en SR_r1 (99,53 %). La
afirmación de la literatura de que el protocolo estructurado reduce
las alucinaciones de formato se confirma. Pero la observación
relevante es que la adherencia estructural mide el formato, no la
corrección: el mismo pipeline que respeta el contrato de salida sin
fallos genera código menos correcto que un agente único sin protocolo.

### 8.3.1. Discusión: por qué el pipeline empeora al baseline

El resultado contradice la lectura habitual de la literatura
multi-agente, que reporta mejoras consistentes del pipeline frente al
monolítico (ChatDev, Qian et al., 2024; MetaGPT, Hong et al., 2024).
Hay tres explicaciones plausibles, no excluyentes, que el experimento
sugiere y que merecen investigación futura.

**Propagación de errores a lo largo del pipeline.** Cada agente del
pipeline recibe como entrada la salida del agente anterior. Un error
de interpretación del Product Manager al traducir el enunciado a PRD
se arrastra al Arquitecto, que lo reinterpreta, y de ahí al Developer,
que implementa contra un diseño ya desviado. En el baseline esa
cadena no existe: el modelo lee el enunciado original y genera código
directamente. El estudio piloto de la sección 7.2 anticipó este
patrón, pero su magnitud cuantitativa solo se hace visible con la
matriz completa.

**Inadecuación del modelo de 7 B parámetros para prompts largos.**
Cada agente del pipeline opera con un prompt de sistema extenso —rol,
formato de salida, restricciones, ejemplos— que ocupa una parte
considerable de la ventana de contexto. Para HumanEval, donde el
enunciado del problema es pequeño y la solución es una función
aislada, esa estructura puede ser contraproducente: el modelo dedica
capacidad cognitiva a respetar el protocolo de rol en vez de a
resolver el problema. Los trabajos que reportan mejoras del
multi-agente —ChatDev, MetaGPT— evalúan con modelos frontera (GPT-4)
sobre tareas mucho mayores que una función aislada. La ventaja del
pipeline puede depender críticamente de que (a) el modelo tenga
capacidad sobrante para gestionar la sobrecarga del protocolo y (b)
la tarea sea suficientemente grande como para que la especialización
compense la fricción.

**HumanEval como benchmark inadecuado para evaluar pipelines
multi-agente.** HumanEval pide implementar una función aislada con un
docstring corto. Es plausible que el pipeline secuencial PM →
Arquitecto → Developer → QA → Reviewer aporte valor en tareas con
componentes interdependientes (varios módulos, decisiones de
arquitectura no triviales, requisitos ambiguos), pero no en
implementaciones de una función con especificación cerrada. El
experimento, por tanto, no refuta el sistema multi-agente en
general: refuta su utilidad en el segmento específico de problemas
que HumanEval representa.

Las tres explicaciones convergen en una recomendación de diseño
honesta: **no introducir multi-agente cuando la tarea cabe en un
prompt y el modelo es pequeño**. El coste adicional es real, los
beneficios no aparecen, y la complejidad arquitectural no se justifica
empíricamente en ese régimen. Cuándo y con qué modelo el pipeline
empieza a ser competitivo es una pregunta abierta y razonable como
extensión natural del trabajo.

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
La respuesta del experimento es honesta y útil aunque no sea la que
intuitivamente cabría esperar: con un modelo de 7 B y problemas que
caben en un único prompt, **el monolítico domina**. La especialización
por roles no se traduce en mejor pass@1 sobre HumanEval, y el coste
adicional del pipeline es de dos órdenes de magnitud.

Hay algo que el número de pass@1 no captura y que vale la pena nombrar
por separado. Trabajar con cinco agentes hablando entre sí a través de
un estado tipado se siente, en la práctica, más cercano a coordinar un
equipo humano que a invocar un modelo. Cuando una solución falla, se
puede mirar qué dijo el Product Manager, qué interpretó el Arquitecto,
qué implementó el Developer y qué señaló el Reviewer; cada paso queda
explícito y auditable. Esa trazabilidad de la decisión es lo único que
el pipeline multi-agente sí ofrece y el baseline monolítico no, y
puede acabar siendo el motivo por el que estas arquitecturas se imponen
en producción aunque pierdan en benchmarks de funciones aisladas. Pero
es una propiedad cualitativa y este TFG no la ha medido.

Tres piezas sobreviven al ciclo del trabajo independientemente del
veredicto cuantitativo: el código del sistema multi-agente, el banco
experimental resumible y el pipeline de análisis automático que
regenera todo desde los CSV. Las líneas futuras de la sección 8.6
—escalado del modelo, integración de SWE-bench Lite, comparativa
heterogénea de backends por rol— pueden empezar desde esa
infraestructura sin reconstruirla, y son las extensiones donde el
multi-agente tiene más probabilidades de demostrar el valor que aquí
no ha podido demostrar.
