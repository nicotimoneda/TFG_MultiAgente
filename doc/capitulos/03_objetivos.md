# Capítulo 3: Objetivos e Hipótesis

## 3.1. Objetivo general

El objetivo general de este trabajo es diseñar, implementar y evaluar
empíricamente un sistema multi-agente basado en modelos de lenguaje de gran
tamaño que resuelva tareas de generación automática de código mediante agentes
especializados en roles del ciclo de desarrollo software. El sistema orquesta
esos agentes mediante un grafo de estado implementado en LangGraph, con un estado
compartido tipado y comunicación estructurada a través de artefactos. Su
rendimiento se compara de forma controlada contra un LLM monolítico como baseline,
sobre benchmarks públicos y con métricas definidas de antemano, con el fin de
determinar si la especialización por roles y la orquestación explícita producen
mejoras medibles en corrección funcional y en qué condiciones justifican el coste
computacional adicional.

## 3.2. Objetivos específicos

**OE1.** Revisar el estado del arte en sistemas multi-agente, agentes LLM y
frameworks multi-agente basados en LLM para identificar el gap que justifica el
enfoque propuesto y situar el trabajo respecto a la literatura existente.

**OE2.** Diseñar una arquitectura multi-agente con cinco roles especializados
—Product Manager, Arquitecto, Developer, QA Tester y Code Reviewer— coordinados
por un agente supervisor, con estado compartido tipado y comunicación estructurada
mediante artefactos.

**OE3.** Implementar un baseline LLM monolítico como referencia comparativa,
que utilice el mismo modelo base y el mismo schema de evaluación que el sistema
multi-agente para garantizar comparaciones válidas entre configuraciones.

**OE4.** Incorporar un mecanismo de auto-revisión iterativa mediante un ciclo
condicional entre el agente revisor y el agente desarrollador, parametrizado por
un número máximo de iteraciones, que permita refinar el código generado sin
intervención humana.

**OE5.** Evaluar empíricamente las tres configuraciones del sistema —baseline,
multi-agente secuencial y multi-agente con self-reflection— sobre HumanEval y un
subconjunto de MBPP, reportando pass@1, pass@k, average test pass rate, coste en
tokens y latencia, con análisis estadístico mediante test de McNemar e intervalos
de confianza bootstrap al 95%.

**OE6.** Analizar cualitativamente el comportamiento del sistema e identificar
bajo qué condiciones —tipo de problema, complejidad, número de iteraciones— el
enfoque multi-agente aporta valor frente al baseline monolítico.

**OE7.** Documentar el trabajo de forma que sea reproducible por terceros, con
código fuente público en GitHub, prompts completos en anexo y configuración
experimental declarada en pyproject.toml.

## 3.3. Hipótesis de investigación

Las hipótesis de investigación se formulan como proposiciones falsables que serán
contrastadas mediante los experimentos descritos en el capítulo 4.

**H1 — Hipótesis de especialización.** El sistema multi-agente secuencial con
roles especializados obtiene un pass@1 superior al LLM monolítico sobre
HumanEval, utilizando el mismo modelo base en ambas configuraciones. Se
contrastará mediante el test de McNemar pareado sobre los resultados pass/fail
por problema.

**H2 — Hipótesis de auto-revisión.** El ciclo iterativo de revisión entre el
agente revisor y el agente desarrollador mejora la corrección funcional —medida
por pass@1 y average test pass rate— respecto al pipeline multi-agente sin ciclo,
con un incremento cuantificable en coste de tokens y latencia. Se contrastará
mediante McNemar para la comparación pass/fail y bootstrap CI 95% para las
métricas continuas.

**H3 — Hipótesis de trade-off.** Existe un trade-off cuantificable entre calidad
de la solución y coste computacional que varía según el tipo y la complejidad del
problema, de forma que el beneficio de la configuración con self-reflection
disminuye en problemas de baja complejidad. Se analizará segmentando los
resultados por categoría de problema y comparando el ratio calidad/coste entre
configuraciones.

## 3.4. Alcance y limitaciones

El trabajo abarca el diseño, implementación y evaluación de tres configuraciones:
baseline monolítico, sistema multi-agente secuencial y sistema multi-agente con
self-reflection. La evaluación cubre HumanEval completo y un subconjunto de 200
problemas de MBPP, con análisis cuantitativo estadístico y análisis cualitativo
del comportamiento del sistema.

SWE-bench completo no se incluye en la evaluación. Sus problemas requieren
reproducir entornos Docker de repositorios externos con dependencias variables,
una complejidad de infraestructura que excede los recursos disponibles para este
TFG. Por la misma razón quedan fuera del alcance los benchmarks DevBench,
ClassEval y APPS.

El trabajo tampoco realiza una comparativa exhaustiva de backends LLM por rol.
Cubrir ese espacio de configuraciones de forma estadísticamente válida requeriría
un número de experimentos inmanejable dentro del plazo disponible. Igualmente, el
sistema no implementa Dynamic Task Decomposition con re-planificación en tiempo de
ejecución: es una extensión natural de la arquitectura propuesta, pero requiere
investigación adicional sobre mecanismos de planificación adaptativa que va más
allá del alcance de este trabajo.

Estas limitaciones se recogen como líneas de trabajo futuro en el capítulo 8.
