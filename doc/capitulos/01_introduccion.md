# Capítulo 1: Introducción

## 1.1. Motivación y contexto

El desarrollo de software es un proceso multi-rol por naturaleza. Un equipo de
ingeniería distribuye el trabajo entre perfiles con responsabilidades distintas
—analista de requisitos, arquitecto, desarrollador, revisor de código, tester—
porque ningún profesional individual puede ejercer todas esas funciones con la
misma eficacia de forma simultánea. Esta división del trabajo no es una convención
cultural; es una respuesta a la complejidad real de construir software que funcione.

Los modelos de lenguaje de gran tamaño han cambiado lo que es posible en este
dominio. La aparición de modelos capaces de generar código fuente a partir de
descripciones en lenguaje natural ha abierto la posibilidad de automatizar partes
del ciclo de desarrollo que hasta hace pocos años requerían intervención humana
exclusiva. Sin embargo, usar un LLM como agente único para resolver una tarea
compleja de ingeniería de software presenta limitaciones concretas. La ventana de
contexto es finita: un agente único no puede mantener simultáneamente el detalle
de los requisitos, la coherencia de la arquitectura, la corrección del código y la
cobertura de las pruebas. Tampoco tiene mecanismos internos para detectar sus
propios errores: genera una solución plausible, pero no puede determinar si es
correcta más allá de que sea sintácticamente válida.

La respuesta que propone la investigación reciente es distribuir el trabajo entre
múltiples agentes con roles diferenciados, coordinados por un mecanismo de
orquestación explícito. Esta idea replica, con un sustrato tecnológico distinto,
la lógica que estructura los equipos de ingeniería humanos: especialización para
ganar profundidad, coordinación para mantener coherencia. Los sistemas que han
adoptado este enfoque —ChatDev, MetaGPT, AutoGen— han mostrado resultados más
consistentes que los agentes únicos en tareas de generación de código de
complejidad moderada, aunque su evaluación empírica sistemática sobre benchmarks
estandarizados sigue siendo escasa.

Este trabajo parte de esa observación y la lleva a un diseño concreto e
implementación evaluable: un sistema multi-agente con roles especializados,
orquestado mediante un grafo de estado en LangGraph, cuyo rendimiento se compara
de forma controlada contra un baseline monolítico en los benchmarks estándar
del campo.

## 1.2. Problema y pregunta de investigación

La pregunta central de este trabajo es si un sistema multi-agente con roles
especializados y orquestación basada en grafos de estado mejora a un LLM
monolítico en tareas de generación automática de código, y en qué condiciones ese
beneficio justifica el coste adicional en términos de tokens consumidos y latencia.

Esta pregunta tiene dos dimensiones. La primera es técnica: ¿produce el sistema
multi-agente soluciones más correctas, medidas por pass@1 y pass@k sobre HumanEval
y MBPP? La segunda es económica: ¿a qué coste computacional se obtiene esa mejora,
y existe un umbral a partir del cual la complejidad del sistema deja de producir
beneficios medibles?

Responder ambas dimensiones requiere un sistema implementado y evaluable, no solo
una arquitectura teórica. Por eso este TFG no se limita a proponer un diseño sino
que lo implementa en Python con LangGraph y lo somete a evaluación empírica
reproducible sobre benchmarks públicos con métricas definidas de antemano.

## 1.3. Objetivos y contribuciones

El objetivo principal de este trabajo es diseñar, implementar y evaluar un sistema
multi-agente basado en LLMs para la resolución colaborativa de tareas de generación
automática de código, con el fin de determinar si la especialización por roles y la
orquestación explícita producen mejoras medibles sobre un agente único.

De ese objetivo se derivan tres líneas de trabajo concretas: el diseño de una
arquitectura con roles diferenciados —Product Manager, Arquitecto, Developer, QA
Tester y Code Reviewer— coordinados por un agente supervisor mediante un grafo de
estado en LangGraph; la implementación de ese sistema en Python de forma modular,
documentada y reproducible; y su evaluación empírica sobre HumanEval y MBPP,
midiendo pass@1, pass@k, coste en tokens y latencia frente a un baseline
monolítico.

Las contribuciones concretas del trabajo son una arquitectura de orquestación
multi-agente orientada a ingeniería de software con definición explícita de roles,
estado compartido y flujo de control condicional; una implementación funcional en
LangGraph que puede ser reproducida y extendida; y un análisis empírico del
trade-off entre calidad de la solución y coste computacional en sistemas
multi-agente para generación de código.

## 1.4. Estructura del documento

El capítulo 2 revisa el estado del arte y el marco teórico, desde los fundamentos
de los sistemas multi-agente clásicos hasta los frameworks actuales basados en LLM,
la generación automática de código y los benchmarks de evaluación disponibles. El
capítulo 3 formaliza los objetivos e hipótesis del trabajo. El capítulo 4 describe
la metodología: criterios de diseño, protocolo de evaluación y configuración
experimental. El capítulo 5 detalla el desarrollo e implementación del sistema,
incluyendo la definición de roles, el grafo de estado y las herramientas disponibles
para cada agente. El capítulo 6 presenta los experimentos realizados. El capítulo 7
analiza los resultados y discute sus implicaciones respecto a las hipótesis
formuladas. El capítulo 8 recoge las conclusiones y propone líneas de investigación
futura.
