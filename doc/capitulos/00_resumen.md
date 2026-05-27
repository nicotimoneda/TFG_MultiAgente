# Resumen

Este Trabajo Fin de Grado diseña, implementa y evalúa empíricamente un
sistema multi-agente basado en modelos de lenguaje de gran tamaño para
la resolución colaborativa de tareas de generación automática de código
Python. El sistema orquesta cinco agentes especializados —Product
Manager, Arquitecto, Developer, QA Tester y Code Reviewer— mediante un
grafo de estado tipado implementado con LangGraph, con comunicación
estructurada a través de artefactos en lugar de conversación libre.

Se comparan tres configuraciones incrementales: un baseline monolítico
con un único LLM, un pipeline secuencial de cinco roles, y un pipeline
con un ciclo iterativo de auto-revisión entre el Code Reviewer y el
Developer. La evaluación se realiza sobre el benchmark HumanEval (164
problemas) con tres réplicas indexadas por semilla pseudoaleatoria, lo
que permite contrastar las hipótesis del estudio mediante el test
pareado de McNemar y mediante intervalos de confianza al 95 % obtenidos
por bootstrap percentil. Las métricas reportadas son pass@1, pass@3
(estimador insesgado de Chen et al., 2021), tasa media de superación
de pruebas, coste en tokens y latencia. Variantes de ablación de rol
quedan implementadas en el runner para una segunda pasada.

El sistema se implementa en Python con el modelo Qwen 2.5 Coder de 7
mil millones de parámetros servido en local mediante Ollama, sobre un
equipo MacBook Air M2. El banco experimental es resumible y
fault-tolerant: cada ejecución individual se persiste atómicamente y
una corrida interrumpida se retoma con un único comando, sin pérdida
de progreso. Un pipeline de análisis post-hoc genera las figuras y
tablas de la memoria a partir de los CSV resultantes, incluyendo un
borrador automático de la narrativa de hallazgos.

**Resultado principal.** El experimento no respalda la hipótesis
intuitiva de que la especialización por roles mejora la corrección.
El baseline alcanza 80,08 % de pass@1; el pipeline secuencial cae a
58,33 % (McNemar p < 0,0001) y la configuración con self-reflection a
64,84 % (p < 0,0001 frente al baseline; p = 0,0066 frente al
secuencial, lo que respalda H2 para `r = 1`). Las configuraciones
multi-agente consumen alrededor de 40 veces más tokens y 77 veces más
latencia por problema para producir peor pass@1, y la frontera de
Pareto coste-calidad la ocupa por completo el baseline. La adherencia
estructural al protocolo, en cambio, se mantiene cercana al 100 % en
las tres configuraciones, lo que confirma la robustez del formato
pero no rescata la calidad del contenido generado. El trabajo
discute tres explicaciones plausibles para el resultado —propagación
de errores entre roles, sobrecarga del prompt de rol en un modelo de
7 B, e inadecuación de HumanEval como banco para evaluar pipelines
multi-agente— y los enmarca en una línea crítica reciente de la
literatura (Chen et al., 2024; Olausson et al., 2024; Huang et al.,
2024) que cuestiona que componer más llamadas a un LLM mejore, sin
condiciones, el rendimiento agregado.

Las contribuciones principales del trabajo son: una arquitectura
multi-agente con definición explícita de roles, estado compartido
tipado y flujo de control condicional, completamente implementada y
documentada; una métrica original de adherencia estructural que
operacionaliza la afirmación —frecuente en la literatura pero
raramente cuantificada— de que el protocolo de comunicación basado
en artefactos reduce la incidencia de alucinaciones de formato; un
análisis empírico del trade-off coste-calidad sobre el plano de
Pareto entre configuraciones; evidencia empírica controlada contra
la hipótesis intuitiva de mejora multi-agente para el régimen
HumanEval × modelo 7 B local; y la implementación completa publicada
como repositorio reproducible.

**Palabras clave:** sistemas multi-agente, modelos de lenguaje de gran
tamaño, generación automática de código, LangGraph, HumanEval, pass@k,
self-reflection, ingeniería de software automatizada.

---

# Abstract

This Bachelor's Thesis designs, implements and empirically evaluates a
multi-agent system based on large language models for the collaborative
resolution of automatic Python code generation tasks. The system
orchestrates five specialised agents —Product Manager, Architect,
Developer, QA Tester and Code Reviewer— through a typed state graph
implemented with LangGraph, with structured communication through
artifacts rather than free-form conversation.

Three incremental configurations are compared: a monolithic baseline
with a single LLM, a sequential five-role pipeline, and a pipeline
with an iterative self-revision loop between the Code Reviewer and
the Developer. Evaluation is conducted on the HumanEval benchmark
(164 problems) with three replicas indexed by pseudorandom seed,
allowing for hypothesis testing via the paired McNemar test and 95 %
confidence intervals via percentile bootstrap. Reported metrics
include pass@1, pass@3 (the unbiased estimator from Chen et al.,
2021), average test pass rate, token cost and latency. Role-ablation
variants remain implemented in the runner for a second pass.

The system is implemented in Python with the 7-billion-parameter
Qwen 2.5 Coder model served locally via Ollama on a MacBook Air M2.
The experimental harness is resumable and fault-tolerant: each
individual execution is persisted atomically and an interrupted run
resumes with a single command, with no progress loss. A post-hoc
analysis pipeline generates the figures and tables of the memoria
from the resulting CSV files, including an automated draft of the
findings narrative.

**Main result.** The experiment does not support the intuitive
hypothesis that role specialisation improves correctness. The
baseline reaches 80.08 % pass@1; the sequential pipeline drops to
58.33 % (McNemar p < 0.0001) and the self-reflection configuration
to 64.84 % (p < 0.0001 against the baseline; p = 0.0066 against the
sequential pipeline, supporting H2 for `r = 1`). Multi-agent
configurations consume roughly 40 times more tokens and 77 times
more latency per problem to produce worse pass@1, and the
cost-quality Pareto front is occupied entirely by the baseline.
Structural adherence to the artifact protocol remains near 100 %
across all three configurations, confirming format robustness but
not rescuing the correctness of the generated content. The work
discusses three plausible explanations —error propagation across
roles, overhead of role prompts on a 7 B model, and HumanEval being
an inadequate benchmark for evaluating multi-agent pipelines— and
positions the result within a recent critical strand of the
literature (Chen et al., 2024; Olausson et al., 2024; Huang et al.,
2024) that questions whether composing more LLM calls unconditionally
improves aggregate performance.

The main contributions of this work are: a multi-agent architecture
with explicit role definition, typed shared state and conditional
control flow, fully implemented and documented; a novel
structural-adherence metric that operationalises the claim —frequent
in the literature but rarely quantified— that artifact-based
communication protocols reduce format hallucination incidence; an
empirical analysis of the cost-quality trade-off along the Pareto
front of configurations; controlled empirical evidence against the
intuitive multi-agent improvement hypothesis for the HumanEval ×
local 7 B model regime; and the complete implementation released as a
reproducible repository.

**Keywords:** multi-agent systems, large language models, automatic
code generation, LangGraph, HumanEval, pass@k, self-reflection,
automated software engineering.
