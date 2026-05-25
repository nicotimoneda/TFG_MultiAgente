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
de pruebas, coste en tokens y latencia. Se incluyen además variantes
de ablación que cuantifican la contribución individual de cada rol
del pipeline.

El sistema se implementa en Python con el modelo Qwen 2.5 Coder de 7
mil millones de parámetros servido en local mediante Ollama, sobre un
equipo MacBook Air M2. El banco experimental es resumible y
fault-tolerant: cada ejecución individual se persiste atómicamente y
una corrida interrumpida se retoma con un único comando, sin pérdida
de progreso. Un pipeline de análisis post-hoc genera las figuras y
tablas de la memoria a partir de los CSV resultantes, incluyendo un
borrador automático de la narrativa de hallazgos.

Las contribuciones principales del trabajo son: una arquitectura
multi-agente con definición explícita de roles, estado compartido
tipado y flujo de control condicional; una métrica original de
adherencia estructural que operacionaliza la afirmación —frecuente en
la literatura pero raramente cuantificada— de que el protocolo de
comunicación basado en artefactos reduce la incidencia de
alucinaciones respecto a la conversación libre; un análisis empírico
del trade-off coste-calidad sobre el plano de Pareto entre
configuraciones; y la implementación completa publicada como
repositorio reproducible.

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
variants further quantify the individual contribution of each
pipeline role.

The system is implemented in Python with the 7-billion-parameter
Qwen 2.5 Coder model served locally via Ollama on a MacBook Air M2.
The experimental harness is resumable and fault-tolerant: each
individual execution is persisted atomically and an interrupted run
resumes with a single command, with no progress loss. A post-hoc
analysis pipeline generates the figures and tables of the memoria
from the resulting CSV files, including an automated draft of the
findings narrative.

The main contributions of this work are: a multi-agent architecture
with explicit role definition, typed shared state and conditional
control flow; a novel structural-adherence metric that operationalises
the claim —frequent in the literature but rarely quantified— that
artifact-based communication protocols reduce hallucination incidence
relative to free-form conversation; an empirical analysis of the
cost-quality trade-off along the Pareto front of configurations; and
the complete implementation released as a reproducible repository.

**Keywords:** multi-agent systems, large language models, automatic
code generation, LangGraph, HumanEval, pass@k, self-reflection,
automated software engineering.
