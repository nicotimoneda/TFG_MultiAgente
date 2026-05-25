# Capítulo 1: Introducción

## 1.1. Motivación y contexto

El desarrollo de software profesional se reparte entre varios roles. Hay
quien define los requisitos, quien diseña la arquitectura, quien escribe
el código, quien lo prueba y quien lo revisa antes de integrarlo. Esta
división del trabajo no es una convención: es una forma de gestionar
complejidad. Ningún profesional puede mantener al mismo tiempo el detalle
de los requisitos, la coherencia del diseño, la corrección del código y
la cobertura de las pruebas; cuando se intenta, suelen aparecer errores
precisamente en las interfaces entre fases.

Los modelos de lenguaje de gran tamaño han cambiado lo que se puede
automatizar de ese ciclo. Hoy un modelo razonable es capaz de generar
una función Python plausible a partir de su docstring. Lo que no puede
hacer, al menos no de forma fiable, es saber si la función que ha
generado es correcta. La ventana de contexto pone un techo a cuánto
puede sostener en mente al mismo tiempo, y el modelo no tiene un
mecanismo interno que distinga "código que compila" de "código que
resuelve el problema". El resultado típico es una solución sintácticamente
válida que falla en casos límite que nadie le pidió comprobar.

La investigación reciente ha respondido a esto con sistemas multi-agente:
varios LLMs con prompts distintos, cada uno encargado de una fase del
trabajo, comunicándose a través de algún tipo de estado compartido.
ChatDev, MetaGPT y AutoGen son los ejemplos más visibles. Los tres
funcionan razonablemente bien en demostraciones, pero la evaluación
empírica controlada sobre benchmarks estándar es menos sistemática de
lo que cabría esperar: la mayoría de los trabajos publicados comparan
el sistema completo contra un baseline monolítico, sin aislar qué papel
juega cada rol concreto en el resultado final.

Este TFG construye un sistema multi-agente con cinco roles
(Product Manager, Arquitecto, Developer, QA Tester, Code Reviewer)
orquestado mediante un grafo de estado en LangGraph, y lo compara
contra un baseline monolítico sobre HumanEval con tres configuraciones
incrementales y tres variantes de ablación. El objetivo es responder
una pregunta concreta: cuándo, cuánto y a qué coste el diseño
multi-agente aporta valor real.

La tabla 1.1 anticipa, en forma sintética, qué limitaciones del LLM
monolítico el sistema propuesto pretende neutralizar y cómo. Los
capítulos siguientes desarrollan cada fila en detalle.

| Limitación del LLM monolítico | Contramedida del sistema multi-agente |
|---|---|
| Ventana de contexto finita; no puede sostener simultáneamente requisitos, diseño, código y pruebas | Distribución de responsabilidades entre cinco agentes con artefactos tipados en el estado compartido |
| No detecta errores en su propia salida | Agente QA Tester con ejecución determinista en sandbox y agente Code Reviewer con veredicto derivado de las pruebas |
| No tiene mecanismo de iteración fundamentada | Bucle condicional Reviewer → Developer parametrizado por `max_revisions` y guiado por evidencia externa |
| Comunicación en texto libre, sensible a alucinaciones | Protocolo estructurado por artefactos tipados (PRD, design, code, tests, review) |
| Difícil auditar el flujo de decisión | Grafo LangGraph explícito e inspeccionable que documenta cada transición |

Tabla 1.1. Limitaciones del LLM monolítico y contramedidas del sistema
propuesto.

## 1.2. Problema y pregunta de investigación

La pregunta central del trabajo es si un sistema multi-agente con
roles especializados y orquestación basada en grafos de estado mejora
a un LLM monolítico en generación automática de código, y bajo qué
condiciones ese beneficio compensa el coste adicional en tokens y
latencia.

Es una pregunta con dos partes. La técnica: ¿produce el sistema
multi-agente soluciones más correctas, medidas por pass@1 y pass@k
sobre HumanEval? La económica: ¿a qué coste se obtiene esa mejora, y
hay un umbral a partir del cual añadir complejidad deja de aportar?

Responder a las dos requiere un sistema realmente implementado y
evaluado, no un diseño en papel. Por eso el trabajo entrega tres
cosas concretas: el código en Python con LangGraph, los datos
experimentales en CSV reproducibles, y el análisis estadístico
formal sobre ellos.

## 1.3. Objetivos y contribuciones

El objetivo principal es diseñar, implementar y evaluar empíricamente
un sistema multi-agente basado en LLMs para generación automática de
código, comparándolo de forma controlada contra un baseline
monolítico.

De ahí se derivan tres líneas de trabajo: diseñar la arquitectura con
los cinco roles (Product Manager, Arquitecto, Developer, QA Tester y
Code Reviewer) sobre un grafo de estado en LangGraph; implementar el
sistema en Python de forma modular y reproducible; y evaluarlo
empíricamente sobre HumanEval midiendo pass@1, pass@3, coste en
tokens y latencia frente al baseline.

Las contribuciones concretas del trabajo son cinco:

1. La arquitectura multi-agente con estado compartido tipado y
   flujo de control condicional, publicada como repositorio
   reproducible en GitHub.
2. Las tres variantes de ablación de rol (`no_pm`,
   `no_architect`, `no_reviewer`), que aíslan empíricamente la
   contribución individual de cada agente al rendimiento global,
   un análisis que la literatura habitualmente omite.
3. La métrica de adherencia estructural, que cuantifica con un
   número la afirmación —común en la literatura pero rara vez
   medida— de que el protocolo por artefactos reduce las
   alucinaciones frente a la conversación libre.
4. El análisis del trade-off coste-calidad como frontera de
   Pareto sobre las configuraciones evaluadas.
5. El banco experimental resumible y el pipeline de análisis
   automático, que generan figuras, tablas y test pareados de
   McNemar directamente desde los CSV.

## 1.4. Estructura del documento

El capítulo 2 revisa el estado del arte: desde los fundamentos de los
sistemas multi-agente clásicos hasta los frameworks actuales basados
en LLM, los benchmarks de generación de código y los mecanismos de
orquestación basados en grafos. El capítulo 3 formaliza objetivos e
hipótesis. El capítulo 4 describe la metodología y el protocolo
experimental. El capítulo 5 detalla la implementación. El capítulo 6
presenta el banco experimental. El capítulo 7 analiza los resultados.
El capítulo 8 recoge las conclusiones y abre líneas de trabajo
futuro.

Seis anexos completan el documento. El Anexo A reproduce verbatim los
prompts de sistema y las plantillas de prompt de usuario de los seis
agentes. El Anexo B sintetiza las decisiones técnicas tomadas sprint
a sprint. El Anexo C consolida los comandos de reproducción del
experimento. El Anexo D recoge un glosario de acrónimos y términos.
El Anexo E aborda los aspectos éticos, legales y de sostenibilidad
del trabajo. El Anexo F contiene los agradecimientos.
