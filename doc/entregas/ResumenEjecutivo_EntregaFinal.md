# Resumen Ejecutivo — Entrega Final

## Título y autor

Proyecto Fin de Grado: **Orquestación de Equipos de Agentes LLM con Roles
Especializados para Resolución Colaborativa de Tareas Complejas de Ingeniería
de Software**.

Autor: Nicolás Timoneda. Grado en Ingeniería en Computación e Inteligencia
Artificial, Universidad Alfonso X el Sabio (UAX).

## Planteamiento y motivación

El desarrollo de software profesional es por naturaleza un proceso multi-rol:
analistas, arquitectos, desarrolladores y revisores se reparten el trabajo
porque ningún profesional individual ejerce todas esas funciones con la misma
eficacia simultáneamente. Los modelos de lenguaje de gran tamaño han abierto
la posibilidad de automatizar parte de ese ciclo, pero un agente único
presenta limitaciones concretas: ventana de contexto finita, ausencia de
mecanismos internos de verificación y dificultad para detectar errores
semánticos sin perspectiva externa. Frameworks recientes —MetaGPT, ChatDev,
AutoGen— sugieren que distribuir el trabajo entre agentes especializados
produce resultados más consistentes, aunque su evaluación empírica
sistemática sobre benchmarks estandarizados sigue siendo escasa.

La pregunta central que articula el trabajo es si un sistema multi-agente
con roles especializados y orquestación basada en grafos de estado mejora a
un LLM monolítico en generación automática de código y bajo qué condiciones
ese beneficio justifica el coste adicional en tokens y latencia.

## Objetivos

El objetivo general es diseñar, implementar y evaluar empíricamente un
sistema multi-agente que resuelva tareas de generación automática de código
mediante agentes especializados, orquestados con un grafo de estado en
LangGraph y comparados de forma controlada contra un LLM monolítico. De
ese objetivo se derivan siete específicos, entre los que destacan el
diseño de una arquitectura con cinco roles —Product Manager, Arquitecto,
Developer, QA Tester y Code Reviewer—, la implementación de un mecanismo
de auto-revisión iterativa, la evaluación cuantitativa sobre benchmarks
públicos con análisis estadístico, y la documentación reproducible del
sistema. El trabajo formula además tres hipótesis falsables sobre
especialización, auto-revisión y trade-off calidad/coste, que se
contrastan mediante test de McNemar pareado e intervalos de confianza
bootstrap al 95%.

## Metodología y diseño experimental

El estudio sigue un diseño empírico-comparativo con tres configuraciones
incrementales: baseline monolítico, pipeline secuencial de cinco roles y
pipeline con bucle de self-reflection entre Reviewer y Developer
parametrizado por un número máximo de iteraciones. Esa tercera
configuración se evalúa con tres ajustes del parámetro —max_revisions
∈ {1, 2, 3}— para caracterizar la curva calidad/coste del ciclo
iterativo. Se añaden además tres ablaciones de rol —no_pm, no_architect
y no_reviewer— que aíslan empíricamente la contribución de cada agente
al rendimiento global del pipeline.

El benchmark primario es HumanEval (164 problemas); MBPP queda cacheado
para ejecución offline y ClassEval y SWE-bench Lite están preparados a
nivel de scaffolding como extensión natural. Cada problema se ejecuta
con tres semillas distintas para soportar el cómputo de pass@k mediante
el estimador insesgado de Chen et al. (2021). Las métricas reportadas
son pass@1 con intervalo de confianza bootstrap percentil al 95%,
pass@3, average test pass rate, tokens consumidos y latencia. La
comparación entre configuraciones se realiza con test de McNemar pareado
sobre los resultados pass/fail por problema.

El stack es Python 3.11 con LangGraph y LangChain. El modelo utilizado
en todos los roles es Qwen 2.5 Coder 7B Instruct con cuantización
Q4_K_M, servido localmente mediante Ollama sobre un MacBook Air M2 de 16
GB de memoria unificada. Usar el mismo modelo en todas las
configuraciones permite atribuir las diferencias observadas
exclusivamente al diseño del pipeline. La generación de código se ejecuta
en un sandbox de subproceso aislado, y el banco experimental
(`experiments/run_experiments.py`) implementa una matriz cartesiana
resumible con persistencia atómica del progreso, de modo que cualquier
interrupción se recupera reejecutando el mismo comando.

## Resultados principales

La corrida experimental se encuentra en ejecución al cierre del
documento. Los datos al cierre del documento muestran, para el baseline
monolítico sobre 492 ejecuciones (164 problemas × 3 semillas), un pass@1
cercano al 83% sobre HumanEval y una adherencia estructural del 100% al
protocolo de artefactos. Esa adherencia establece el listón superior con
el que las configuraciones más complejas se contrastarán y permitirá
discriminar si el protocolo estructurado mantiene su consistencia al
añadir más agentes al pipeline. El pipeline de análisis automático
(`experiments/analyze_results.py`) regenera de forma incremental, sobre
los CSV de `experiments/results/`, las tablas y figuras del capítulo 7
—comparativas pass@1 con CI, frontera de Pareto coste-calidad,
distribuciones de latencia y de revisiones— a medida que la corrida
avanza. La comparación cuantitativa completa entre baseline, secuencial,
las tres variantes de self-reflection y las tres ablaciones se reporta
en el capítulo 7 con los datos finales del banco.

## Conclusiones y trabajo futuro

El trabajo entrega tres piezas operativas que sobreviven al ciclo del
TFG: el sistema multi-agente implementado y documentado, un banco
experimental reproducible y resumible, y un pipeline de análisis
automático con figuras y tablas listas para publicación. Sobre esa
infraestructura se contrastan empíricamente las tres hipótesis del
estudio.

Las limitaciones reconocidas —modelo de 7 B parámetros, benchmark de
funciones aisladas y uso del mismo modelo en todos los roles— delimitan
el alcance de las conclusiones y abren líneas de trabajo futuro
concretas: escalado del modelo a configuraciones frontera, integración
de SWE-bench Lite con infraestructura Docker controlada, incorporación
de Dynamic Task Decomposition con un agente Planner, comparativa
heterogénea de backends por rol y extensión del estado compartido a
proyectos multi-fichero. Independientemente del veredicto cuantitativo
final, la infraestructura construida permite articular esas extensiones
sin reconstruir la base desde cero.
