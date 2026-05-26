# Resumen Ejecutivo — Entrega Final

**Título:** Orquestación de Equipos de Agentes LLM con Roles Especializados para Resolución Colaborativa de Tareas Complejas de Ingeniería de Software.
**Autor:** Nicolás Timoneda. **Titulación:** Grado en Computación e IA, UAX. **Curso 2025-2026.**

## Planteamiento

¿Mejora un sistema multi-agente con roles especializados a un LLM monolítico en generación automática de código, y a qué coste? El TFG diseña, implementa y evalúa tres configuraciones incrementales sobre HumanEval (164 problemas) con el modelo Qwen 2.5 Coder 7B servido localmente en Ollama. La pregunta se formula de forma neutra: la literatura más visible (ChatDev, MetaGPT) presupone una respuesta positiva, pero trabajos críticos recientes (Chen et al., 2024; Olausson et al., 2024; Huang et al., 2024) sugieren que la mejora no es automática.

## Configuraciones

1. **Baseline** monolítico: una sola llamada al LLM.
2. **Sequential**: pipeline de cinco roles (Product Manager → Arquitecto → Developer → QA Tester → Code Reviewer) con estado compartido tipado en LangGraph y comunicación por artefactos.
3. **Self-reflection (r=1)**: pipeline secuencial con bucle condicional Reviewer → Developer cuando el veredicto no aprueba.

Tres variantes de ablación (sin PM, sin Arquitecto, sin Reviewer) quedan implementadas y disponibles en el runner para una segunda pasada.

## Metodología

Tres réplicas por par (configuración, problema). Métricas: pass@1 con IC 95 % bootstrap, pass@3 (estimador insesgado de Chen et al., 2021), tasa media de tests, tokens y latencia. Contraste de hipótesis pareado mediante test de McNemar y análisis coste-calidad en la frontera de Pareto. Métrica original de **adherencia estructural** que operacionaliza la afirmación de que el protocolo por artefactos reduce alucinaciones de formato.

## Resultados principales

*Datos al cierre del documento, 1 195 / 1 476 ejecuciones (80,9 % de la matriz).*

| Configuración | n | pass@1 | IC 95 % | Tokens | Latencia (s) |
|---|---:|---:|---|---:|---:|
| Baseline | 492 | **80,08 %** | [76,4 ; 83,5] | 283 | 5,1 |
| Sequential | 492 | 58,33 % | [53,9 ; 62,6] | 11 614 | 396,4 |
| SR (r=1) | 211 | 67,30 % | [60,7 ; 73,5] | 13 719 | 386,8 |

**El experimento no respalda la hipótesis intuitiva** de que la especialización por roles mejora la corrección. El pipeline secuencial cae 21,8 puntos respecto al baseline (McNemar p < 0,0001, b = 128, c = 21) y SR_r1 cae 12,8 puntos. Las configuraciones multi-agente cuestan aproximadamente 40 veces más tokens y 77 veces más latencia para producir peor pass@1, y la frontera de Pareto coste-calidad la ocupa por completo el baseline. La adherencia estructural se mantiene cercana al 100 % en las tres configuraciones (100 / 100 / 99,53 %), lo que confirma la robustez del protocolo de formato pero no rescata la calidad del contenido generado.

## Discusión

Tres explicaciones plausibles, no excluyentes, son consistentes con los datos: (a) propagación de errores entre roles del pipeline; (b) sobrecarga del prompt de rol en un modelo de 7 B con ventana de contexto limitada; (c) inadecuación de HumanEval —funciones aisladas con docstring corto— como banco para evaluar pipelines multi-agente diseñados para coordinación entre fases. El resultado se inscribe en la línea crítica reciente que cuestiona que componer más llamadas a un LLM mejore, sin condiciones, el rendimiento agregado.

## Conclusiones y trabajo futuro

El sistema y su banco experimental quedan operativos, resumibles y reproducibles, con código y datos públicos en GitHub. La evidencia empírica controlada contra la hipótesis intuitiva de mejora multi-agente para el régimen HumanEval × modelo 7 B local es, junto con la métrica de adherencia estructural y el banco reproducible, una contribución del trabajo. Como líneas futuras se contemplan: escalado del modelo (Qwen 32 B vía Cerebras, modelos frontera vía API) para contrastar si el resultado se invierte con más capacidad; integración real de SWE-bench Lite mediante harness Docker; ejecución de las ablaciones ya implementadas; y comparativa heterogénea de LLMs por rol.
