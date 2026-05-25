# Resumen Ejecutivo — Entrega Final

**Título:** Orquestación de Equipos de Agentes LLM con Roles Especializados para Resolución Colaborativa de Tareas Complejas de Ingeniería de Software.
**Autor:** Nicolás Timoneda. **Titulación:** Grado en Computación e IA, UAX. **Curso 2025-2026.**

## Planteamiento

¿Mejora un sistema multi-agente con roles especializados a un LLM monolítico en generación automática de código, y a qué coste? El TFG diseña, implementa y evalúa tres configuraciones incrementales sobre HumanEval (164 problemas) con el modelo Qwen 2.5 Coder 7B servido localmente en Ollama.

## Configuraciones

1. **Baseline** monolítico: una sola llamada al LLM.
2. **Sequential**: pipeline de cinco roles (Product Manager → Arquitecto → Developer → QA Tester → Code Reviewer) con estado compartido tipado en LangGraph y comunicación por artefactos.
3. **Self-reflection (r=1)**: pipeline secuencial con bucle condicional Reviewer → Developer cuando el veredicto no aprueba.

Se incluyen además tres variantes de ablación implementadas y testadas (sin PM, sin Arquitecto, sin Reviewer) que cuantifican la contribución individual de cada rol.

## Metodología

Tres réplicas por par (configuración, problema). Métricas: pass@1 con IC 95 % bootstrap, pass@3 (estimador insesgado de Chen et al., 2021), tasa media de tests, tokens y latencia. Contraste de hipótesis pareado mediante test de McNemar y análisis coste-calidad en la frontera de Pareto. Métrica original de **adherencia estructural** que operacionaliza la afirmación de que el protocolo por artefactos reduce alucinaciones.

## Resultados principales

*Datos al cierre del documento (corrida en curso).* Baseline cierra 492 ejecuciones con pass@1 ≈ 83 % e IC 95 % [78.7 %, 86.4 %], adherencia estructural del 100 % y consumo medio de 280 tokens y 5 s por problema. Sequential cierra 492 ejecuciones y SR_r1 avanza. Los números finales y los contrastes pareados se reportan en el capítulo 7, regenerados automáticamente desde los CSV mediante el pipeline de análisis.

## Conclusiones y trabajo futuro

El sistema y su banco experimental quedan operativos, resumibles y reproducibles, con código y datos públicos en GitHub. Las tres configuraciones permiten responder las tres hipótesis del estudio (especialización, auto-revisión y trade-off). Como líneas futuras se contemplan: integración real de SWE-bench mediante harness Docker, evaluación de ClassEval con el Developer adaptado (ya implementado), comparativa heterogénea de LLMs por rol, y escalado del modelo a tamaños mayores como verificación cruzada.
