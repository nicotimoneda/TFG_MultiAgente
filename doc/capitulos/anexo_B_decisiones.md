# Anexo B: Decisiones técnicas

Este anexo presenta, en forma resumida, las decisiones técnicas que han
guiado el diseño y la implementación del sistema a lo largo de los siete
sprints que componen el desarrollo del TFG. Cada entrada documenta el
contexto en que surgió la decisión, las alternativas consideradas y la
justificación que llevó a la elección final.

El documento canónico, con cada entrada expandida con motivos, alternativas
evaluadas y matices de implementación, es `doc/decisiones.md` en la raíz
del repositorio. Este anexo es una versión sintética orientada a la lectura
del tribunal evaluador.

## B.1. Sprint S1 — Estado compartido como TypedDict

Se eligió `TypedDict` frente a `pydantic.BaseModel` para el estado
compartido `AgentState`. La validación en tiempo de ejecución que aporta
Pydantic no compensa el coste de serialización en cada transición del
grafo, dado que cada campo del estado tiene un único agente productor y la
condición de carrera lógica está descartada por diseño. LangGraph
serializa el estado entre nodos mediante sus mecanismos internos de
checkpointing y `TypedDict` es directamente compatible.

## B.2. Sprint S2 — QA Tester sin LLM y veredicto determinista

El agente QA Tester se mantiene como nodo del grafo aunque no invoca al
LLM. Su trabajo —ejecutar tests en el sandbox— es puramente determinista.
Mantenerlo como nodo explícito hace visible la etapa de verificación al
imprimir el grafo y preserva el contrato del `AgentState`. El veredicto
del Code Reviewer (APPROVE / REQUEST_CHANGES) se deriva deterministamente
de `test_results`, no se pide al LLM, para evitar la fragilidad observada
en modelos pequeños al emitir etiquetas literales dentro de texto libre.

## B.3. Sprint S3 — Temperatura asimétrica en self-reflection

El Developer reflexivo de la configuración con self-reflection opera con
temperatura 0.4, frente a 0.2 del resto de agentes. La asimetría es
deliberada: una diversidad mayor en la generación facilita la
convergencia del ciclo Reviewer → Developer al permitir explorar
soluciones alternativas en lugar de repetir variantes de la primera
propuesta.

## B.4. Sprint S4 — Runner resumible y atómico

El runner principal persiste el estado del experimento mediante escritura
atómica del fichero `progress.json` (escritura a `.json.tmp` + rename).
Cada ejecución individual se envuelve en `try/except` amplio que registra
el error en CSV con el campo `error` poblado, sin abortar la corrida
global. Esta política, contraria al fail-fast habitual, es necesaria para
corridas de miles de ejecuciones donde un fallo aislado no debe invalidar
el resto.

## B.5. Sprint S5 — Inferencia muestral, no determinista

Las llamadas al LLM se realizan sin fijar la `seed` de inferencia de
Ollama. El estimador pass@k de Chen et al. (2021) asume `n` generaciones
independientes por problema y forzar determinismo invalidaría su
definición. Las tres semillas del experimento indexan **réplicas del
experimento**, no semillas del modelo: cada réplica constituye una
muestra independiente de la distribución de salidas del pipeline.

## B.6. Sprint S6 — Backend local (Ollama) como referencia, Cerebras evaluado y descartado

El backend principal de inferencia es Ollama local con
`qwen2.5-coder:7b-instruct-q4_K_M`. La elección equilibra
reproducibilidad —cualquier investigador con el binario público puede
replicar la corrida— con eliminación del coste marginal por inferencia.

La alternativa de Cerebras Inference (Qwen-3 235B y Llama 3.1 8B en
tier gratuito) se evaluó empíricamente durante S6 y se descartó: bajo
la carga real de un pipeline con 5-6 llamadas concatenadas por
problema, el rate limit del tier público (≈1-2 req/min sostenido tras
errores 429) hace inviable cubrir la matriz experimental en plazo. Se
probaron pacing artificial de 2,5 s entre llamadas y workers en {1, 3}
sin que el límite cediera. El detalle cuantitativo de las pruebas está
en la entrada D1 del documento canónico `doc/decisiones.md`.

La variable `LLM_BACKEND` permite alternar a Cerebras desde código sin
tocar el pipeline, lo que mantiene abierta la opción para verificaciones
cruzadas sobre subconjuntos pequeños con un plan de pago futuro.

## B.7. Sprint S7 — Ablaciones, análisis y métrica de adherencia

Tres compromisos de la propuesta no estaban operacionalizados al cerrar
S6: el análisis de contribución por rol, una métrica que cuantifique la
reducción de alucinaciones del protocolo estructurado y un análisis
sistemático coste-vs-calidad. S7 añade las tres variantes de ablación
(`no_pm`, `no_architect`, `no_reviewer`), un pipeline post-hoc de figuras
y tablas con intervalos de confianza bootstrap y test pareado de McNemar,
y una métrica de adherencia estructural que cuenta ejecuciones con fence
malformado. La alternativa de ampliar el banco con SWE-bench se descarta
por presupuesto de cómputo, recogido en cap 3.4 y cap 8.5.

## B.8. Sprint S8 — Recorte de scope del barrido principal

Tras 18 horas de corrida con el backend local sobre el MacBook Air
M2, el ritmo observado sostenido (≈5,3 ejecuciones/hora para
secuencial, ≈12-16 min/ejecución para self-reflection) proyectaba la
matriz completa de 5 configuraciones (baseline + sequential + SR×3)
sobre 164 problemas × 3 semillas en aproximadamente 20 días, y en 30
días añadiendo las tres ablaciones. Con la entrega institucional el
8 de junio de 2026 y la imposibilidad de sostener al equipo bajo
carga térmica continua durante semanas (el MacBook Air M2 carece de
ventilación activa), el plazo no permitía la matriz completa.

**Decisión.** Mantener Ollama local como backend y recortar el
barrido principal a las **tres configuraciones más informativas**
para las hipótesis del estudio:

1. `baseline` (configuración 1, monolítica)
2. `sequential` (configuración 2, pipeline de 5 roles)
3. `self_reflection_r1` (configuración 3 con `max_revisions = 1`)

Las variantes `self_reflection_r2`, `self_reflection_r3` y las tres
ablaciones (`no_pm`, `no_architect`, `no_reviewer`) quedan
implementadas, testadas y disponibles en el runner como segunda
pasada, activables mediante variables de entorno
(`ENABLE_SR_R2`, `ENABLE_SR_R3`, `ABLATION_SUBSET_SIZE`) sin
modificar código.

**Justificación académica.** Las tres configuraciones conservadas
cubren las tres hipótesis del estudio: H1 (especialización) se
contrasta con baseline vs sequential; H2 (auto-revisión) con
sequential vs SR_r1; H3 (trade-off coste-calidad) con los tres
puntos en la frontera de Pareto. SR_r2 y SR_r3 son experimentos
de hiperparámetro que no son necesarios para contrastar H2. Las
ablaciones quedan para una segunda corrida.

**Coste.** Cero coste económico (Ollama local). El equipo opera bajo
throttling térmico moderado durante ~9 días, reversible y dentro de
tolerancias del fabricante. La cardinalidad efectiva del experimento
queda en 3 × 164 × 3 = **1 476 ejecuciones**, todas reportadas en el
capítulo 7.

El detalle completo de esta decisión, incluyendo las pruebas
empíricas de Cerebras descartadas, está en la entrada S8 de
`doc/decisiones.md`.

## B.9. Notas sobre el feedback del tutor

Dos observaciones del tutor han condicionado la forma final del
documento: la bibliografía no se duplica al final de cada capítulo; toda
referencia se resuelve contra el [Anexo de Bibliografía](../referencias/bibliografia.md)
único, recogido al cierre del documento. La densidad visual se ha
incrementado significativamente a partir del capítulo 5, con tablas y
figuras explícitamente numeradas en cada sección que introduce un
concepto técnico o un resultado numérico.

---

Para el detalle completo de cada decisión, con los motivos descartados y
las implicaciones operativas, consultar `doc/decisiones.md`.
