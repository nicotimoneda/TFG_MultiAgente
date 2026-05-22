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

## B.6. Sprint S6 — Backend local (Ollama) como referencia y Cerebras como verificación

El backend principal de inferencia es Ollama local con
`qwen2.5-coder:7b-instruct-q4_K_M`. La elección equilibra
reproducibilidad —cualquier investigador con el binario público puede
replicar la corrida— con eliminación del coste marginal por inferencia.
La variable `LLM_BACKEND` permite alternar a Cerebras (Qwen-3 235B) para
verificaciones cruzadas sobre subconjuntos sin tocar la lógica del
pipeline.

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

## B.8. Notas sobre el feedback del tutor

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
