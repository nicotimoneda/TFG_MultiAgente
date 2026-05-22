# Bitácora de decisiones técnicas

Este documento recoge las decisiones técnicas tomadas durante el
desarrollo del TFG que no son derivables de los capítulos principales
por sí solos. Para cada decisión se documenta el contexto, las
alternativas evaluadas, el criterio de selección, la evidencia que la
respalda y sus implicaciones.

El objetivo es proporcionar trazabilidad de las decisiones de ingeniería
que afectan a la validez y reproducibilidad del estudio experimental.

---

## D1 — Selección iterativa del backend de inferencia

**Contexto.** El sistema multiagente requiere un proveedor de inferencia
LLM con las siguientes propiedades: (a) compatibilidad con la API
OpenAI para integración directa con `langchain-openai`; (b) acceso a un
modelo de calidad suficiente para tareas de generación de código; (c)
coste compatible con un trabajo académico (gratuito o de bajo coste);
(d) capacidad de absorber el volumen de los experimentos planificados
(≈10 k–18 k tokens por ejecución de pipeline multiagente).

**Recorrido de alternativas evaluadas.**

| # | Proveedor | Modelo | Motivo de descarte |
|---|-----------|--------|--------------------|
| 1 | HuggingFace Inference API | Llama 3.1 70B Instruct | Cuotas muy restrictivas en el plan gratuito; cold starts impredecibles que rompen la temporización de los experimentos. |
| 2 | Groq | Llama 3.3 70B Versatile | Rate limits agresivos por minuto que obligaban a serializar todas las ejecuciones, multiplicando la duración del experimento. |
| 3 | Together AI | Qwen 2.5 Coder 32B | Requiere registro de tarjeta de crédito; no compatible con la restricción de coste cero del trabajo. |
| 4 | Cerebras Inference | Qwen-3 235B A22B Instruct (2507) | **Adoptado provisionalmente** (commit `34404fe`). Plan gratuito sin tarjeta, OpenAI-compatible, modelo grande de alta capacidad. Tras la corrida parcial se observa que el límite de tokens por minuto (TPM) del plan gratuito se satura en las configuraciones `sequential` y `self_reflection`, donde cada ejecución consume entre 10 k y 18 k tokens. Las reintentos automáticos por error 429 introducían latencias artificiales no representativas del coste real del pipeline. |
| 5 | Ollama (servidor local) | Qwen 2.5 Coder 7B Instruct, cuantizado Q4_K_M | **Decisión final** (commit `7f99c52`). |

**Justificación de la decisión final.**

La transición a un backend local elimina la única variable no controlada
del experimento: la disponibilidad y latencia variable del proveedor
remoto. La reproducibilidad del estudio aumenta porque otro investigador
puede replicar el experimento con el mismo binario del modelo
(descargable desde el registro público de Ollama) sin depender de la
política de cuotas de un proveedor comercial. El coste marginal de cada
ejecución pasa a ser únicamente energético, lo que permite ampliar el
número de seeds o problemas sin restricciones presupuestarias.

La contrapartida es una reducción significativa de la capacidad del
modelo (de 235 B a 7 B parámetros). Esta reducción se justifica
empíricamente en la decisión D2.

**Implicaciones.**

- El factory de clientes LLM (`src/llm/client_factory.py`) se generaliza
  para seleccionar el backend mediante la variable de entorno
  `LLM_BACKEND ∈ {cerebras, ollama}`. Esto preserva la posibilidad de
  reejecutar con Cerebras como verificación cruzada en un subconjunto.
- El capítulo de resultados debe reportar claramente el modelo
  efectivamente utilizado en la corrida final, no el modelo
  inicialmente planificado.

---

## D2 — Selección del modelo local

**Contexto.** Tras la decisión D1 (backend local vía Ollama), el modelo
a desplegar queda restringido por el hardware disponible: MacBook Air
M2, 8 núcleos de CPU (4 P + 4 E), 10 núcleos de GPU, **16 GB de memoria
unificada**. Descontada la reserva de macOS y el resto del entorno de
desarrollo, el techo práctico para modelo más KV-cache es de
aproximadamente 10–11 GB.

**Alternativas evaluadas.**

| Modelo | Tamaño en disco | RAM en uso | Veredicto |
|---|---|---|---|
| `qwen2.5-coder:14b-instruct-q4_K_M` | 9 GB | ~11–12 GB | Al límite; arriesga *swap* con contextos largos del pipeline multiagente. |
| `deepseek-coder-v2:16b-lite-q4` | 9 GB | ~11 GB | Idem. Además rompe la coherencia de familia con el modelo de referencia. |
| `llama3.1:8b-instruct-q4` | 4.9 GB | ~7 GB | Modelo generalista; rendimiento inferior en código según *leaderboards* públicos. Conservado como *fallback*. |
| `qwen2.5-coder:7b-instruct-q4_K_M` | 4.7 GB | ~6–7 GB | **Seleccionado.** |

**Justificación.**

Tres criterios convergen en `qwen2.5-coder:7b-instruct-q4_K_M`:

1. **Coherencia de familia con la fase Cerebras.** El modelo Qwen-3 235B
   empleado en la fase previa pertenece a la misma familia de
   arquitectura y tokenización. Esto permite encuadrar la transición
   como un *downscale* dentro de la misma familia de modelos, en lugar
   de un cambio de proveedor cualitativamente distinto.
2. **Holgura de memoria.** El consumo en uso (~6–7 GB) deja margen para
   ejecutar el servidor de Ollama, el entorno Python con los procesos
   del experimento, y un navegador con el dashboard de monitorización en
   tiempo real sin entrar en *swap*.
3. **Especialización en código.** Qwen 2.5 Coder está entrenado
   específicamente sobre corpus de programación, lo que es relevante
   para la tarea evaluada (HumanEval).

**Evidencia empírica.**

En la calibración inicial (`experiments/quick_check.py`, 10 problemas
HumanEval, semilla 42, configuración `baseline`) el modelo obtiene
**8/10 problemas resueltos correctamente (80 %)** con una latencia
media de ≈6.4 s por problema y un consumo medio de ≈290 tokens. Este
rendimiento es comparable al obtenido con Qwen-3 235B en la fase
Cerebras sobre el mismo subconjunto, lo que aporta evidencia de que la
reducción de escala no degrada significativamente la métrica de interés
en el régimen de problemas considerado.

**Implicaciones.**

- El modelo seleccionado se documenta como variable controlada del
  experimento. La extrapolación de los resultados a modelos de mayor
  capacidad queda fuera del alcance, pero la comparación entre
  configuraciones (baseline vs. sequential vs. self_reflection) sigue
  siendo válida porque las tres se ejecutan con el mismo modelo.
- El uso de un modelo más pequeño obliga a adaptar algunos
  componentes del pipeline (véanse decisiones D3 y D4).

---

## D3 — Veredicto del agente Reviewer derivado de los tests

**Commit:** `7f99c52`

**Contexto.** En el diseño original, el agente Reviewer recibe el
código generado, los resultados de los tests y el documento de diseño,
y produce una respuesta cuya primera línea debe ser exactamente
`VERDICT: APPROVE` o `VERDICT: REQUEST_CHANGES`. Un parser estricto
valida este formato. Esta línea es la señal que utiliza el *router* de
la configuración `self_reflection` para decidir si el grafo termina o
vuelve al Developer para revisión.

**Problema observado.** Con el modelo `qwen2.5-coder:7b-instruct-q4_K_M`,
el Reviewer ignora sistemáticamente la restricción de formato y comienza
su respuesta con un encabezado Markdown (`### Review of …`) o, en algunos
casos, escribe un review del documento de diseño en lugar del código.
Esta es una limitación conocida de los modelos cuantizados de menor
capacidad: el *instruction-following* con restricciones de formato
estrictas es significativamente menos fiable que en modelos de tamaño
superior.

**Alternativas evaluadas.**

1. **Refuerzo del prompt** con instrucciones más explícitas
   (`"OUTPUT FORMAT — your response MUST begin with exactly: …"`) y
   ejemplos *few-shot*. Aumenta el coste por llamada y la fiabilidad
   sigue por debajo del 100 %.
2. **Parser tolerante** que busque el token `VERDICT:` en cualquier
   línea, no solo la primera. Resuelve el caso del preámbulo Markdown,
   pero no el caso en que el modelo no emite el token en absoluto.
3. **Derivación determinista del veredicto a partir de los resultados
   de los tests.** Adoptada.

**Justificación.** El veredicto APPROVE/REQUEST_CHANGES es, por
construcción, una función objetiva del resultado de los tests:
APPROVE si y solo si todos los tests pasan. Pedirle al LLM que
recompute esta función es semánticamente redundante y operativamente
frágil con modelos pequeños. La función real del Reviewer dentro del
pipeline es producir el **comentario cualitativo** (issues detectadas,
propuestas de corrección) que alimenta el siguiente ciclo del Developer
en `self_reflection`. Esa función cualitativa sí requiere al LLM y se
preserva intacta.

**Implementación.** El método `run()` del `CodeReviewerAgent`:

1. Llama al LLM solicitando únicamente comentario estructurado (issues +
   fixes), sin emitir línea de veredicto.
2. Deriva `verdict ∈ {APPROVE, REQUEST_CHANGES}` aplicando
   `all(tests_pass)` sobre `state["test_results"]`.
3. Antepone la línea canónica de veredicto al comentario antes de
   escribirlo en `state["review_comments"]`, garantizando que el
   *router* de `self_reflection` siga operando sin modificación.

**Implicaciones para la interpretación del experimento.**

- La métrica pass@1 no se ve afectada: los tests se ejecutan
  determinísticamente en el sandbox antes de que el Reviewer intervenga.
- La calidad del bucle de `self_reflection` depende ahora de la calidad
  del *comentario* del Reviewer, no de su capacidad de emitir un
  veredicto. Esto enfoca correctamente el papel del LLM en el pipeline.
- En la discusión del capítulo de resultados se debe reportar
  explícitamente esta decisión como una **adaptación metodológica** y
  no como un cambio de definición del experimento.

---

## D4 — Concatenación del prompt original de HumanEval al artefacto de código

**Contexto.** Tras la adopción del modelo local, las configuraciones
`sequential` y `self_reflection` arrojaban un pass@1 de 0 % de manera
sistemática mientras `baseline` obtenía un 80 %, a pesar de que el
código generado por el Developer era inspeccionablemente correcto en
varios casos.

**Diagnóstico.** El Developer del pipeline produce únicamente el cuerpo
de la función (instruido por su *system prompt* a evitar imports
superfluos). Cuando la firma de la función contiene anotaciones del
módulo `typing` —por ejemplo `List[int]`— y la salida del Developer
no incluye `from typing import List`, la ejecución en el sandbox falla
con `NameError` en cada caso de prueba, lo que se contabiliza como 0 %
de tests superados aunque la lógica del cuerpo de la función sea
correcta. El `baseline` no presentaba este patrón porque el modelo, ante
un prompt mínimo y monolítico, tiende a incluir los imports
explícitamente.

**Solución adoptada.** Concatenar el *prompt* original del problema
HumanEval (que ya incluye los imports y la firma) al artefacto de
código generado **antes de ejecutarlo en el sandbox**. La doble
definición de la función no es un problema: Python utiliza la última
definición, que es la generada por el agente.

**Justificación metodológica.** Esta es exactamente la convención
canónica de evaluación de HumanEval establecida por Chen et al. (2021,
§2.2): pass@k se computa sobre `prompt + completion`, no sobre
`completion` aislada. Los proveedores del benchmark distribuyen el
problema dividido en dos campos precisamente con este uso en mente. El
patrón previo a esta decisión —ejecutar solo el `completion`—
representaba en realidad una desviación involuntaria del protocolo
estándar.

**Verificación.** Tras el cambio, una ejecución aislada de HE/5 con
configuración `sequential` y el mismo modelo pasa a obtener 1.0 de
test pass rate, frente al 0.0 anterior. La calibración completa
(`experiments/quick_check.py` con `LLM_BACKEND=ollama`) confirma el
patrón sobre los 10 problemas seleccionados.

**Implicaciones.**

- El cambio se aplica en tres puntos: `src/agents/roles/qa_tester.py`
  (configuraciones `sequential` y `self_reflection`),
  `src/evaluation/runner.py` y `experiments/run_experiments.py` (rama
  baseline, que no pasa por el QA Tester).
- La descripción del protocolo de evaluación en el capítulo 4 debe
  explicitar esta convención, citando a Chen et al. (2021).

---

## D5 — Reducción del alcance experimental

**Contexto.** El diseño experimental inicial contemplaba 164 problemas
de HumanEval × 200 problemas seleccionados de MBPP × 5 configuraciones
× 5 *seeds* ≈ 9 100 ejecuciones. Tras la transición al modelo local y
considerando la latencia observada (≈6 s por ejecución `baseline`,
≈200 s por ejecución `sequential`), el coste temporal total de la
corrida completa excedería ampliamente la fecha de entrega.

**Decisión.** Se reduce el alcance a:

- **Benchmark único:** HumanEval (164 problemas). Se descarta MBPP.
- **Semillas:** {42, 123, 456} (3 en lugar de 5).
- **Configuraciones:** las 5 originales se mantienen (baseline,
  sequential, self_reflection × {1, 2, 3} revisiones).

Total revisado: 164 × 5 × 3 ≈ 2 460 ejecuciones.

**Justificación.**

- **HumanEval ofrece variedad funcional suficiente** para distinguir
  entre configuraciones; el solapamiento de habilidades evaluadas entre
  HumanEval y MBPP es alto, por lo que añadir MBPP incrementa el coste
  temporal sin aportar nueva información cualitativa al estudio. La
  comparación con la literatura sigue siendo posible: HumanEval es el
  benchmark más reportado en el área.
- **Tres semillas son suficientes para estimar la variabilidad** del
  pipeline en el régimen experimental considerado. Para el test de
  McNemar pareado, el incremento de 3 a 5 semillas reduce el ancho del
  intervalo de confianza en un factor ≈√(3/5) ≈ 0.77, una mejora
  marginal frente al doblado del coste de cómputo.
- **Las cinco configuraciones se mantienen porque son el objeto central
  del estudio.** Reducir el número de configuraciones invalidaría la
  pregunta de investigación del TFG.

**Implicaciones.**

- El capítulo 4 (Metodología) debe actualizar el diseño experimental
  reportado: benchmark único, 3 semillas en lugar de 5.
- El protocolo estadístico del capítulo 4 no se modifica: el test de
  McNemar pareado y los intervalos de confianza por bootstrap siguen
  siendo aplicables.

---

## D6 — Naturaleza muestral de la inferencia y papel de las semillas

**Contexto.** Durante la calibración del modelo local se observa que
ejecuciones repetidas del mismo problema con la misma configuración y
la misma semilla del experimento producen resultados ligeramente
distintos: el modelo puede resolver un problema en una ejecución y
fallarlo en otra. Esto se manifestó como una aparente regresión al
relanzar `quick_check.py`: HumanEval/5 con `sequential` pasó en una
corrida y falló en la siguiente, sin haber cambios en el código entre
ambas.

**Diagnóstico.** El campo `seed` registrado en los CSV de resultados
controla el generador pseudoaleatorio de Python (`random.seed(seed)`)
pero no se propaga al servidor de inferencia. Sin pasar explícitamente
un parámetro `seed` por petición a la API de Ollama, el muestreo
utiliza una semilla derivada del tiempo del sistema, lo que hace que
las llamadas al LLM sean no deterministas incluso a temperatura baja.
La fluctuación observada entre ejecuciones se sitúa en el orden de
unas decenas de tokens y, en problemas que se encuentran en la
frontera de capacidad del modelo de 7 B, basta para alternar entre
solución correcta e incorrecta.

**Alternativas evaluadas.**

1. **Fijar `seed` en cada petición al LLM** (parámetro soportado por la
   API OpenAI-compatible de Ollama). Garantizaría reproducibilidad
   bit-a-bit entre ejecuciones idénticas del experimento.
2. **Asumir el carácter muestral de la inferencia y reportar
   resultados como distribución sobre varias semillas.** Adoptada.

**Justificación de la decisión.** La métrica pass@k de Chen et al.
(2021) está definida explícitamente sobre **muestras** del modelo:
asume `n` generaciones independientes por problema y estima la
probabilidad de éxito como esperanza. Forzar inferencia determinista
violaría la definición de la métrica y reduciría artificialmente la
varianza reportada. Las múltiples semillas del experimento son el
mecanismo correcto para estimar esa varianza, y el ancho de los
intervalos de confianza calculados por bootstrap captura
automáticamente la variabilidad de muestreo del modelo.

**Implicaciones.**

- El campo `seed` del CSV se mantiene, pero su descripción en el
  capítulo 4 debe matizar que indexa una **réplica del experimento**,
  no una semilla de muestreo del modelo.
- Las soluciones que el modelo resuelve de forma intermitente (es
  decir, con pass@1 estimado entre 0 y 1 a lo largo de las réplicas)
  son una observación legítima del experimento y aportan información:
  describen el régimen de problemas que se encuentra al borde de la
  capacidad del modelo de 7 B.
- En la discusión de resultados conviene reportar, además de la
  media, la desviación típica entre réplicas para cada par
  (configuración, problema). Esto permite distinguir entre problemas
  resueltos de forma robusta y problemas resueltos de forma
  intermitente.

---

## S7 — Ablaciones de rol, pipeline de análisis y métrica de adherencia

**Fecha.** 2026-05-22.

**Contexto.** Tras cerrar S1–S6, al cotejar el código con la **Propuesta
TFG** (PROJENER.AI, modelos propuestos 1–5) se identifican tres compromisos
de la propuesta que aún no estaban operacionalizados en el experimento:

1. *"Análisis de qué combinaciones de roles aportan más valor (¿el reviewer
   mejora la calidad? ¿el tester encuentra bugs que el developer no ve?)"*
   No existían configuraciones que **suprimieran un rol** para aislar su
   contribución; sólo se comparaba el pipeline completo contra el baseline
   monolítico.
2. *"Comparativa rigurosa cuantitativa de coste y calidad."* Hasta ahora
   los CSV se inspeccionaban a mano. Faltaba una capa que produjera figuras
   y tablas listas para la memoria.
3. *"Protocolo de comunicación que reduce >40% las alucinaciones vs.
   conversación libre."* No había métrica que cuantificara la adherencia
   al protocolo estructurado (fences `python`, PRD/design no vacíos).

A lo anterior se suma un feedback explícito del tutor: la memoria debe
contener **más figuras y tablas** y **no incluir referencias al final de
cada capítulo** — sólo una bibliografía global al cierre.

**Alternativas evaluadas.**

1. **Reabrir el alcance experimental:** añadir benchmarks adicionales
   (SWE-bench, DevBench, ClassEval, APPS) prometidos en la propuesta.
   Descartado por presupuesto de cómputo: el run completo de 7 B sobre
   HumanEval + MBPP ya es de orden de horas-CPU; añadir SWE-bench cambiaría
   la escala a días-GPU.
2. **Operacionalizar in situ los tres compromisos pendientes:**
   ablaciones de rol, pipeline de análisis y métrica de adherencia, dentro
   del banco actual (HumanEval + MBPP). Adoptada.

**Justificación de la decisión.** Cerrar los compromisos analíticos de la
propuesta sobre el banco ya cubierto es más informativo —y honesto— que
ampliar el alcance hacia benchmarks que no podrán ejecutarse antes de la
entrega. Las tres piezas son aditivas, no destructivas: cada una se puede
re-correr sobre los CSV existentes sin invalidar lo ya completado.

**Entregables.**

- `src/graph/ablation_graphs.py` — tres variantes nuevas (`no_pm`,
  `no_architect`, `no_reviewer`) que comparten el `AgentState`, los
  agentes y el sandbox del pipeline secuencial. Cada variante elimina
  exactamente un rol y semilla el campo del rol ausente con el artefacto
  inmediatamente anterior, manteniendo válidas las plantillas de prompt
  de los roles restantes.
- `experiments/run_experiments.py` — extensión del registro de
  configuraciones para incluir las tres ablaciones, con CSV propio por
  variante; el resto del runner (resume, dashboard, error log) funciona
  sin cambios.
- `experiments/cache/mbpp.json` — caché local de 200 problemas MBPP,
  pre-construida para que el run pueda reanudarse con `--benchmarks
  humaneval,mbpp` sin tocar HuggingFace.
- `experiments/analyze_results.py` — post-procesador de todos los
  `*_results.csv`. Calcula pass@1, pass@3 (estimador insesgado de Chen
  et al. 2021), tokens medios, latencia mediana y media, y número medio
  de revisiones. Genera cuatro figuras (`pass_at_1.png`,
  `cost_quality_pareto.png`, `latency_box.png`,
  `revision_distribution.png`) y tres tablas (`summary.md`,
  `summary.tex`, `per_benchmark.md`). Tolerante a datos parciales: se
  re-ejecuta mientras el experimento sigue en marcha.
- `experiments/adherence_metric.py` — métrica post-hoc de adherencia al
  protocolo estructurado: cuenta, por configuración, cuántas ejecuciones
  emitieron una advertencia de fence `python` ausente (señal directa
  de hallucinación de formato). Sólo lee el log y los CSV; no re-invoca
  el LLM.

**Implicaciones para la memoria.**

- El capítulo de **resultados** puede ahora ilustrarse con figuras y
  tablas reproducibles directamente desde `experiments/analyze_results.py`
  cada vez que avance el run. Esto responde al feedback del tutor sobre
  densidad visual.
- La sección de **discusión** puede contrastar el pipeline secuencial
  completo contra cada ablación para responder, con datos, a las
  preguntas de la propuesta sobre qué rol aporta más a la calidad final.
- La sección que defiende el protocolo estructurado de comunicación
  pasa a poder citar un número concreto de adherencia, en lugar de
  apoyarse sólo en argumento cualitativo.
- Las referencias bibliográficas no deben aparecer al final de cada
  capítulo; las citas seguirán claves dentro del cuerpo, pero la lista
  bibliográfica se consolida una sola vez al final del documento.

**Coste.** Cero impacto en el run en curso (las ablaciones se ejecutarán
en una segunda pasada tras el primer barrido sobre baseline + sequential +
self_reflection). Los analizadores son post-hoc y no consumen tokens.
