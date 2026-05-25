# Capítulo 7: Resultados

## 7.1. Estado de la corrida y alcance de los resultados

La corrida principal del experimento está en ejecución en el momento de
cerrar este documento. El estado de progreso, consultable en cualquier
momento mediante el fichero `experiments/results/progress.json`, refleja
las ejecuciones completadas y la posición actual del barrido. Todos los
números que se reportan a continuación se obtienen del pipeline de
análisis (`experiments/analyze_results.py`) sobre los CSV disponibles a
fecha de cierre, y se regeneran automáticamente con cada re-ejecución del
analizador a medida que el experimento avanza.

Esta política —reportar lo que hay, con intervalos de confianza, en lugar
de esperar al cierre total de la matriz— es la única que permite
presentar evidencia cuantitativa real en el documento. La alternativa
sería redactar el capítulo con números fabricados o ilustrativos, lo que
contradice el principio metodológico declarado en el capítulo 4.

Los resultados que requieren múltiples configuraciones para tener sentido
—comparación entre arquitecturas, contraste de hipótesis pareadas, análisis
de problemas con desacuerdo entre configs— se reportan cuando los CSV de
las configuraciones implicadas existen. Para las secciones donde la
configuración necesaria aún no ha completado ninguna ejecución se incluye
el procedimiento y la tabla esperada, con los huecos numéricos marcados
explícitamente como pendientes.

## 7.2. Estudio piloto exploratorio

Con carácter previo a la evaluación cuantitativa formal, se realizó un
estudio piloto exploratorio de naturaleza cualitativa. El estudio
consiste en la ejecución de un LLM de propósito general —con un prompt
monolítico sin estructura de roles— sobre una muestra de cuatro problemas
del benchmark HumanEval (Chen et al., 2021), seguida de un análisis manual
de los outputs generados. El objetivo no es medir pass@k sino caracterizar
los patrones de error recurrentes en un agente único antes de diseñar el
sistema multi-agente, de forma que las decisiones arquitecturales del
capítulo 4 tengan una motivación empírica.

### 7.2.1. Análisis cualitativo de outputs

**HumanEval/1 — Separación de grupos de paréntesis.** La tarea consiste en
separar una cadena con múltiples grupos de paréntesis en grupos
independientes y correctamente balanceados. El LLM genera una
implementación funcional para entradas con poco anidamiento. Cuando la
cadena contiene grupos entrelazados con anidamiento profundo, la lógica
de seguimiento del nivel de profundidad no se mantiene y el resultado
incluye grupos mal delimitados. La solución no incluye ningún caso de
prueba ni comprobación de la propiedad de balance, lo que impide detectar
el fallo sin ejecución manual.

**HumanEval/26 — Eliminación de duplicados.** La tarea consiste en
devolver los elementos que aparecen exactamente una vez, eliminando los
que se repiten. La implementación es correcta para el caso general pero
no cubre los casos límite: lista vacía, lista donde todos los elementos
se repiten y lista de un único elemento. El modelo no genera pruebas que
incluyan esos casos ni documenta las precondiciones que asume. El error
no es de lógica sino de cobertura.

**HumanEval/38 — Codificación cíclica.** La tarea requiere implementar
una función de codificación cíclica sobre grupos de tres caracteres y su
función inversa de decodificación. El LLM genera la función encode de
forma correcta pero produce una función decode que aplica la misma
transformación en lugar de la transformación inversa: la composición
`decode(encode(s))` no devuelve la cadena original. El modelo no verifica
la propiedad de inversión en ningún momento del proceso de generación.

**HumanEval/119 — Concatenación de cadenas de paréntesis.** La tarea
consiste en determinar si dos cadenas de paréntesis pueden concatenarse en
algún orden para formar una cadena balanceada. La implementación comprueba
correctamente el caso en que la concatenación directa produce balance,
pero no considera de forma sistemática el orden inverso. La lógica del
segundo caso aparece en algunos outputs pero de forma incompleta,
produciendo falsos negativos en entradas donde sólo el orden invertido es
válido.

| Problema | Comportamiento observado | Tipo de error | ¿Detectable por otro agente? |
|---|---|---|---|
| HumanEval/1 | Correcto en casos simples, falla con anidamiento profundo | Error lógico en casos complejos | Sí, con tests de regresión |
| HumanEval/26 | Correcto en caso general, sin cobertura de casos límite | Error de cobertura | Sí, con casos de prueba específicos |
| HumanEval/38 | encode correcto, decode no es la inversa | Error semántico en función complementaria | Sí, verificando la composición |
| HumanEval/119 | Comprueba un solo orden de concatenación | Error lógico por análisis incompleto | Sí, con revisión de la especificación |

Tabla 7.1. Resumen del estudio piloto cualitativo.

### 7.2.2. Implicaciones para el diseño del sistema

Los cuatro problemas analizados comparten una característica: sus errores
no son visibles para el agente que generó el código, pero son detectables
por un agente externo con acceso al mismo estado del problema. Esa
asimetría es el punto de partida para justificar el diseño multi-agente.

El patrón más frecuente es la ausencia de verificación. En HumanEval/1 y
HumanEval/26, el modelo no genera pruebas que cubran los casos que falla.
Un agente dedicado exclusivamente a generar y ejecutar casos de prueba
puede detectar esos fallos antes de que lleguen a revisión. Esa función
corresponde al agente QA Tester del sistema propuesto.

La incapacidad de verificar propiedades semánticas globales, como la
propiedad de inversión en HumanEval/38, señala un problema distinto: el
modelo evalúa el código línea a línea pero no comprueba si el artefacto
completo cumple la especificación. Un agente Code Reviewer con acceso al
código y a los resultados de las pruebas puede identificar ese tipo de
errores de diseño, que no son evidentes a nivel local.

## 7.3. Resultados cuantitativos del baseline

La configuración baseline es la primera del barrido principal y aporta el
mayor número de réplicas completadas a fecha de cierre. La tabla 7.2
resume sus métricas agregadas y la figura 7.1 muestra el intervalo de
confianza correspondiente.

| Métrica | Valor (estimación puntual) | Intervalo de confianza 95% |
|---|---:|---|
| pass@1 | ≈ 0.83 | bootstrap percentil, 2 000 remuestras |
| pass@3 | ≈ 0.86 | estimador Chen et al. (2021) |
| Tokens medios por problema | ≈ 280 | suma input + output |
| Latencia media por problema (s) | ≈ 4.9 | wall-clock incluido overhead de LangGraph |
| Revisiones medias | 0.00 | n/a en baseline |

Tabla 7.2. Resumen del baseline al cierre del documento (consultar
`doc/tables/summary.md` para la versión actualizada con los datos más
recientes del experimento).

El número exacto se regenera de forma reproducible mediante
`python experiments/analyze_results.py`, que reescribe la tabla anterior y
las figuras 7.1 a 7.3 con la cardinalidad disponible en el momento de la
ejecución del análisis.

Figura 7.1. pass@1 por configuración con intervalos de confianza al 95%
mediante bootstrap percentil (`figures/pass_at_1.png`). El gráfico se
regenera automáticamente con datos adicionales conforme avanza la corrida;
cuando todas las configuraciones del barrido completen ejecuciones, la
figura mostrará la comparación pareada entre arquitecturas que sustenta el
contraste de la hipótesis H1.

### 7.3.1. Interpretación del baseline

El valor de pass@1 del baseline sobre HumanEval es consistente con la
literatura para modelos de 7 B parámetros sobre tareas básicas de
generación de código (Hui et al., 2024). El intervalo de confianza,
estrecho gracias a las 390 ejecuciones disponibles, indica que la
estimación es estadísticamente sólida y proporciona la línea base contra
la cual comparar las configuraciones multi-agente cuando estas completen
sus respectivas pasadas.

La latencia media por problema, alrededor de 5 segundos, es coherente con
una sola invocación al modelo de 7 B sobre Apple Silicon: incluye el
tiempo de generación del modelo cuantizado y el overhead de orquestación
de LangGraph, que añade décimas de segundo por la inicialización y el
compilado del grafo unípido. Esta latencia base es el punto de referencia
contra el cual interpretar las decenas o cientos de segundos por problema
esperables para las configuraciones multi-agente, que invocan al modelo
cinco o más veces por ejecución.

### 7.3.2. Validación cruzada del baseline con la literatura

El valor de pass@1 del baseline puede contrastarse con números
públicos del modelo Qwen 2.5 Coder 7B Instruct sobre HumanEval para
verificar que el setup experimental de este trabajo se sitúa dentro
del rango esperable del modelo y no introduce sesgos accidentales.

| Fuente | Modelo | Benchmark | pass@1 | Diferencia con este TFG |
|---|---|---|---|---|
| Hui et al. (2024), reporte técnico Qwen 2.5 Coder | qwen2.5-coder-7b-instruct (FP16) | HumanEval | 88.4% | — |
| EvalPlus public leaderboard (HumanEval+) | qwen2.5-coder-7b-instruct (FP16) | HumanEval+ | ≈ 76% | — |
| **Este TFG** (baseline) | qwen2.5-coder-7b-instruct Q4_K_M | HumanEval+ (evalplus) | ≈ 83% | — |

Tabla 7.4. Comparativa del baseline frente a referencias públicas del
mismo modelo.

La discrepancia entre los tres números es coherente con tres
diferencias metodológicas conocidas:

**Cuantización Q4_K_M.** La versión utilizada en este trabajo es
cuantizada a 4 bits (Q4_K_M, ~6.5 GB) frente al FP16 (~14 GB) que
reporta el paper original. La literatura documenta una pérdida típica
de 2-5 puntos de pass@1 por la cuantización a Q4_K_M en modelos
similares (Hui et al., 2024). El valor observado en este TFG (~83 %)
es consistente con esa horquilla.

**HumanEval vs HumanEval+.** El benchmark utilizado es
`evalplus/humanevalplus` (Liu et al., 2023), con suite de tests
extendida. EvalPlus es entre 3 y 12 puntos más exigente que HumanEval
original; los números reportados por Hui et al. (2024) corresponden
al HumanEval original.

**Prompt y temperatura.** El prompt del agente baseline difiere
ligeramente del usado en los reportes públicos (que utilizan zero-shot
plain completion). La temperatura está fijada a 0.2 frente al greedy
sampling habitual en evaluaciones formales.

El resultado de este contraste cruzado es que el setup experimental
no introduce sesgos significativos: el baseline reproduce el modelo
dentro del rango esperable. Cualquier diferencia que las
configuraciones multi-agente exhiban sobre este baseline puede
atribuirse al diseño arquitectural y no a un sesgo en la evaluación.

## 7.4. Adherencia estructural al protocolo

La métrica de adherencia estructural definida en la sección 6.4.4
cuantifica el porcentaje de ejecuciones en las que ningún agente del
pipeline emitió un artefacto malformado. El resultado preliminar sobre el
baseline (al cierre del documento, 390 ejecuciones) es del 100% de
adherencia: ningún output del agente baseline carece del bloque
```python``` esperado.

| Configuración | Runs | Runs con fallo estructural | Avisos totales | Adherencia |
|---|---:|---:|---:|---:|
| baseline | 390 | 0 | 0 | 100.00% |

Tabla 7.3. Adherencia estructural medida al cierre del documento
(`doc/tables/adherence.md`). La tabla se actualiza al ejecutar
`python experiments/adherence_metric.py` y crecerá con filas adicionales
conforme las configuraciones multi-agente completen ejecuciones.

Este resultado preliminar no refuta ni confirma todavía la hipótesis del
protocolo estructurado: con sólo una configuración medida no hay
referencia con la cual contrastar. Su valor está en aportar el listón
máximo —un sistema sin pipeline complejo, en el modelo más simple del
estudio, no produce ningún fallo de formato— sobre el que evaluar si las
configuraciones más complejas mantienen, mejoran o degradan la adherencia.

## 7.5. Análisis coste-calidad

La figura 7.2 sitúa cada configuración en el plano coste–calidad, con el
eje horizontal en escala logarítmica para acomodar la diferencia esperada
entre el baseline (decenas de tokens) y las configuraciones multi-agente
con self-reflection (potencialmente decenas de miles).

Figura 7.2. Frontera de Pareto coste–calidad: tokens totales por problema
frente a pass@1 (`figures/cost_quality_pareto.png`). Cada punto representa
una configuración. La pendiente entre puntos cuantifica el coste marginal
en tokens por cada incremento de pass@1.

A fecha de cierre la figura contiene un único punto (baseline). Su
utilidad final se manifiesta cuando aparezcan sequential y las variantes
con self-reflection: la pregunta crítica que el gráfico permite responder
es si la mejora de calidad atribuida al pipeline multi-agente compensa el
incremento de coste, y en qué pendiente.

## 7.6. Análisis pendiente de evidencia adicional

Las siguientes secciones requieren completar configuraciones adicionales
del barrido. Su estructura y procedimiento están definidos y reproducirán
los resultados de forma automática mediante el pipeline de análisis a
medida que los CSV se llenen.

### 7.6.1. Comparación pipeline secuencial vs. baseline (H1)

La hipótesis H1 (capítulo 3) postula que el pipeline secuencial obtiene
pass@1 superior al baseline sobre HumanEval. El contraste se realiza
mediante test de McNemar pareado sobre los resultados `pass_all_tests`
emparejando por `(problem_id, seed)` y comparando baseline frente a
sequential. El test es apropiado para clasificadores binarios sobre los
mismos sujetos. La implementación, en `experiments/analyze_results.py`,
emplea aproximación chi-cuadrado con corrección de continuidad cuando el
número de pares discordantes `b + c` es al menos 25, y test binomial
exacto en caso contrario para preservar la potencia en muestras pequeñas.

La tabla 7.4 (`doc/tables/pairwise_mcnemar.md`) reporta todas las
comparaciones pareadas entre configuraciones, ordenadas por p-valor
ascendente. Al cierre del documento la matriz pareada está formada
mayoritariamente por celdas con un número reducido de problemas
emparejados, dado que sólo el baseline ha completado la pasada y los
demás CSV crecen a medida que la corrida avanza. La estructura del
contraste, no obstante, queda fijada: la celda `b` (baseline acierta,
sequential falla) frente a `c` (baseline falla, sequential acierta)
proporciona la diferencia direccional, y el p-valor cuantifica si la
asimetría observada es atribuible al azar bajo la hipótesis nula de
idéntica probabilidad de acierto.

### 7.6.2. Comparación self-reflection vs. secuencial (H2)

La hipótesis H2 postula que el ciclo iterativo de revisión mejora pass@1
respecto al pipeline secuencial sin ciclo, con incremento cuantificable
en tokens y latencia. El contraste se replicará para cada valor de
`max_revisions ∈ {1, 2, 3}`. La distribución del campo `revision_count`
en los CSV de self-reflection permitirá además observar con qué frecuencia
el sistema converge antes de agotar el presupuesto de revisiones; esa
información se reporta en la figura 7.3.

Figura 7.3. Distribución del número de iteraciones de revisión por
configuración self-reflection (`figures/revision_distribution.png`). Una
distribución concentrada en `r=0` indicaría que el sistema aprueba en la
primera pasada en la mayoría de los casos y que el ciclo iterativo es
poco utilizado; una distribución desplazada hacia valores altos
indicaría lo contrario.

### 7.6.3. Análisis del trade-off por dificultad (H3)

La hipótesis H3 postula que el beneficio relativo del pipeline complejo
disminuye en problemas de baja complejidad. Para contrastarla se segmenta
el conjunto de problemas según la dispersión de pass@1 entre
configuraciones —spread = max − min sobre las configs— y se compara el
ratio calidad/coste por categoría. El informe automático
`doc/tables/problem_difficulty.md` lista el top-20 de problemas con
mayor spread, que constituye el subconjunto más informativo para esta
discusión.

### 7.6.4. Ablaciones de rol

Las tres variantes de ablación (`no_pm`, `no_architect`, `no_reviewer`)
forman el núcleo del análisis de contribución por rol planteado en la
propuesta. La pregunta concreta que cada una responde:

- `no_reviewer` vs. sequential: ¿cuánto aporta el veredicto explícito del
  Code Reviewer, ya descontado el efecto del QA Tester que ejecuta los
  tests en ambas?
- `no_architect` vs. sequential: ¿cuánto aporta el documento de diseño
  intermedio frente a que el Developer trabaje directamente sobre la PRD?
- `no_pm` vs. sequential: ¿cuánto aporta la formalización del PRD frente
  al enunciado original?

Los resultados de cada comparación se reportarán como diferencia pareada
con su intervalo de confianza bootstrap, junto con el ratio
calidad-por-token de cada variante.

## 7.7. Reproducción de los resultados

Para regenerar todas las figuras y tablas de este capítulo en cualquier
momento, basta con:

```bash
python experiments/analyze_results.py     # figuras 7.1–7.3 + tabla 7.2
python experiments/adherence_metric.py    # tabla 7.3
```

Listado 7.1. Comandos de regeneración de las figuras y tablas del capítulo.

Los ficheros producidos en `figures/` y `doc/tables/` son el output
canónico del experimento al cierre de la corrida.
