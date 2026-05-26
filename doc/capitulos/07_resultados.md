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

La política es reportar lo que ya está medido, con sus intervalos de
confianza, en vez de esperar a que la matriz se cierre por completo. Es
la única forma de que el capítulo contenga evidencia cuantitativa real;
la alternativa sería rellenar las tablas con números ilustrativos, y eso
choca de frente con el principio metodológico del capítulo 4.

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

## 7.3. Resultados cuantitativos por configuración

La tabla 7.2 reúne las métricas agregadas de las tres configuraciones de
la corrida principal con los datos disponibles al cierre del documento
(1 195 ejecuciones, 80,9 % de la matriz). El baseline y el pipeline
secuencial tienen la matriz completa de 492 ejecuciones (164 problemas ×
3 semillas); self_reflection_r1 está en curso y aporta 211 ejecuciones.
La figura 7.1 muestra los intervalos de confianza correspondientes.

| Configuración | n | pass@1 | IC 95 % | pass@3 | Tokens medios | Latencia media (s) | Revisiones medias |
|---|---:|---:|---|---:|---:|---:|---:|
| Baseline | 492 | 80,08 % | [76,4 ; 83,5] | 83,54 % | 283 | 5,13 | 0,00 |
| Sequential | 492 | 58,33 % | [53,9 ; 62,6] | 77,44 % | 11 614 | 396,35 | 0,00 |
| SR (r=1) | 211 | 67,30 % | [60,7 ; 73,5] | 84,51 % | 13 719 | 386,79 | 0,39 |

Tabla 7.2. Resumen comparativo de las tres configuraciones al cierre del
documento. La versión vigente, regenerada cada vez que el pipeline de
análisis se ejecuta sobre los CSV actualizados, está en
`doc/tables/summary.md`.

Los números de la tabla se regeneran de forma reproducible mediante
`python experiments/analyze_results.py`, que reescribe la tabla y las
figuras 7.1 a 7.3 con la cardinalidad disponible en el momento de la
ejecución del análisis.

Figura 7.1. pass@1 por configuración con intervalos de confianza al 95%
mediante bootstrap percentil (`figures/pass_at_1.png`). El gráfico se
regenera automáticamente con datos adicionales conforme avanza la corrida;
cuando todas las configuraciones del barrido completen ejecuciones, la
figura mostrará la comparación pareada entre arquitecturas que sustenta el
contraste de la hipótesis H1.

### 7.3.1. Interpretación

El primer hallazgo, y el más relevante, es que el baseline **supera a
las dos configuraciones multi-agente** sobre HumanEval con este modelo.
El pipeline secuencial cae 21,8 puntos respecto al baseline (de 80,08 %
a 58,33 %) y la configuración con self-reflection cae 12,8 puntos
(67,30 %), con datos parciales pero ya suficientes para que la diferencia
sea estadísticamente significativa (sección 7.6.1). La hipótesis intuitiva
de que distribuir el trabajo entre agentes especializados mejoraría la
corrección no se sostiene en este experimento.

El segundo hallazgo es el coste. Las configuraciones multi-agente
consumen alrededor de 40 veces más tokens y 77 veces más latencia por
problema que el baseline, **para producir peores resultados**. La
frontera de Pareto coste-calidad (sección 7.5) la ocupa por completo el
baseline: ninguna configuración multi-agente le es competitiva en
ninguna de las dos dimensiones.

El valor del baseline (80,08 % pass@1) es coherente con la literatura
para Qwen 2.5 Coder 7B Instruct con cuantización Q4_K_M sobre
HumanEval+. La validación cruzada de la sección 7.3.2 sitúa el número
dentro del rango esperable y descarta sesgos en el setup experimental.

La latencia del baseline, 5,13 segundos por problema, es coherente con
una sola invocación al modelo de 7 B sobre Apple Silicon. La latencia
de sequential y SR_r1 —del orden de 400 segundos por problema— refleja
las cinco o seis invocaciones encadenadas del pipeline, con
acumulación de contexto creciente en cada llamada.

### 7.3.2. Validación cruzada del baseline con la literatura

El valor de pass@1 del baseline puede contrastarse con números
públicos del modelo Qwen 2.5 Coder 7B Instruct sobre HumanEval para
verificar que el setup experimental de este trabajo se sitúa dentro
del rango esperable del modelo y no introduce sesgos accidentales.

| Fuente | Modelo | Benchmark | pass@1 | Diferencia con este TFG |
|---|---|---|---|---|
| Hui et al. (2024), reporte técnico Qwen 2.5 Coder | qwen2.5-coder-7b-instruct (FP16) | HumanEval | 88.4% | — |
| EvalPlus public leaderboard (HumanEval+) | qwen2.5-coder-7b-instruct (FP16) | HumanEval+ | ≈ 76% | — |
| **Este TFG** (baseline) | qwen2.5-coder-7b-instruct Q4_K_M | HumanEval+ (evalplus) | 80,08 % | — |

Tabla 7.3. Comparativa del baseline frente a referencias públicas del
mismo modelo.

La discrepancia entre los tres números es coherente con tres
diferencias metodológicas conocidas:

**Cuantización Q4_K_M.** La versión utilizada en este trabajo es
cuantizada a 4 bits (Q4_K_M, ~6.5 GB) frente al FP16 (~14 GB) que
reporta el paper original. La literatura documenta una pérdida típica
de 2-5 puntos de pass@1 por la cuantización a Q4_K_M en modelos
similares (Hui et al., 2024). El valor observado en este TFG (80,08 %)
es consistente con esa horquilla: cabe dentro del descenso esperado por
cuantización respecto al 88,4 % en FP16.

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
pipeline emitió un artefacto malformado. La tabla 7.4 reúne los
valores observados en las tres configuraciones de la corrida principal.

| Configuración | Runs | Runs con fallo estructural | Avisos totales | Adherencia |
|---|---:|---:|---:|---:|
| baseline | 492 | 0 | 0 | 100,00 % |
| sequential | 492 | 0 | 0 | 100,00 % |
| self_reflection_r1 | 211 | 1 | 1 | 99,53 % |

Tabla 7.4. Adherencia estructural medida al cierre del documento
(`doc/tables/adherence.md`). La tabla se actualiza al ejecutar
`python experiments/adherence_metric.py` y crecerá conforme SR_r1
complete la pasada.

El resultado es claro: el protocolo de comunicación estructurada por
artefactos se cumple casi sin excepciones, también en las configuraciones
más complejas. La caída de medio punto en SR_r1 (un fallo de formato
sobre 211 ejecuciones) entra dentro del ruido y no establece un patrón.
La afirmación de la literatura de que el protocolo estructurado reduce
alucinaciones de formato queda, en este experimento, **confirmada**: el
modelo de 7 B respeta el contrato de salida con consistencia en
las tres configuraciones. Es, sin embargo, una victoria parcial: la
adherencia estructural mide el *formato*, no la *corrección* del
contenido. Como muestran las cifras de pass@1 de la sección 7.3, el
mismo pipeline que respeta el formato a rajatabla genera código menos
correcto que el baseline.

## 7.5. Análisis coste-calidad

La figura 7.2 sitúa las tres configuraciones en el plano coste–calidad,
con el eje horizontal en escala logarítmica para acomodar la diferencia
de dos órdenes de magnitud entre el baseline (283 tokens por problema)
y las configuraciones multi-agente (más de 11 000 tokens por problema).

Figura 7.2. Frontera de Pareto coste–calidad: tokens totales por problema
frente a pass@1 (`figures/cost_quality_pareto.png`). Cada punto representa
una configuración.

El resultado es contundente: **la frontera de Pareto está formada por un
único punto, el baseline**. Tanto el pipeline secuencial como
self_reflection_r1 están dominados —son simultáneamente más caros y
menos correctos—. En términos de coste marginal por punto de pass@1
ganado, las configuraciones multi-agente no solo no ganan: pierden. El
pipeline secuencial paga 11 331 tokens adicionales por problema para
caer 21,8 puntos de pass@1; self_reflection_r1 paga 13 436 tokens
adicionales para caer 12,8 puntos.

Este patrón —pipeline más largo, peor resultado, mucho más coste— es el
hallazgo más importante del experimento. La sección 8.3 discute las
causas plausibles. En términos de decisión de diseño, el dato es claro:
para tareas de HumanEval y un modelo de 7 B parámetros, **un único
agente bien promptado domina al pipeline multi-agente en las dos
dimensiones**.

## 7.6. Contraste de hipótesis

### 7.6.1. Comparación pipeline secuencial vs. baseline (H1)

La hipótesis H1 (capítulo 3) postula que el pipeline secuencial obtiene
pass@1 superior al baseline sobre HumanEval. El contraste se realiza
mediante test de McNemar pareado sobre los resultados `pass_all_tests`
emparejando por `(problem_id, seed)` y comparando baseline frente a
sequential. La implementación, en `experiments/analyze_results.py`,
emplea aproximación chi-cuadrado con corrección de continuidad cuando
el número de pares discordantes `b + c` es al menos 25, y test binomial
exacto en caso contrario.

Con la matriz baseline × sequential completa (492 pares) los datos son:

| Comparación | n_pares | b (A acierta, B falla) | c (B acierta, A falla) | p-valor | Método |
|---|---:|---:|---:|---:|---|
| Baseline vs Sequential | 492 | 128 | 21 | < 0,0001 | chi² |
| Baseline vs SR (r=1) | 211 | 51 | 4 | < 0,0001 | chi² |
| Sequential vs SR (r=1) | 211 | 19 | 29 | 0,194 | chi² |

Tabla 7.5. Comparaciones pareadas de McNemar entre configuraciones.
La columna `b` recoge problemas donde la primera configuración acierta
y la segunda falla; `c` el caso simétrico. La tabla viva está en
`doc/tables/pairwise_mcnemar.md`.

**Resultado para H1.** El baseline supera al pipeline secuencial en 128
problemas donde sequential falla; sequential solo gana en 21 problemas
donde el baseline falla. La asimetría es enorme y la diferencia es
estadísticamente significativa al nivel del 0,01 %. **H1 se rechaza**
con la dirección opuesta a la hipotetizada: el pipeline secuencial no
solo no mejora al baseline, sino que lo empeora de forma consistente.

### 7.6.2. Comparación self-reflection vs. secuencial (H2)

La hipótesis H2 postula que el ciclo iterativo de revisión mejora
pass@1 respecto al pipeline secuencial sin ciclo. Con los 211 pares
de SR_r1 disponibles al cierre, la diferencia direccional va en el
sentido esperado por H2 (SR_r1 acierta 29 problemas donde sequential
falla; sequential acierta 19 donde SR_r1 falla), pero el p-valor de
0,194 indica que la diferencia no es significativa con los datos
actuales. **H2 queda en estado no concluyente** a la espera del
cierre de SR_r1.

La distribución del campo `revision_count` aporta una observación
relacionada: el 61,14 % de las ejecuciones de SR_r1 aprueba sin
ninguna revisión (`r=0`); en el resto el revisor solo dispara una
iteración antes del veredicto final. La media de revisiones por
ejecución es 0,39. Con `max_revisions = 1`, el grafo o bien acepta
en la primera pasada o bien hace un único ciclo Reviewer →
Developer. Las configuraciones r2 y r3, que permitirían más
iteraciones, no se ejecutan en esta corrida (S8 del anexo de
decisiones); su contraste queda como línea futura.

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
python experiments/analyze_results.py     # figuras 7.1–7.3 + tablas 7.2, 7.3 y 7.5
python experiments/adherence_metric.py    # tabla 7.4
```

Listado 7.1. Comandos de regeneración de las figuras y tablas del capítulo.

Los ficheros producidos en `figures/` y `doc/tables/` son el output
canónico del experimento al cierre de la corrida.
