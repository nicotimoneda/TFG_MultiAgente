# Capítulo 7: Resultados

## 7.1. Estudio piloto exploratorio

Con carácter previo a la evaluación cuantitativa formal prevista para la entrega
final, se realizó un estudio piloto exploratorio de naturaleza cualitativa. El
estudio consiste en la ejecución de un LLM de propósito general —con un prompt
monolítico sin estructura de roles— sobre una muestra de cuatro problemas del
benchmark HumanEval (Chen et al., 2021), seguida de un análisis manual de los
outputs generados. El objetivo no es medir pass@k sino caracterizar los patrones
de error recurrentes en un agente único antes de diseñar el sistema multi-agente,
de forma que las decisiones arquitecturales del capítulo 4 tengan una motivación
empírica, aunque preliminar.

Los resultados de este estudio no sustituyen a la evaluación cuantitativa formal,
que se presentará en la entrega final con el sistema completo implementado.

## 7.2. Análisis cualitativo de outputs

**HumanEval/1 — Separación de grupos de paréntesis**

La tarea consiste en separar una cadena con múltiples grupos de paréntesis en
grupos independientes y correctamente balanceados. El LLM genera una
implementación funcional para entradas con poco anidamiento. Cuando la cadena
contiene grupos entrelazados con anidamiento profundo, la lógica de seguimiento
del nivel de profundidad no se mantiene y el resultado incluye grupos mal
delimitados. La solución no incluye ningún caso de prueba ni comprobación de la
propiedad de balance, lo que impide detectar el fallo sin ejecución manual.

**HumanEval/26 — Eliminación de duplicados**

La tarea consiste en devolver los elementos que aparecen exactamente una vez,
eliminando los que se repiten. La implementación es correcta para el caso general
pero no cubre los casos límite: lista vacía, lista donde todos los elementos se
repiten y lista de un único elemento. El modelo no genera pruebas que incluyan
esos casos ni documenta las precondiciones que asume. El error no es de lógica
sino de cobertura, y no resulta visible sin un conjunto de pruebas suficientemente
denso.

**HumanEval/38 — Codificación cíclica**

La tarea requiere implementar una función de codificación cíclica sobre grupos de
tres caracteres y su función inversa de decodificación. El LLM genera la función
encode de forma correcta pero produce una función decode que aplica la misma
transformación en lugar de la transformación inversa: la composición
decode(encode(s)) no devuelve la cadena original. El modelo no verifica la
propiedad de inversión en ningún momento del proceso de generación.

**HumanEval/119 — Concatenación de cadenas de paréntesis**

La tarea consiste en determinar si dos cadenas de paréntesis pueden concatenarse
en algún orden para formar una cadena balanceada. La implementación comprueba
correctamente el caso en que la concatenación directa produce balance, pero no
considera de forma sistemática el orden inverso. La lógica del segundo caso
aparece en algunos outputs pero de forma incompleta, produciendo falsos negativos
en entradas donde solo el orden invertido es válido.

**Tabla resumen**

| Problema | Comportamiento observado | Tipo de error | ¿Detectable por otro agente? |
|---|---|---|---|
| HumanEval/1 | Correcto en casos simples, falla con anidamiento profundo | Error lógico en casos complejos | Sí, con tests de regresión |
| HumanEval/26 | Correcto en caso general, sin cobertura de casos límite | Error de cobertura | Sí, con casos de prueba específicos |
| HumanEval/38 | encode correcto, decode no es la inversa | Error semántico en función complementaria | Sí, verificando la composición |
| HumanEval/119 | Comprueba un solo orden de concatenación | Error lógico por análisis incompleto | Sí, con revisión de la especificación |

## 7.3. Implicaciones para el diseño del sistema

Los cuatro problemas analizados comparten una característica: sus errores no son
visibles para el agente que generó el código, pero son detectables por un agente
externo con acceso al mismo estado del problema. Esa asimetría es el punto de
partida para justificar el diseño multi-agente.

El patrón más frecuente es la ausencia de verificación. En HumanEval/1 y
HumanEval/26, el modelo no genera pruebas que cubran los casos que falla. Un
agente dedicado exclusivamente a generar y ejecutar casos de prueba —incluyendo
casos límite que el desarrollador no contempla— puede detectar esos fallos antes
de que lleguen a revisión. Esa función corresponde al agente QA Tester del sistema
propuesto.

La incapacidad de verificar propiedades semánticas globales, como la propiedad de
inversión en HumanEval/38, señala un problema distinto: el modelo evalúa el código
línea a línea pero no comprueba si el artefacto completo cumple la especificación.
Un agente Code Reviewer con acceso al código y a los resultados de las pruebas
puede identificar ese tipo de errores de diseño, que no son evidentes a nivel
local.

El hecho de que estos errores sean corregibles cuando se señalan de forma explícita
apunta a que un único pase de generación no es suficiente. El ciclo iterativo entre
Code Reviewer y Developer existe precisamente para dar al sistema la oportunidad de
corregir fallos que no se resuelven en la primera iteración.

Por último, los fallos en HumanEval/119 —donde la interpretación incompleta del
enunciado lleva a una implementación parcialmente incorrecta— indican que el
problema no siempre está bien especificado para el agente que lo implementa. Contar
con un agente Product Manager que formalice los requisitos antes de la
implementación reduce la probabilidad de que el Developer trabaje sobre una lectura
incompleta del problema.

La evaluación cuantitativa formal, con el sistema completo implementado sobre
HumanEval y MBPP y con análisis estadístico de pass@1, pass@k y coste en tokens,
se presentará en la entrega final.
