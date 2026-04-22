# Capítulo 4: Metodología

## 4.1. Enfoque metodológico

Este trabajo adopta un enfoque empírico-comparativo controlado. El objetivo es
medir con precisión qué diferencia produce cada componente arquitectural añadido,
bajo qué condiciones y a qué coste computacional, no demostrar en abstracto que
los sistemas multi-agente superan a los LLMs monolíticos.

Para eso se diseñan tres configuraciones incrementales: un baseline monolítico, un
sistema multi-agente secuencial y un sistema multi-agente con self-reflection. La
comparación entre el baseline y la configuración secuencial mide el efecto de la
especialización por roles y el estado compartido; la comparación entre la
configuración secuencial y la de self-reflection mide el efecto del ciclo de
revisión iterativo. Sin esa estructura incremental, una diferencia de rendimiento
entre el baseline y el sistema completo no podría atribuirse a ningún componente
específico.

Las tres configuraciones se evalúan sobre los mismos benchmarks y con las mismas
métricas, lo que garantiza que las diferencias observadas son atribuibles al diseño
del sistema y no a variaciones en el entorno de evaluación.

## 4.2. Descripción de los modelos implementados

**Configuración 1 — Baseline monolítico**

El baseline consiste en un único nodo LangGraph que recibe el enunciado del
problema y genera directamente la solución en código. No hay coordinación entre
agentes, no hay estado compartido estructurado y el prompt incluye en un único
mensaje la especificación completa de la tarea. Esta configuración es
funcionalmente equivalente a usar el LLM directamente a través de su API con un
prompt diseñado para maximizar la calidad de la respuesta en una sola llamada.
Sirve como punto de referencia para cuantificar la diferencia que introducen las
configuraciones más complejas.

**Configuración 2 — Multi-agente secuencial**

Esta configuración orquesta cinco agentes especializados en un pipeline sin
ciclos: Product Manager, Arquitecto, Developer, QA Tester y Code Reviewer. El
flujo es estrictamente secuencial: cada agente recibe el estado acumulado hasta
ese punto, añade su artefacto al estado compartido y pasa el control al siguiente
nodo.

El estado compartido es un TypedDict con campos tipados que corresponden a los
artefactos de cada agente: PRD (documento de requisitos), DesignDoc (documento de
diseño técnico), CodeArtifact (código fuente generado), TestResults (resultados
de los casos de prueba) y ReviewComments (comentarios del revisor). Este diseño
garantiza que cada agente trabaja sobre información estructurada y verificable, y
no sobre texto libre sin tipo.

Los prompts de sistema de cada agente definen su rol, las entradas que debe
consumir del estado y el formato exacto del artefacto que debe producir. La
ejecución de los casos de prueba generados por QA Tester se realiza en un sandbox
aislado con timeout para prevenir ejecuciones indefinidas.

**Configuración 3 — Multi-agente con self-reflection**

Esta configuración extiende la anterior con un ciclo iterativo de revisión. Tras
la ejecución del Code Reviewer, una arista condicional evalúa el campo
ReviewComments del estado: si el revisor aprueba el código, el flujo termina; si
solicita cambios, el control vuelve al Developer para una nueva iteración.

El número de iteraciones realizadas se registra en el campo revision_count del
estado compartido. El parámetro max_revisions actúa como hiperparámetro del
sistema: fija el número máximo de ciclos Reviewer→Developer antes de forzar la
terminación, independientemente del resultado de la revisión. En los experimentos
se evalúan valores de max_revisions ∈ {1, 2, 3}.

## 4.3. Stack tecnológico y justificación

**LangGraph** es el framework de orquestación central del sistema. Modela el
flujo de trabajo como un grafo dirigido de estados finitos: los nodos son los
agentes especializados y las aristas definen las transiciones entre ellos, que
pueden ser incondicionales o condicionales sobre el estado compartido (LangChain
Inc., 2024). El flujo de control es explícito e inspeccionable, y el estado
persiste entre nodos sin depender de la longitud de la ventana de contexto. Ambas
propiedades son necesarias para implementar el ciclo de revisión de la
configuración 3 de forma reproducible y auditable.

**LangChain** proporciona las abstracciones de integración con el backend LLM:
gestión de prompts, invocación de herramientas y manejo de salidas estructuradas.
Su uso desacopla la lógica de los agentes del proveedor de modelo subyacente, lo
que permite cambiar el backend sin modificar el código de los agentes.

**Python** es el lenguaje del ecosistema de investigación en IA y dispone del
tooling más completo para la evaluación de benchmarks de código: ejecutores de
pruebas, entornos sandbox y librerías de análisis estadístico.

**Backend LLM** — Los experimentos utilizan modelos de código abierto servidos a
través de la API de Groq, concretamente Llama 3.3 70B y Qwen 2.5 Coder. El uso
de modelos abiertos cumple dos funciones: la reproducibilidad —cualquier
investigador puede ejecutar los mismos experimentos accediendo a los mismos pesos
de modelo— y el control de costes, que permite ejecutar múltiples seeds y
configuraciones sin restricciones económicas que distorsionen el diseño
experimental.

## 4.4. Diseño experimental

**Benchmarks seleccionados**

Los experimentos se realizan sobre HumanEval (164 problemas) y un subconjunto de
200 problemas de MBPP. Ambos benchmarks tienen especificación en lenguaje natural,
solución verificable mediante pruebas unitarias y uso extendido en la literatura,
lo que permite comparar los resultados con los de trabajos previos (Chen et al.,
2021).

SWE-bench se descartó para esta evaluación. Sus problemas requieren reproducir el
entorno de ejecución de repositorios externos con dependencias variables, lo que
introduce una complejidad de infraestructura fuera del alcance de este TFG.
HumanEval y MBPP ofrecen un entorno suficientemente controlado para los objetivos
del trabajo.

**Métricas**

pass@1 mide la probabilidad de que una única solución generada supere todas las
pruebas del problema. Para estimarlo de forma insesgada se utiliza el estimador
de Chen et al. (2021):

$$\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

donde n es el número de muestras generadas por problema, c el número de soluciones
correctas y k el número de intentos considerados. En este trabajo se usa n = 10
y k ∈ {1, 5, 10}.

La tasa media de superación de pruebas (average test pass rate) mide, para las
soluciones que no superan todas las pruebas, qué fracción de ellas sí superan.
Esta métrica complementa pass@k al capturar soluciones parcialmente correctas.

El coste en tokens se registra como número total de tokens consumidos por problema
(entrada + salida). La latencia se mide como tiempo de wall-clock desde la
recepción del enunciado hasta la entrega de la solución final.

**Protocolo experimental**

Cada problema se ejecuta con cinco seeds distintas para estimar la varianza de
los resultados. Las llamadas al modelo se realizan con temperatura 0.2 para las
configuraciones 1 y 2, y con temperatura 0.4 para la configuración 3, donde una
mayor diversidad en la generación del Developer puede facilitar la convergencia
del ciclo de revisión. La ejecución de código se realiza en un sandbox Python con
timeout de 5 segundos por caso de prueba.

Los resultados de cada ejecución se guardan en un fichero CSV estructurado con los
campos benchmark, problem_id, configuration, seed, pass_all_tests,
test_pass_rate, tokens_input, tokens_output y latency_seconds.

**Análisis estadístico**

Para comparar las tasas de pass@1 entre configuraciones se aplica el test de
McNemar pareado, apropiado para comparar dos clasificadores sobre los mismos
problemas. Para las métricas continuas —tasa de superación de pruebas, coste en
tokens, latencia— se calculan intervalos de confianza al 95% mediante bootstrap
con 10.000 remuestras.

## 4.5. Consideraciones de reproducibilidad

El código fuente del sistema se publica en un repositorio público de GitHub,
incluyendo la definición del grafo LangGraph, los prompts completos de cada
agente, los scripts de evaluación y los resultados en bruto. Los prompts de
sistema de cada agente se recogen también en el Anexo A del documento.

Las dependencias del proyecto se declaran en un fichero pyproject.toml con
versiones fijadas para todos los paquetes. Las seeds utilizadas en los
experimentos se documentan en el fichero de configuración del experimento. El
entorno de ejecución de pruebas es un proceso Python independiente sin acceso
a red, con límites de memoria y tiempo configurables vía parámetros.
