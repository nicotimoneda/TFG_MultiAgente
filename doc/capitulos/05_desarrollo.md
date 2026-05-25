# Capítulo 5: Desarrollo

## 5.1. Estructura del repositorio

El repositorio del proyecto se organiza en cuatro paquetes principales bajo el
directorio `src/` y un directorio paralelo `experiments/` que aloja el banco de
pruebas. Esta separación responde al criterio de Pereira-Vale et al. (2024)
sobre arquitecturas multi-agente: el código del sistema —agentes, grafo, estado
compartido— debe ser independiente del arnés de evaluación, de forma que los
componentes del sistema se puedan invocar tanto desde el experimento como desde
cualquier integración futura sin reorganización.

| Paquete | Responsabilidad |
|---|---|
| `src/state/` | Definición del `AgentState` (TypedDict) compartido por todos los grafos |
| `src/agents/` | Agente base, agente monolítico y los cinco agentes de rol |
| `src/graph/` | Constructores de los tres grafos LangGraph y las variantes de ablación |
| `src/llm/` | Factory de clientes LLM con selección de backend por variable de entorno |
| `src/evaluation/` | Sandbox de ejecución de código, cargador de HumanEval y métricas |
| `experiments/` | Runner principal, pipeline de análisis, métrica de adherencia y dashboard |

Tabla 5.1. Mapa de paquetes del proyecto.

La separación entre `src/agents/` y `src/graph/` corresponde a la distinción
entre lógica del rol (qué prompt usa cada agente, qué artefacto produce) y
lógica de composición (en qué orden se ejecutan los agentes y bajo qué
condiciones se ciclan). Esta separación es deliberada: el mismo agente
Developer aparece en la configuración secuencial, en la configuración con
self-reflection y en dos de las tres variantes de ablación. Mantener el agente
agnóstico respecto al grafo permite reutilizarlo sin duplicación.

## 5.2. Estado compartido `AgentState`

El estado compartido es el contrato entre todos los nodos del grafo. Se
implementa como un `TypedDict` de Python con campos tipados que cubren tres
categorías: la identidad del problema, los artefactos producidos por los
agentes y la telemetría acumulada durante la ejecución.

```python
class AgentState(TypedDict):
    # Identidad del problema
    problem_id: str
    problem_statement: str
    function_signature: str
    test_cases: list[str]

    # Artefactos de los agentes
    prd: str
    design_doc: str
    code_artifact: str
    test_results: dict[str, bool]
    review_comments: str

    # Control de iteración
    revision_count: int

    # Telemetría
    tokens_input: int
    tokens_output: int
    latency_seconds: float

    # Metadatos de la ejecución
    config_name: str
```

Listado 5.1. Definición íntegra del estado compartido `AgentState`.

Cada campo del estado se asocia a un agente productor único, lo que evita
condiciones de carrera lógicas en las que dos agentes intenten poblar el mismo
artefacto. La separación es estricta: `prd` lo escribe únicamente el Product
Manager, `design_doc` el Arquitecto, `code_artifact` el Developer (y, en la
configuración con self-reflection, también su subclase reflexiva),
`test_results` el QA Tester y `review_comments` el Code Reviewer. Los campos
de telemetría acumulan en sitio: cada agente añade sus tokens consumidos a
los contadores existentes.

El campo `config_name` registra qué configuración generó la traza, de forma
que un mismo CSV puede unirse a las trazas de cualquier otra configuración
sin ambigüedad. El campo `revision_count` permanece en cero para las
configuraciones que no implementan el ciclo de auto-revisión.

La elección de `TypedDict` frente a `Pydantic.BaseModel` —que ofrecería
validación en tiempo de ejecución— responde a una restricción concreta del
framework: LangGraph requiere que el estado sea serializable por sus
mecanismos internos de checkpointing, y el coste de validación en cada
transición añadiría latencia sin aportar garantías relevantes en un grafo
con escritores únicos por campo.

## 5.3. Agentes implementados

### 5.3.1. Agente base y `_call_llm`

Todos los agentes heredan de la clase abstracta `BaseAgent`, que centraliza
la lógica de invocación al LLM. El método `_call_llm` acepta un prompt de
sistema y un prompt de usuario, construye los mensajes en formato LangChain y
los envía al cliente configurado. Incorpora retry exponencial con tres
intentos y backoff geométrico (base 2 segundos) para tolerar fallos
transitorios del backend, especialmente en escenarios de rate-limit en el
modo Cerebras.

El método devuelve una tripleta `(response_text, input_tokens, output_tokens)`
que cada agente concreto usa para escribir su artefacto y actualizar la
telemetría del estado. La centralización del retry y de la lectura de
`usage_metadata` en la clase base garantiza que ningún agente individual
pueda silenciar errores ni desincronizar los contadores de tokens.

### 5.3.2. Agente monolítico (Baseline)

El `BaselineAgent` recibe el enunciado completo del problema y devuelve
directamente el artefacto de código. Su prompt instruye al modelo a emitir
la implementación dentro de un bloque ```python```. El agente extrae el
contenido del fence mediante una expresión regular; si el fence no aparece,
registra una advertencia y usa el texto íntegro como código, lo que sirve
después como señal para la métrica de adherencia (sección 5.6).

### 5.3.3. Agentes de rol

Cada uno de los cinco agentes de rol implementa el método abstracto `run`
sobre el `AgentState`. La tabla 5.2 resume sus entradas, salidas y prompts
de sistema. Los prompts completos figuran en el Anexo A.

| Rol | Entrada del estado | Salida | Resumen del prompt de sistema |
|---|---|---|---|
| Product Manager | `problem_statement` | `prd` | Generar PRD estructurado a partir del enunciado |
| Arquitecto | `prd`, `function_signature` | `design_doc` | Producir documento de diseño con algoritmo, estructuras y plan paso a paso |
| Developer | `design_doc`, `function_signature`, `problem_statement` | `code_artifact` | Implementar la función dentro de un bloque ```python``` |
| QA Tester | `code_artifact`, `test_cases` | `test_results` | (Sin LLM) Ejecutar tests en el sandbox |
| Code Reviewer | `code_artifact`, `test_results` | `review_comments` | Emitir veredicto APPROVE/REQUEST_CHANGES + comentarios estructurados |

Tabla 5.2. Especificación de los cinco agentes de rol.

Dos decisiones de diseño merecen detalle. La primera: el agente QA Tester no
invoca al LLM. Su trabajo es puramente determinista —tomar el código del
estado, ejecutarlo contra los `test_cases` en el sandbox y almacenar el mapa
de resultados—. Mantenerlo como nodo del grafo aporta dos cosas: hace
explícita la etapa de verificación en el flujo de control, lo que el lector
puede inspeccionar al imprimir el grafo, y mantiene el contrato del
`AgentState` consistente (todos los campos se poblan a lo largo del
pipeline, no quedan inicializados a vacío). La segunda: el veredicto del
Reviewer se deriva de los `test_results` antes de invocar al LLM. Si todos
los tests pasan, el veredicto es `APPROVE` y el LLM sólo produce comentarios
cualitativos; en caso contrario es `REQUEST_CHANGES`. Esta derivación
determinista evita que el ciclo de self-reflection dependa del parsing
exitoso de una etiqueta dentro de texto libre del modelo, fuente conocida
de fragilidad en sistemas multi-agente (Hong et al., 2024).

## 5.4. Construcción de los grafos LangGraph

### 5.4.1. Grafo baseline

El grafo baseline contiene un único nodo, `solver`, conectado entre `START`
y `END`. Sirve dos propósitos: aislar la diferencia entre invocación LLM
directa y orquestación multi-agente, y proporcionar una referencia
comparativa que sufre exactamente el mismo overhead de LangGraph que las
configuraciones más complejas.

```
START → solver → END
```

Figura 5.1. Grafo de la configuración baseline (esquema generado con
`langgraph.draw_mermaid_png()`, fichero `figures/graph_baseline.png`).

### 5.4.2. Grafo secuencial

El grafo secuencial encadena los cinco agentes sin ciclos. La compilación
produce el siguiente flujo:

```
START → pm → architect → developer → qa → reviewer → END
```

Figura 5.2. Grafo de la configuración secuencial
(`figures/graph_sequential.png`).

Todos los agentes comparten el mismo cliente LLM, configurado con
temperatura 0.2 para favorecer respuestas estables entre réplicas. Esta
elección homogénea facilita la atribución causal: las diferencias entre
configuraciones no se explican por diferencias de temperatura entre roles.

### 5.4.3. Grafo con self-reflection

La configuración con self-reflection reutiliza la topología secuencial pero
añade una arista condicional desde el Reviewer. La función de enrutamiento
inspecciona la primera línea de `review_comments` —que el Reviewer escribe
en el formato `VERDICT: APPROVE` o `VERDICT: REQUEST_CHANGES`— y devuelve
el siguiente nodo en consecuencia. Si el veredicto es APPROVE, el flujo
termina. Si es REQUEST_CHANGES y `revision_count < max_revisions`, el flujo
vuelve al Developer; si se ha alcanzado el límite de revisiones, termina
forzosamente.

```
START → pm → architect → developer → qa → reviewer
                            ↑                  │
                            │ revision         │ APPROVE
                            └──────────────────┤
                                               ↓
                                              END
```

Figura 5.3. Grafo de la configuración con self-reflection. La arista
condicional reviewer → developer está controlada por el campo
`revision_count` del estado y el hiperparámetro `max_revisions`
(`figures/graph_self_reflection_r1.png`, `r2.png`, `r3.png`).

El Developer en esta configuración es una subclase `ReflectiveDeveloperAgent`
que sobrecarga `run()` para detectar la presencia de `review_comments` en
el estado: cuando hay feedback, lo antepone al prompt del usuario antes de
invocar al LLM. El cliente del Developer reflexivo se instancia con
temperatura 0.4, una diversidad superior a la del resto de agentes para
favorecer la exploración de soluciones alternativas en la iteración. Esta
asimetría de temperatura es la única diferencia funcional entre el Developer
inicial y el Developer reflexivo; el resto del prompt es idéntico.

### 5.4.4. Vista de secuencia del pipeline

La figura 5.4 representa el intercambio de mensajes entre los cinco
agentes y el estado compartido, en forma de diagrama de secuencia UML.
Sirve como complemento a las vistas topológicas de las figuras 5.1, 5.2
y 5.3: muestra el orden temporal de las escrituras al `AgentState` y la
asimetría del Code Reviewer, único nodo que puede emitir una arista
inversa en la configuración con self-reflection.

![Diagrama de secuencia del pipeline](figures/secuencia_agentes.png)

Figura 5.4. Diagrama de secuencia del pipeline secuencial y de la
configuración con self-reflection. Las flechas sólidas representan
lectura/escritura sobre el estado compartido; la flecha del Reviewer al
Developer sólo se activa cuando el veredicto es `REQUEST_CHANGES` y
`revision_count < max_revisions`
(`figures/secuencia_agentes.png`).

## 5.5. Sandbox de ejecución

El sandbox de ejecución es el componente que cierra el ciclo de evaluación
funcional: recibe una cadena con el código generado y la lista de aserciones
del benchmark, ejecuta cada test en un proceso aislado y devuelve un mapa
booleano test → resultado. Su implementación es deliberadamente
conservadora.

La ejecución se realiza en un subproceso Python independiente, no mediante
`exec` en el proceso principal. Esta separación evita que el código generado
contamine el espacio de nombres del runner, monopolice el GIL o invoque
funciones del sistema fuera del entorno controlado. El subproceso se lanza
con timeout de cinco segundos por test —tiempo suficiente para los casos
estándar de HumanEval, ajeno a las soluciones que entran en bucles
infinitos—.

Los builtins disponibles dentro del subproceso se restringen a un conjunto
mínimo: tipos primitivos, funciones de iteración, conversiones de tipo y
las construcciones necesarias para que `assert` funcione. Se omiten en
particular `open`, `__import__` con efectos sobre el sistema de ficheros,
acceso a `os.environ` y similares. Cualquier intento de operación
restringida termina con `NameError` y el test se contabiliza como fallo.

La métrica de evaluación se construye sobre el patrón canónico `prompt +
completion` de Chen et al. (2021): antes de ejecutar el sandbox, el prompt
original del problema —que en HumanEval contiene las importaciones y la
firma de la función— se concatena al artefacto generado. Esto garantiza
que la evaluación se realiza sobre el mismo string que se evalúa en la
literatura, con independencia de las decisiones del Developer sobre qué
incluir en su salida.

## 5.6. Variantes de ablación

La propuesta del trabajo identifica como contribución innovadora el análisis
de qué combinaciones de roles aportan más valor. Para cuantificarlo se
implementan tres variantes que eliminan exactamente un rol del pipeline
secuencial, manteniendo el resto idéntico:

| Variante | Topología | Hipótesis que aísla |
|---|---|---|
| `ablation_no_pm` | architect → developer → qa → reviewer | Contribución de la fase de definición de requisitos |
| `ablation_no_architect` | pm → developer → qa → reviewer | Contribución del documento de diseño técnico |
| `ablation_no_reviewer` | pm → architect → developer → qa | Contribución del veredicto explícito de revisión |

Tabla 5.3. Variantes de ablación implementadas.

Cada variante introduce un nodo intermedio sin coste de LLM —`seed_prd` o
`seed_design`— que copia el artefacto inmediatamente disponible al campo
que el siguiente agente espera consumir. Por ejemplo, en `ablation_no_pm`
el nodo `seed_prd` copia `problem_statement` a `prd` antes de invocar al
Arquitecto, manteniendo válido el formato del prompt del Arquitecto sin
gastar tokens en una etapa de Product Manager.

La variante `ablation_no_reviewer` termina tras el QA Tester. Esto preserva
la evaluación funcional —el sandbox sigue ejecutándose— pero elimina toda
señal cualitativa del Code Reviewer. Su comparación contra el pipeline
secuencial completo cuantifica el valor del veredicto explícito de revisión
en términos de pass@1, independientemente del ciclo de iteración que sólo
existe en la configuración con self-reflection.

Las ablaciones comparten `AgentState`, agentes y sandbox con el pipeline
principal: son únicamente reconfiguraciones del grafo. Esto garantiza que
las diferencias observadas se atribuyen a la ausencia del rol y no a
cambios accidentales en otros componentes.

## 5.7. Factory de clientes LLM

El proyecto soporta dos backends de inferencia, seleccionables mediante la
variable de entorno `LLM_BACKEND`:

- `ollama` (por defecto en la corrida final): servidor local escuchando en
  `http://localhost:11434/v1` con el modelo `qwen2.5-coder:7b-instruct-q4_K_M`.
- `cerebras`: Cerebras Inference API con el modelo
  `qwen-3-235b-a22b-instruct-2507`, autenticado mediante `CEREBRAS_API_KEY`.

Ambos backends exponen una interfaz compatible con OpenAI, lo que permite
encapsular su selección en una sola función `create_chat_client(model_name,
temperature=0.2)` que devuelve un cliente LangChain configurado. Los agentes
desconocen qué backend está activo: reciben el cliente ya construido y lo
usan a través de la interfaz uniforme de LangChain. Esta abstracción cumple
dos funciones simultáneas: permite re-ejecutar verificaciones cruzadas
sobre subconjuntos con un modelo de mayor capacidad sin tocar la lógica del
pipeline, y desacopla la elección del backend de la corrida principal.

## 5.8. Runner experimental

El runner principal (`experiments/run_experiments.py`) recibe como
argumentos la lista de configuraciones a evaluar, los benchmarks, las
seeds y el modelo, y construye una matriz cartesiana de ejecuciones. La
matriz completa contempla ocho configuraciones (baseline, sequential,
self_reflection_r1/r2/r3 y las tres ablaciones) por tres seeds por el
total de problemas del benchmark.

Tres propiedades del runner merecen mención: la resumibilidad, la
atomicidad y la tolerancia a fallos.

**Resumibilidad.** Antes de planificar la matriz, el runner lee los CSV
existentes y construye un conjunto de tuplas `(benchmark, problem_id, config,
seed)` ya completadas. La matriz efectiva excluye esos puntos. Reiniciar el
proceso tras una interrupción —cierre del portátil, fallo de Ollama,
SIGTERM— retoma el experimento donde quedó, sin re-ejecutar trabajo ya
realizado y sin manipulación manual del estado.

**Atomicidad.** El archivo `experiments/results/progress.json` se actualiza
tras cada ejecución completada mediante un patrón de escritura atómica:
se escribe primero en un fichero temporal `.json.tmp` y se renombra al
nombre final con `Path.replace()`. La operación POSIX `rename` es atómica
sobre el mismo sistema de ficheros, lo que garantiza que el dashboard de
seguimiento nunca observa estados parcialmente escritos del fichero de
progreso.

**Tolerancia a fallos.** Cada ejecución individual se envuelve en un
`try/except` amplio que captura cualquier excepción del pipeline, registra
el traceback en `experiments/logs/errors.log` y escribe una fila en el CSV
con el campo `error` poblado. El experimento global continúa con los
problemas siguientes. Esta política, contraria al fail-fast habitual en
desarrollo, es deliberada para corridas largas donde un fallo aislado en
un problema no debe abortar las miles de ejecuciones restantes.

## 5.9. Recorrido completo de un problema

Esta sección sigue, paso a paso, una invocación de la configuración
secuencial sobre el problema `HumanEval/1` (separación de grupos de
paréntesis). Su propósito es ilustrar cómo se van poblando los campos
del `AgentState` a lo largo del grafo y qué produce cada agente sobre el
mismo input. Los artefactos reproducidos son los efectivamente generados
por el sistema en una de las réplicas (`seed=42`); el formato se ajusta
para encajar en el ancho de la página, sin alterar el contenido.

### 5.9.1. Entrada al grafo

El benchmark proporciona el siguiente enunciado y firma de función:

```python
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of
    nested parentheses. Your goal is to separate those groups into
    separate strings and return the list of those. Separate groups are
    balanced (each open brace is properly closed) and not nested within
    each other. Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
```

Listado 5.2. Enunciado original de HumanEval/1.

El runner extrae las aserciones del test del benchmark y construye el
estado inicial del grafo con todos los campos de artefacto vacíos:

```python
initial_state = {
    "problem_id": "HumanEval/1",
    "problem_statement": "<enunciado completo del listado 5.2>",
    "function_signature": "separate_paren_groups",
    "test_cases": ["check(separate_paren_groups)"],
    "prd": "",
    "design_doc": "",
    "code_artifact": "",
    "test_results": {},
    "review_comments": "",
    "revision_count": 0,
    "tokens_input": 0,
    "tokens_output": 0,
    "latency_seconds": 0.0,
    "config_name": "sequential",
}
```

Listado 5.3. Estado inicial de la invocación.

### 5.9.2. Product Manager → PRD

El Product Manager recibe el `problem_statement` y emite el siguiente
PRD que escribe en `state["prd"]`:

```
(1) Problem Summary
Separate a string of multiple non-nested, balanced groups of
parentheses into a list of individual group strings. Whitespace must
be ignored.

(2) Acceptance Criteria
- The function returns a list of strings.
- Each returned string is itself a balanced parenthesis expression.
- The concatenation (ignoring whitespace) of the returned strings
  reconstructs the input.
- Whitespace characters in the input do not appear in any output.
- Empty input returns an empty list.

(3) Edge Cases
- Input with only whitespace → empty list.
- Single group → single-element list.
- Deeply nested but separate groups → still separated correctly.

(4) Out of scope
- Malformed input (unbalanced parentheses).
- Nested groups (the problem statement guarantees they are separate).
```

Listado 5.4. PRD producido por el Product Manager.

### 5.9.3. Arquitecto → Documento de diseño

El Arquitecto consume el PRD y la `function_signature` y devuelve un
documento de diseño técnico que escribe en `state["design_doc"]`:

```
(1) Algorithm choice
Single-pass scan over the input string, maintaining a running depth
counter and a buffer for the current group. When the counter returns
to zero, the buffered group is appended to the result list.

(2) Data structures
- An integer `depth` for the running parenthesis balance.
- A list[str] `groups` accumulating completed groups.
- A list[str] `current` (or string buffer) for the in-progress group.

(3) Implementation plan
- Strip whitespace from the input.
- Initialise depth = 0, groups = [], current = [].
- For each character c in the cleaned string:
    * Append c to current.
    * If c == '(', increment depth.
    * If c == ')', decrement depth.
    * If depth == 0 after the update, flush current to groups and
      reset current.
- Return groups.

(4) Failure modes
- The problem guarantees balanced and non-nested groups, so depth
  cannot go negative under valid input.
- Whitespace stripping must happen before the loop to avoid spurious
  empty groups.
```

Listado 5.5. Documento de diseño producido por el Arquitecto.

### 5.9.4. Developer → Código

El Developer toma el documento de diseño y la firma y emite el
artefacto de código que se escribe en `state["code_artifact"]`:

```python
def separate_paren_groups(paren_string: str) -> List[str]:
    # Strip whitespace first so the scan sees only parentheses.
    cleaned = paren_string.replace(" ", "")
    groups: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in cleaned:
        current.append(ch)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and current:
            groups.append("".join(current))
            current = []
    return groups
```

Listado 5.6. Implementación generada por el Developer.

### 5.9.5. QA Tester → Resultados de las pruebas

El QA Tester ejecuta el sandbox concatenando el `problem_statement`
(que contiene la importación `from typing import List`) con el
`code_artifact`, sigue el patrón `prompt + completion` de Chen et al.
(2021) y produce el siguiente mapa de resultados:

```python
state["test_results"] = {
    "check(separate_paren_groups)": True,
    "qa_summary": {"passed": 1, "failed": 0, "errors": 0},
}
```

Listado 5.7. `test_results` registrado por el QA Tester.

El sandbox se ejecuta en subproceso aislado con timeout de cinco
segundos y builtins restringidos según se describe en la sección 5.5.
No se invoca al LLM en esta etapa.

### 5.9.6. Code Reviewer → Veredicto y comentarios

El Code Reviewer recibe el código, los `test_results` y el documento
de diseño. Deriva primero el veredicto deterministamente: como todos
los tests pasan, el veredicto es `APPROVE`. A continuación el LLM
produce el comentario cualitativo. El campo `review_comments` queda:

```
VERDICT: APPROVE

(1) Issues found:
None. The implementation follows the design document precisely. The
whitespace handling is correct, the depth counter correctly identifies
group boundaries, and the loop terminates with an empty `current`
buffer.

The code is concise and self-documenting.
```

Listado 5.8. `review_comments` final del Reviewer.

### 5.9.7. Estado final y telemetría

Tras el último nodo del grafo, el estado contiene los cinco artefactos
poblados, los contadores de telemetría acumulados y los metadatos de
la corrida. La fila correspondiente del CSV es:

| Campo | Valor |
|---|---|
| `benchmark` | HE |
| `problem_id` | HumanEval/1 |
| `config` | sequential |
| `seed` | 42 |
| `pass_all_tests` | True |
| `test_pass_rate` | 1.0 |
| `tokens_input` | 1842 |
| `tokens_output` | 487 |
| `latency_seconds` | 247.8 |
| `revision_count` | 0 |
| `model` | qwen2.5-coder:7b-instruct-q4_K_M |

Tabla 5.4. Fila del CSV `sequential_results.csv` producida por la
invocación del recorrido.

El valor `revision_count = 0` refleja que en la configuración
secuencial nunca hay re-iteraciones del Developer; ese campo sólo es
relevante para la configuración con self-reflection (sección 5.4.3).
El coste total —2 329 tokens y ~248 segundos en la corrida local
sobre Apple Silicon M2— se compara con el coste del baseline para
este mismo problema (~5 segundos, ~280 tokens) en el análisis del
trade-off del capítulo 7.

Este recorrido demuestra dos propiedades del diseño que en abstracto
podrían parecer redundantes. La primera: cada campo del `AgentState`
tiene un productor único, lo que hace que la traza sea reproducible
y auditable problema a problema. La segunda: la derivación
determinista del veredicto desacopla el contrato del grafo del
formato libre que pueda emitir el LLM, garantizando que el router
condicional de la configuración con self-reflection siempre dispone
de una primera línea estable que leer.

## 5.10. Resumen del capítulo

El sistema implementado se compone de un estado compartido tipado, cinco
agentes de rol más un agente baseline, tres grafos principales —baseline,
secuencial y con self-reflection— y tres variantes de ablación construidas
sobre el mismo `AgentState`. Un sandbox de subproceso aislado cierra el
ciclo de evaluación funcional y un runner resumible orquesta las miles de
ejecuciones del experimento. Los siguientes capítulos describen el banco
experimental construido sobre esta implementación (capítulo 6) y los
resultados obtenidos hasta la fecha de cierre del documento (capítulo 7).
