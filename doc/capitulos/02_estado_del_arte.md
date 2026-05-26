# Capítulo 2: Estado del Arte y Marco Teórico

## 2.1. Sistemas multi-agente: fundamentos clásicos

El concepto de agente autónomo no nació con los modelos de lenguaje. Tiene raíces bien establecidas en la inteligencia artificial distribuida de los años ochenta y noventa, y entender esa herencia es necesario para evaluar qué aportan los sistemas actuales y qué simplemente replican con otras herramientas.

Wooldridge y Jennings (1995) propusieron la definición que ha resultado más duradera: un agente es un sistema computacional situado en un entorno, capaz de actuar de forma autónoma para alcanzar sus objetivos. Esa definición implica cuatro propiedades. La **autonomía** —el agente opera sin intervención humana directa en cada decisión. La **reactividad** —percibe cambios en el entorno y responde a ellos. La **proactividad** —no se limita a reaccionar; toma iniciativa. Y la **sociabilidad** —interactúa con otros agentes de forma intencionada. Estas cuatro propiedades han guiado el diseño de agentes durante décadas, y siguen siendo el estándar de facto para caracterizar si un sistema merece ese nombre.

Un sistema multi-agente (MAS) es, en términos de Wooldridge (2009), un conjunto de agentes que interactúan en un entorno compartido. Lo que lo distingue de un sistema distribuido convencional no es la concurrencia de procesos en sí, sino la posibilidad de comportamientos colectivos que no están codificados en ningún agente individual. Esa propiedad emergente es lo que hace los MAS adecuados para problemas cuya complejidad supera lo que un agente único puede gestionar: planificación descentralizada, asignación de recursos, negociación, o resolución colaborativa de tareas.

El modelo de agente más estudiado formalmente es el BDI (*Beliefs, Desires, Intentions*). Sus bases filosóficas vienen del trabajo de Bratman (1987) sobre teoría de la acción racional: un agente no actúa con información completa ni recalcula su comportamiento desde cero en cada instante, sino que adopta compromisos y los mantiene mientras sean viables. Rao y Georgeff (1995) trasladaron esta idea a una arquitectura computacional concreta: las *beliefs* representan el conocimiento del agente sobre el mundo, los *desires* son los estados que quiere alcanzar, y las *intentions* son los planes actualmente en ejecución. El modelo BDI captura una tensión real: entre la capacidad de replanificar y el coste de hacerlo continuamente.

La coordinación entre agentes es el otro problema central. Cuando comparten recursos o tienen metas parcialmente conflictivas, necesitan comunicación explícita. Cohen y Levesque (1990) formalizaron la noción de intención comunicativa: un agente no solo transmite información, sino que actúa sobre el estado mental del receptor. Esta distinción —entre intercambiar datos y realizar *actos de habla*— fue la base sobre la que se construyeron los protocolos de comunicación inter-agente, como los especificados por la Foundation for Intelligent Physical Agents (FIPA) a finales de los noventa. Stone y Veloso (2000) extendieron esta visión hacia el aprendizaje: los agentes no solo pueden coordinarse con protocolos predefinidos, sino aprender estrategias de coordinación mediante interacción repetida.

Estos fundamentos —autonomía, comunicación estructurada, especialización de roles, coordinación emergente— son los que los frameworks modernos de agentes LLM intentan recuperar. Pero el sustrato es completamente distinto: en lugar de agentes con arquitecturas simbólicas y protocolos formales, ahora se trabaja con modelos de lenguaje cuyo comportamiento emerge del preentrenamiento y del prompting. La siguiente sección examina qué significa exactamente que un LLM actúe como agente, y qué propiedades del agente clásico se conservan, cuáles se transforman, y cuáles desaparecen.

## 2.2. Modelos de Lenguaje de Gran Tamaño como agentes

Los modelos de lenguaje de gran tamaño (LLM) nacieron como sistemas de
predicción de la siguiente palabra. A partir de 2022 empezó a verse que, por
encima de cierta escala, esos modelos hacían cosas que no se les había
entrenado a hacer de forma explícita: razonar paso a paso, planificar
acciones, llamar a herramientas externas, evaluar su propia salida. Esas
capacidades emergentes son las que abrieron la puerta a usarlos como núcleo
cognitivo de un agente. Las dos revisiones sistemáticas más citadas —Xi et al.
(2023) y Wang et al. (2024)— convergen en una descomposición funcional de
tres bloques: percepción, razonamiento y acción, con memoria y planificación
como dimensiones donde las arquitecturas se diferencian.

El punto de inflexión conceptual fue la introducción del prompting con cadena de
pensamiento (*Chain-of-Thought*, CoT). Wei et al. (2022) demostraron que instruir a
un LLM para que descomponga un problema en pasos intermedios verbalizados mejora
sustancialmente su rendimiento en tareas de razonamiento aritmético, de sentido
común y simbólico. Esta técnica no modifica los pesos del modelo; actúa sobre el
proceso de inferencia mediante el formato del prompt. Su relevancia para los sistemas
agénticos reside en que hace explícito el razonamiento interno del agente, lo cual
facilita tanto la depuración como la coordinación con otros agentes.

El siguiente avance estructural fue el framework ReAct (Yao et al., 2023). ReAct
intercala razonamiento y acción en el ciclo de generación del LLM: el modelo alterna
entre pensamientos internos y llamadas a herramientas externas —motores de búsqueda,
APIs, intérpretes de código—, incorporando las observaciones resultantes en el
contexto antes de continuar. Este patrón observar-razonar-actuar es, en la práctica,
una implementación del bucle percepción-deliberación-acción propio de los agentes
BDI descritos en la sección anterior, aunque sin la formalización lógica de aquellos.

Una limitación del ciclo ReAct es que los errores no se corrigen automáticamente: un
razonamiento incorrecto se propaga sin mecanismo de revisión. Shinn et al. (2023)
propusieron Reflexion como respuesta a este problema: el agente genera una respuesta,
recibe una señal evaluativa —procedente de un entorno externo o de un segundo LLM
actuando como crítico— y produce una reflexión verbal que se incorpora al contexto de
la siguiente iteración. Reflexion transforma así el ciclo de retroalimentación en un
proceso de refinamiento lingüístico sin necesidad de actualizar los parámetros del
modelo.

Toolformer (Schick et al., 2023) sistematizó el uso de herramientas externas.
El trabajo mostró que un LLM puede aprender, de forma autosupervisada, a
insertar llamadas a APIs en mitad de su propia generación. Eso es lo que
permite que un agente haga cosas que la generación de texto pura no alcanza:
consultar una base de datos, ejecutar código, acceder a información que el
modelo no tiene en sus pesos.

Estos cuatro trabajos —CoT, ReAct, Reflexion, Toolformer— definen el
repertorio sobre el que se han construido los sistemas multi-agente actuales.
La siguiente pregunta es de coordinación: cómo hacer que varios de estos
agentes cooperen en tareas que ninguno resolvería por separado. Los
frameworks que han intentado responderla se examinan a continuación.

## 2.3. Frameworks multi-agente basados en LLM

El agente individual de la sección anterior tiene un techo claro: la ventana
de contexto es finita y un único prompt no puede ser, a la vez, especialista
en requisitos, diseño, implementación, pruebas y revisión. Una tarea de
ingeniería de software de cierta complejidad exige todas esas competencias.
Lo natural es repartirlas entre agentes con roles distintos y dejar la
coordinación a un orquestador externo. Esa es la idea sobre la que se han
construido los frameworks multi-agente con LLM que han ido apareciendo desde
2023.

El primero en formalizar el concepto de rol como mecanismo de especialización fue
CAMEL (Li et al., 2023). En CAMEL dos agentes —uno con rol de «instructor» y otro de
«asistente»— mantienen una conversación dirigida a resolver una tarea. El protocolo
de intercambio sigue un ciclo estricto: el instructor emite instrucciones, el
asistente las ejecuta y reporta, el instructor evalúa y emite la siguiente instrucción.
Este trabajo mostró que la asignación de roles mediante prompts de sistema es suficiente
para producir comportamientos cooperativos estables, sin necesitar mecanismos de
coordinación explícitos fuera del propio lenguaje natural.

ChatDev (Qian et al., 2024) trasladó esta arquitectura al dominio específico del
desarrollo de software. En ChatDev, distintos agentes adoptan roles propios de un
equipo de ingeniería —director ejecutivo, jefe de producto, ingeniero de software,
revisor de código, tester— y se comunican a través de una secuencia de fases de
chat predefinidas. Cada fase produce artefactos concretos: un documento de
requisitos, un diseño de clases, código fuente, un informe de pruebas. La evaluación
empírica del trabajo demostró que este flujo produce software funcional en tareas de
programación de complejidad moderada con mayor consistencia que un agente único.
ChatDev es, en términos de dominio, el precedente más directo del sistema
desarrollado en este TFG.

MetaGPT (Hong et al., 2024) profundizó en la dimensión estructural. Su aportación
central es la noción de procedimiento operativo estándar (SOP) como mecanismo de
coordinación: los roles de los agentes no solo definen quién hace qué, sino en qué
orden, con qué entradas esperadas y qué artefactos deben producir como salida. Esta
formalización reduce la ambigüedad en el flujo de información entre agentes y mejora
la reproducibilidad de los resultados. MetaGPT introduce además una memoria
compartida estructurada —basada en un repositorio de mensajes y artefactos accesible
por todos los agentes— que resuelve parcialmente el problema de la fragmentación del
contexto en sistemas con muchos agentes.

AutoGen (Wu et al., 2023) propone un enfoque distinto: en lugar de fijar de antemano
el grafo de interacciones, ofrece un marco conversacional flexible en el que cualquier
agente puede iniciar, responder o interrumpir la conversación según condiciones
definidas por el diseñador. AutoGen introduce también el patrón human-in-the-loop de
forma explícita: un agente puede delegar una decisión al operador humano cuando
detecta ambigüedad o riesgo, lo que lo hace adecuado para entornos de desarrollo
asistido. Desde el punto de vista de la arquitectura, AutoGen se aleja de las
topologías secuenciales fijas y se acerca a una red de agentes con capacidad de
enrutamiento dinámico.

De la revisión de estos frameworks se extraen dos dimensiones de clasificación
relevantes para este trabajo. La primera es la topología de coordinación: los sistemas
pueden organizarse de forma secuencial (pipeline), jerárquica (con un agente
supervisor) o como red plana con enrutamiento dinámico. La segunda es el grado de
estructuración del flujo: desde protocolos fijos con SOPs estrictos (ChatDev,
MetaGPT) hasta conversaciones abiertas con condiciones de terminación configurables
(AutoGen). El sistema propuesto en este TFG se posiciona en la intersección: utiliza
una topología jerárquica con supervisor y un flujo parcialmente estructurado mediante
un grafo de estado, lo que permite tanto reproducibilidad como adaptabilidad ante
tareas imprevistas.

La tabla 2.1 sintetiza las características de los cuatro frameworks
revisados en esta subsección, lo que permite situar el TFG por contraste
con cada uno de ellos sin necesidad de releer las descripciones en prosa.

| Framework | Topología | Comunicación | Roles fijos | Flujo de control | Determinismo | Estado tipado |
|---|---|---|---|---|---|---|
| AutoGen (Wu et al., 2023) | Red sin jerarquía | Conversación libre | No | Dinámico, dependiente del LLM | Bajo | No |
| CAMEL (Li et al., 2023) | Diádica (2 agentes) | Conversación bilateral | Sí (2) | Lineal | Bajo | No |
| ChatDev (Qian et al., 2024) | Cascada por fases | Estructurada por fase | Sí (5+) | Lineal (cascada de waterfall) | Medio | Parcial |
| MetaGPT (Hong et al., 2024) | Jerárquica con supervisor | SOPs estructuradas + artefactos | Sí (5) | Lineal con SOPs | Alto | Sí |
| **TFG (este trabajo)** | Grafo de estado explícito | Estado compartido tipado | Sí (5) | Lineal + arista condicional | **Alto** | **Sí (TypedDict)** |

Tabla 2.1. Posicionamiento del sistema propuesto frente a los frameworks
multi-agente más representativos de la literatura.

El sustrato tecnológico que hace posible esta combinación es LangGraph,
cuya base teórica se examina en la subsección 2.6. Antes, conviene
revisar el estado del arte en generación automática de código, dado que
esa es la capacidad central que los agentes del sistema deben ejercer.

## 2.4. Generación automática de código con LLMs

La capacidad de los LLM para generar código fuente correcto —no solo
sintácticamente válido— quedó delimitada con precisión por Chen et al. (2021) en
el trabajo que introdujo Codex. El modelo, derivado de GPT-3 y ajustado sobre
código público de GitHub, podía producir implementaciones en Python a partir de
descripciones en lenguaje natural. Más que el modelo en sí, lo que aportó ese
trabajo fue la métrica *pass@k*: la probabilidad de que al menos una de las k
soluciones generadas supere el conjunto de pruebas del problema. Esa métrica
señala algo que la exactitud sintáctica no captura: que el código sea ejecutable
y se comporte como se espera.

Austin et al. (2021) añadieron el benchmark MBPP (Mostly Basic Python Problems),
374 problemas con especificación en lenguaje natural y pruebas unitarias
asociadas. Lo que mostraron sus resultados fue un patrón que se repetiría en
estudios posteriores: los modelos generan código plausible sin dificultad, pero
la tasa de superación cae en cuanto el problema exige razonar sobre estructuras
de datos no triviales o manejar casos límite.

AlphaCode (Li et al., 2022) abordó ese problema desde otro ángulo: generar hasta
un millón de candidatos por problema y filtrar los que pasan las pruebas públicas
disponibles. En benchmarks de competición algorítmica los resultados fueron
notables, pero el coste computacional hace el enfoque inviable fuera de ese
contexto específico.

Code Llama (Rozière et al., 2023) cambió el tipo de pregunta que vale la pena
hacerse. En lugar de intentar maximizar el rendimiento en competición, el trabajo
produjo una familia de modelos de código abierto con rendimiento comparable a
Codex en HumanEval, con variantes especializadas en completado de fragmentos
(*fill-in-the-middle*) e instrucción. El resultado práctico fue que la generación
de código dejó de depender de APIs externas y pasó a ser desplegable en entornos
locales.

Lo que el conjunto de estos trabajos deja sin resolver es la verificación. Un
agente que genera código no tiene forma interna de saber si lo que ha producido es
correcto; solo sabe que es sintácticamente válido. Para detectar errores se
necesita o bien ejecución contra pruebas, o bien un segundo agente que revise la
salida. Esa limitación es una de las razones concretas por las que el sistema
propuesto en este TFG distribuye la tarea de generación y la de verificación en
nodos distintos del grafo de orquestación. Los benchmarks que permiten medir esa
capacidad de forma objetiva se examinan en la subsección siguiente.

## 2.5. Benchmarks de evaluación de generación de código

Medir si un modelo genera código correcto requiere algo más que comprobar si
compila. Los benchmarks de código intentan resolver ese problema con distintos
niveles de ambición, y la evolución de esas propuestas dice bastante sobre los
límites que se han ido descubriendo en los modelos.

HumanEval (Chen et al., 2021) y MBPP (Austin et al., 2021), ya presentados en
la sección anterior, establecieron el formato estándar: especificación en lenguaje
natural, solución generada por el modelo, verificación mediante pruebas unitarias.
Son reproducibles y comparables entre sistemas. El problema es que sus conjuntos
de pruebas son pequeños —HumanEval incluye entre 7 y 8 casos por problema de
media— y no cubren casos límite con suficiente densidad. Liu et al. (2023)
cuantificaron ese déficit en EvalPlus: al ampliar los conjuntos de pruebas entre
80 y 125 veces más casos por problema, el rendimiento de los modelos cae de forma
significativa. Las cifras de pass@k publicadas sobre HumanEval sobreestiman la
corrección real.

Hay además un problema más estructural. Los ejercicios de HumanEval y MBPP son
problemas de algoritmia diseñados para la evaluación, no tareas representativas
del trabajo de ingeniería de software. Escribir una función que invierta una lista
es diferente a localizar y corregir un bug en una base de código con historial de
cambios, dependencias externas y pruebas de regresión.

SWE-bench (Jiménez et al., 2024) intentó cerrar esa distancia. El benchmark
recopila 2.294 issues reales de repositorios de GitHub en Python, cada uno con
una descripción del problema y un conjunto de pruebas que validan el parche
resultante. El modelo debe generar un diff que resuelva el issue sin romper el
resto del código. Los resultados iniciales del paper —con los mejores modelos
resolviendo menos del 4% de los problemas— dejan claro que la brecha entre
generar funciones aisladas y resolver tareas reales de ingeniería de software
es grande.

El sistema propuesto en este TFG se evalúa sobre HumanEval, no sobre SWE-bench.
La razón es práctica: ejecutar SWE-bench requiere un harness Docker por
instancia y un coste de cómputo incompatible con los recursos disponibles para
un trabajo de fin de grado. HumanEval ofrece, en cambio, un banco controlado
que permite aislar el efecto de la orquestación multi-agente sin que el ruido
de la infraestructura domine los resultados. La hipótesis que se contrasta es,
por tanto, más acotada: si distribuir el trabajo entre agentes especializados
—uno que redacta requisitos, otro que diseña, otro que implementa, otro que
prueba, otro que revisa— mejora la tasa de superación de pruebas unitarias
sobre funciones aisladas. SWE-bench queda como referencia del techo de
dificultad que cualquier sistema de este tipo aspira a abordar a medio plazo.

La tabla 2.2 contrasta los cinco benchmarks más relevantes de la literatura
en términos de granularidad, cardinalidad y tipo de tarea. Aporta el
marco de comparación que justifica las decisiones de scope tomadas en el
capítulo 3.

| Benchmark | Granularidad | Problemas | Verificación | Cardinalidad infra |
|---|---|---|---|---|
| HumanEval (Chen et al., 2021) | Función | 164 | Tests unitarios | Sandbox simple |
| MBPP (Austin et al., 2021) | Función | 974 | Tests unitarios | Sandbox simple |
| APPS (Hendrycks et al., 2021) | Función | 10 000 | Tests unitarios | Sandbox simple |
| ClassEval (Du et al., 2023) | Clase completa | 100 | Tests unitarios | Sandbox + imports |
| SWE-bench (Jiménez et al., 2024) | Proyecto multi-fichero | 2 294 | Suite del repo | Docker por repo |

Tabla 2.2. Comparativa de benchmarks de generación de código. La columna
"Cardinalidad infra" indica el coste de infraestructura necesario para
ejecutar la evaluación, que crece desde un sandbox Python aislado hasta
un harness Docker por instancia.

Los mecanismos de orquestación que hacen posible la distribución del
trabajo entre agentes especializados se examinan en la subsección
siguiente.

## 2.6. Orquestación basada en grafos de estado

Los frameworks descritos en la sección 2.3 comparten un problema de diseño: el
flujo de control entre agentes es en gran medida fijo. Las fases de ChatDev siguen
un orden predefinido; las conversaciones de CAMEL tienen inicio y fin claros.
Cuando una tarea exige condicionar el siguiente paso al resultado del anterior
—iterar sobre un fragmento de código hasta que pase las pruebas, o derivar el
trabajo a un agente distinto según el tipo de error detectado— estas arquitecturas
no tienen un mecanismo limpio para expresarlo.

LangGraph (LangChain Inc., 2024) aborda ese problema modelando el flujo de
orquestación como un grafo dirigido de estados finitos. Cada nodo es una unidad
de procesamiento —en el contexto de este TFG, un agente especializado— y cada
arista define una transición entre nodos. Las aristas pueden ser incondicionales
o condicionales: en el segundo caso, la transición que se activa depende del
contenido del estado compartido en ese momento. Eso permite expresar ciclos de
revisión, bifurcaciones según el tipo de tarea y condiciones de terminación
complejas, sin codificar esa lógica dentro de los propios agentes.

El estado compartido es el otro elemento central del modelo. LangGraph mantiene
un objeto de estado mutable que todos los nodos pueden leer y escribir según sus
responsabilidades. La información producida por un agente no se pierde entre
llamadas, sino que persiste en el estado del grafo hasta que el flujo termina.
Esto resuelve el problema de fragmentación de contexto que afecta a los sistemas
basados en conversaciones encadenadas.

Esta arquitectura conecta con la noción de sistema multi-agente coordinado por un
mecanismo de control explícito, tal como la formuló Wooldridge (2009). La
diferencia es que en LangGraph ese mecanismo no es externo al sistema sino parte
de su definición formal. Cómo este modelo se articula con el resto de elementos
revisados en el capítulo es lo que sintetiza la sección siguiente.

## 2.7. Síntesis y posicionamiento del trabajo

Este capítulo ha recorrido un arco largo: desde la definición clásica de
agente autónomo de Wooldridge y Jennings (1995) hasta los frameworks actuales
de orquestación multi-agente sobre LLM. El problema que aparece en todas las
etapas es el mismo: cómo coordinar agentes especializados para que produzcan,
en conjunto, algo que ninguno alcanzaría por separado.

De esa revisión salen tres observaciones que justifican el diseño del
sistema. Primero, los LLM generan código plausible pero no saben si es
correcto; la verificación tiene que venir de fuera, vía ejecución contra
pruebas o vía un segundo agente revisor. Segundo, frameworks como ChatDev y
MetaGPT confirman que la especialización por roles mejora la coherencia del
resultado, pero su flujo de control es lineal y no permite iteración
condicional. Tercero, SWE-bench deja claro que la distancia entre resolver
ejercicios aislados y resolver issues reales de software es enorme, y que
ningún sistema actual la cierra de forma fiable.

Conviene, sin embargo, no asumir como artículo de fe que más agentes y
más llamadas al modelo equivalgan, sin más, a mejores resultados. Una
línea reciente de la literatura empieza a cuestionar esa lectura. Chen
et al. (2024) muestran que, en sistemas de inferencia compuestos por
múltiples llamadas a un LLM, el rendimiento agregado puede degradarse
respecto al uso de una única llamada, especialmente cuando los errores
intermedios se propagan y los criterios de filtrado no son estrictos.
Olausson et al. (2024) llegan a una conclusión análoga para
self-repair de código: la auto-corrección con un modelo único no es
una *silver bullet*, y su efecto es más débil cuanto menor es la
capacidad del modelo base. Huang et al. (2024) reportan en AgentCoder
que las ganancias del multi-agente sobre HumanEval dependen de la
calidad del agente de pruebas y se diluyen con modelos pequeños. Estos
trabajos no invalidan los frameworks revisados antes; matizan el
optimismo con el que suelen presentarse y delimitan el régimen en el
que la coordinación entre agentes aporta valor. El experimento de
este TFG aporta evidencia empírica adicional a esa discusión, como
se documenta en los capítulos 7 y 8.

El sistema propuesto en este TFG responde a esas limitaciones con un diseño
concreto: cinco agentes especializados (PM, Arquitecto, Developer, QA, Code
Reviewer) coordinados por un grafo de estado en LangGraph, con un bucle de
revisión condicional Reviewer → Developer acotado por un número máximo de
iteraciones. La evaluación se hace sobre HumanEval con tres configuraciones
incrementales (baseline, secuencial sin ciclo, secuencial con self-reflection)
y un conjunto de ablaciones por rol. El objetivo no es proponer una
arquitectura sin precedentes, sino medir, de forma controlada y reproducible,
qué aporta cada decisión de diseño sobre la tasa de superación, el coste en
tokens y la latencia. El capítulo siguiente concreta ese diseño en una serie
de objetivos verificables.
