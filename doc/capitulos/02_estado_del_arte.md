# Capítulo 2: Estado del Arte y Marco Teórico

## 2.1. Sistemas multi-agente: fundamentos clásicos

El concepto de agente autónomo no nació con los modelos de lenguaje. Tiene raíces bien establecidas en la inteligencia artificial distribuida de los años ochenta y noventa, y entender esa herencia es necesario para evaluar qué aportan los sistemas actuales y qué simplemente replican con otras herramientas.

Wooldridge y Jennings (1995) propusieron la definición que ha resultado más duradera: un agente es un sistema computacional situado en un entorno, capaz de actuar de forma autónoma para alcanzar sus objetivos. Esa definición implica cuatro propiedades. La **autonomía** —el agente opera sin intervención humana directa en cada decisión. La **reactividad** —percibe cambios en el entorno y responde a ellos. La **proactividad** —no se limita a reaccionar; toma iniciativa. Y la **sociabilidad** —interactúa con otros agentes de forma intencionada. Estas cuatro propiedades han guiado el diseño de agentes durante décadas, y siguen siendo el estándar de facto para caracterizar si un sistema merece ese nombre.

Un sistema multi-agente (MAS) es, en términos de Wooldridge (2009), un conjunto de agentes que interactúan en un entorno compartido. Lo que lo distingue de un sistema distribuido convencional no es la concurrencia de procesos en sí, sino la posibilidad de comportamientos colectivos que no están codificados en ningún agente individual. Esa propiedad emergente es lo que hace los MAS adecuados para problemas cuya complejidad supera lo que un agente único puede gestionar: planificación descentralizada, asignación de recursos, negociación, o resolución colaborativa de tareas.

El modelo de agente más estudiado formalmente es el BDI (*Beliefs, Desires, Intentions*). Sus bases filosóficas vienen del trabajo de Bratman (1987) sobre teoría de la acción racional: un agente no actúa con información completa ni recalcula su comportamiento desde cero en cada instante, sino que adopta compromisos y los mantiene mientras sean viables. Rao y Georgeff (1995) trasladaron esta idea a una arquitectura computacional concreta: las *beliefs* representan el conocimiento del agente sobre el mundo, los *desires* son los estados que quiere alcanzar, y las *intentions* son los planes actualmente en ejecución. El modelo BDI captura una tensión real: entre la capacidad de replanificar y el coste de hacerlo continuamente.

La coordinación entre agentes es el otro problema central. Cuando comparten recursos o tienen metas parcialmente conflictivas, necesitan comunicación explícita. Cohen y Levesque (1990) formalizaron la noción de intención comunicativa: un agente no solo transmite información, sino que actúa sobre el estado mental del receptor. Esta distinción —entre intercambiar datos y realizar *actos de habla*— fue la base sobre la que se construyeron los protocolos de comunicación inter-agente, como los especificados por la Foundation for Intelligent Physical Agents (FIPA) a finales de los noventa. Stone y Veloso (2000) extendieron esta visión hacia el aprendizaje: los agentes no solo pueden coordinarse con protocolos predefinidos, sino aprender estrategias de coordinación mediante interacción repetida.

Estos fundamentos —autonomía, comunicación estructurada, especialización de roles, coordinación emergente— son los que los frameworks modernos de agentes LLM intentan recuperar. Pero el sustrato es completamente distinto: en lugar de agentes con arquitecturas simbólicas y protocolos formales, ahora se trabaja con modelos de lenguaje cuyo comportamiento emerge del preentrenamiento y del prompting. La siguiente sección examina qué significa exactamente que un LLM actúe como agente, y qué propiedades del agente clásico se conservan, cuáles se transforman, y cuáles desaparecen.

### Referencias 

Bratman, M. E. (1987). *Intention, plans, and practical reason*. Harvard University Press.

Cohen, P. R., & Levesque, H. J. (1990). Intention is choice with commitment. *Artificial Intelligence*, *42*(2–3), 213–261. https://doi.org/10.1016/0004-3702(90)90055-5

Rao, A. S., & Georgeff, M. P. (1995). BDI agents: From theory to practice. En *Proceedings of the First International Conference on Multiagent Systems (ICMAS-95)* (pp. 312–319). AAAI Press.

Stone, P., & Veloso, M. (2000). Multiagent systems: A survey from a machine learning perspective. *Autonomous Robots*, *8*(3), 345–383. https://doi.org/10.1023/A:1008942012299

Wooldridge, M. (2009). *An introduction to multiagent systems* (2nd ed.). Wiley.

Wooldridge, M., & Jennings, N. R. (1995). Intelligent agents: Theory and practice. *The Knowledge Engineering Review*, *10*(2), 115–152. https://doi.org/10.1017/S0269888900008122

## 2.2. Modelos de Lenguaje de Gran Tamaño como agentes

Los modelos de lenguaje de gran tamaño (LLM) surgieron inicialmente como sistemas
de predicción textual. A partir de 2022 quedó patente que modelos de escala
suficiente exhiben capacidades emergentes que los hacen viables como núcleos
cognitivos de agentes autónomos: razonamiento encadenado, planificación de pasos,
uso de herramientas externas y autoevaluación de resultados. Dos revisiones
sistemáticas recientes —Xi et al. (2023) y Wang et al. (2024)— coinciden en
identificar tres componentes funcionales en un agente LLM: un módulo de percepción,
un módulo de razonamiento y un módulo de acción, con mecanismos de memoria y
planificación como dimensiones de variación entre arquitecturas.

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

El uso de herramientas externas recibió tratamiento sistemático con Toolformer
(Schick et al., 2023). En ese trabajo se mostró que un LLM puede aprender de forma
autosupervisada a insertar llamadas a APIs dentro de su propia generación de texto.
Este resultado establece una base técnica para equipar a los agentes con capacidades
que trascienden la generación textual: consulta de bases de datos, ejecución de
código, acceso a información en tiempo real.

En conjunto, estos trabajos definen el repertorio de capacidades sobre el que se
construyen los sistemas multi-agente modernos. La pregunta que se abre de inmediato
es cómo coordinar múltiples instancias de estos agentes de forma que cooperen en la
resolución de tareas que ninguno podría abordar individualmente; ese es el objeto de
los frameworks que se examinan en la siguiente subsección.

### Referencias
 
- Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N. y Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *Advances in Neural Information Processing Systems*, 36.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. y Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems*, 36.
- Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., ... y Wen, J. (2024). A Survey on Large Language Model based Autonomous Agents. *Frontiers of Computer Science*, 18(6), 186345.
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q. y Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems*, 35, 24824–24837.
- Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., ... et al. (2023). The Rise and Potential of Large Language Model Based Agents: A Survey. *arXiv preprint arXiv:2309.07864*.
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. y Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *International Conference on Learning Representations*.

## 2.3. Frameworks multi-agente basados en LLM

El agente individual descrito en la sección anterior tiene un límite cognitivo
evidente: la longitud finita de su ventana de contexto y la ausencia de
especialización. Una tarea compleja de ingeniería de software —análisis de
requisitos, diseño de arquitectura, implementación, revisión de código, generación
de pruebas— implica competencias heterogéneas que difícilmente coexisten en un único
prompt. La respuesta natural es dividir el trabajo entre agentes con roles distintos,
coordinados por un mecanismo de orquestación. Esta idea es el núcleo de los
frameworks multi-agente basados en LLM que han proliferado desde 2023.

El primero en formalizar el concepto de rol como mecanismo de especialización fue
CAMEL (Li et al., 2023). En CAMEL dos agentes —uno con rol de «instructor» y otro de
«asistente»— mantienen una conversación dirigida a resolver una tarea. El protocolo
de intercambio sigue un ciclo estricto: el instructor emite instrucciones, el
asistente las ejecuta y reporta, el instructor evalúa y emite la siguiente instrucción.
Este trabajo mostró que la asignación de roles mediante prompts de sistema es suficiente
para producir comportamientos cooperativos estables, sin necesitar mecanismos de
coordinación explícitos fuera del propio lenguaje natural.

ChatDev (Qian et al., 2023) trasladó esta arquitectura al dominio específico del
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

El sustrato tecnológico que hace posible esta combinación es LangGraph, cuya base
teórica se examina en la subsección 2.6. Antes, conviene revisar el estado del arte
en generación automática de código, dado que esa es la capacidad central que los
agentes del sistema deben ejercer.

### Referencias

- Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., ... y
  Schmidhuber, J. (2024). MetaGPT: Meta Programming for a Multi-Agent
  Collaborative Framework. *International Conference on Learning Representations*.
- Li, G., Hammoud, H., Itani, H., Khizbullin, D. y Ghanem, B. (2023). CAMEL:
  Communicating Agents for "Mind" Exploration of Large Language Model Society.
  *Advances in Neural Information Processing Systems*, 36.
- Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., Liu, Z. y Sun, M.
  (2024). Communicative Agents for Software Development. *Proceedings of the
  62nd Annual Meeting of the Association for Computational Linguistics*,
  15174–15186.
- Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., ... y Wang, C. (2023).
  AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.
  *arXiv preprint arXiv:2308.08155*.

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

### Referencias

- Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D.,
  Jiang, E., Cai, C., Terry, M., Le, Q. y Sutton, C. (2021). Program Synthesis
  with Large Language Models. *arXiv preprint arXiv:2108.07732*.
- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ...
  y Zaremba, W. (2021). Evaluating Large Language Models Trained on Code.
  *arXiv preprint arXiv:2107.03374*.
- Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., ...
  y Vinyals, O. (2022). Competition-Level Code Generation with AlphaCode.
  *Science*, 378(6624), 1092–1097.
- Rozière, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., ...
  y Synnaeve, G. (2023). Code Llama: Open Foundation Models for Code.
  *arXiv preprint arXiv:2308.12950*.

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

Ese resultado tiene consecuencias directas para el diseño del sistema propuesto
en este TFG. Si un agente único no resuelve SWE-bench de forma fiable, la
hipótesis es que distribuir el trabajo entre agentes especializados —uno que
localiza el error, otro que propone el parche, otro que verifica la regresión—
puede mejorar la tasa de éxito. Por ese motivo, SWE-bench es el benchmark de
referencia en la evaluación del sistema. Los mecanismos de orquestación que hacen
posible esa distribución se examinan en la subsección siguiente.

### Referencias

- Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., ...
  y Sutton, C. (2021). Program Synthesis with Large Language Models. *arXiv
  preprint arXiv:2108.07732*.
- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ...
  y Zaremba, W. (2021). Evaluating Large Language Models Trained on Code.
  *arXiv preprint arXiv:2107.03374*.
- Jiménez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O. y
  Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World
  GitHub Issues? *International Conference on Learning Representations*.
- Liu, J., Xia, C. S., Wang, Y. y Zhang, L. (2023). Is Your Code Generated by
  ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code.
  *Advances in Neural Information Processing Systems*, 36.

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

### Referencias

- LangChain Inc. (2024). *LangGraph: Build Stateful, Multi-Actor Applications
  with LLMs*. https://github.com/langchain-ai/langgraph
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems* (2.ª ed.).
  Wiley.

## 2.7. Síntesis y posicionamiento del trabajo

La revisión de este capítulo recorre un arco que va desde la definición clásica
de agente autónomo (Wooldridge y Jennings, 1995) hasta los frameworks actuales de
orquestación multi-agente basados en LLM. El hilo conductor es un problema que
persiste a lo largo de toda esa trayectoria: cómo coordinar agentes especializados
para que produzcan resultados que ninguno alcanzaría por separado.

De esa revisión se derivan tres observaciones que motivan el diseño de este
trabajo. Los modelos de lenguaje generan código plausible pero carecen de
verificación interna; la corrección depende de ejecución contra pruebas o de un
segundo agente que revise la salida. Los frameworks como ChatDev y MetaGPT
muestran que la especialización por roles mejora la coherencia del resultado, pero
sus flujos de control son rígidos y no admiten iteración condicional. SWE-bench,
por su parte, evidencia que la brecha entre resolver ejercicios sintéticos y
resolver issues reales de software sigue siendo grande para cualquier sistema
actual.

El sistema propuesto en este TFG responde a esas limitaciones con un diseño
concreto: topología jerárquica con agente supervisor, flujo de control condicional
implementado en LangGraph y evaluación sobre SWE-bench como referencia empírica.
El objetivo no es proponer una arquitectura sin precedentes, sino estudiar de
forma controlada si la orquestación mediante grafos de estado con roles
especializados produce mejoras medibles sobre un agente único en tareas reales de
ingeniería de software. El capítulo siguiente describe ese diseño.

### Referencias

- Wooldridge, M. y Jennings, N. R. (1995). Intelligent Agents: Theory and
  Practice. *The Knowledge Engineering Review*, 10(2), 115–152.