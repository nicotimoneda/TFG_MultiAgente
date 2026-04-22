# Capítulo 2: Estado del Arte y Marco Teórico

## 2.1. Sistemas multi-agente: fundamentos clásicos

El concepto de agente autónomo no nació con los modelos de lenguaje. Tiene raíces bien establecidas en la inteligencia artificial distribuida de los años ochenta y noventa, y entender esa herencia es necesario para evaluar qué aportan los sistemas actuales y qué simplemente replican con otras herramientas.

Wooldridge y Jennings (1995) propusieron la definición que ha resultado más duradera: un agente es un sistema computacional situado en un entorno, capaz de actuar de forma autónoma para alcanzar sus objetivos. Esa definición implica cuatro propiedades. La **autonomía** —el agente opera sin intervención humana directa en cada decisión. La **reactividad** —percibe cambios en el entorno y responde a ellos. La **proactividad** —no se limita a reaccionar; toma iniciativa. Y la **sociabilidad** —interactúa con otros agentes de forma intencionada. Estas cuatro propiedades han guiado el diseño de agentes durante décadas, y siguen siendo el estándar de facto para caracterizar si un sistema merece ese nombre.

Un sistema multi-agente (MAS) es, en términos de Wooldridge (2009), un conjunto de agentes que interactúan en un entorno compartido. Lo que lo distingue de un sistema distribuido convencional no es la concurrencia de procesos en sí, sino la posibilidad de comportamientos colectivos que no están codificados en ningún agente individual. Esa propiedad emergente es lo que hace los MAS adecuados para problemas cuya complejidad supera lo que un agente único puede gestionar: planificación descentralizada, asignación de recursos, negociación, o resolución colaborativa de tareas.

El modelo de agente más estudiado formalmente es el BDI (*Beliefs, Desires, Intentions*). Sus bases filosóficas vienen del trabajo de Bratman (1987) sobre teoría de la acción racional: un agente no actúa con información completa ni recalcula su comportamiento desde cero en cada instante, sino que adopta compromisos y los mantiene mientras sean viables. Rao y Georgeff (1995) trasladaron esta idea a una arquitectura computacional concreta: las *beliefs* representan el conocimiento del agente sobre el mundo, los *desires* son los estados que quiere alcanzar, y las *intentions* son los planes actualmente en ejecución. El modelo BDI captura una tensión real: entre la capacidad de replanificar y el coste de hacerlo continuamente.

La coordinación entre agentes es el otro problema central. Cuando comparten recursos o tienen metas parcialmente conflictivas, necesitan comunicación explícita. Cohen y Levesque (1990) formalizaron la noción de intención comunicativa: un agente no solo transmite información, sino que actúa sobre el estado mental del receptor. Esta distinción —entre intercambiar datos y realizar *actos de habla*— fue la base sobre la que se construyeron los protocolos de comunicación inter-agente, como los especificados por la Foundation for Intelligent Physical Agents (FIPA) a finales de los noventa. Stone y Veloso (2000) extendieron esta visión hacia el aprendizaje: los agentes no solo pueden coordinarse con protocolos predefinidos, sino aprender estrategias de coordinación mediante interacción repetida.

Estos fundamentos —autonomía, comunicación estructurada, especialización de roles, coordinación emergente— son los que los frameworks modernos de agentes LLM intentan recuperar. Pero el sustrato es completamente distinto: en lugar de agentes con arquitecturas simbólicas y protocolos formales, ahora se trabaja con modelos de lenguaje cuyo comportamiento emerge del preentrenamiento y del prompting. La siguiente sección examina qué significa exactamente que un LLM actúe como agente, y qué propiedades del agente clásico se conservan, cuáles se transforman, y cuáles desaparecen.

### Referencias (sección 2.1)

Bratman, M. E. (1987). *Intention, plans, and practical reason*. Harvard University Press.

Cohen, P. R., & Levesque, H. J. (1990). Intention is choice with commitment. *Artificial Intelligence*, *42*(2–3), 213–261. https://doi.org/10.1016/0004-3702(90)90055-5

Rao, A. S., & Georgeff, M. P. (1995). BDI agents: From theory to practice. En *Proceedings of the First International Conference on Multiagent Systems (ICMAS-95)* (pp. 312–319). AAAI Press.

Stone, P., & Veloso, M. (2000). Multiagent systems: A survey from a machine learning perspective. *Autonomous Robots*, *8*(3), 345–383. https://doi.org/10.1023/A:1008942012299

Wooldridge, M. (2009). *An introduction to multiagent systems* (2nd ed.). Wiley.

Wooldridge, M., & Jennings, N. R. (1995). Intelligent agents: Theory and practice. *The Knowledge Engineering Review*, *10*(2), 115–152. https://doi.org/10.1017/S0269888900008122
