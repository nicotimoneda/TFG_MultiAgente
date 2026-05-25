# Anexo D: Glosario de acrónimos y términos técnicos

Este anexo recoge los acrónimos y términos técnicos que aparecen
reiteradamente en la memoria, agrupados por área temática.

## D.1. Sistemas multi-agente

| Sigla | Expansión | Definición breve |
|---|---|---|
| **MAS** | Multi-Agent System | Sistema computacional compuesto por agentes autónomos que interactúan en un entorno compartido. |
| **BDI** | Beliefs, Desires, Intentions | Modelo de arquitectura de agentes en el que el comportamiento emerge de creencias sobre el mundo, deseos y compromisos activos (Rao y Georgeff, 1995). |
| **SOP** | Standard Operating Procedure | Protocolo de actuación estructurado asignado a un rol; término popularizado por MetaGPT (Hong et al., 2024). |
| **FIPA** | Foundation for Intelligent Physical Agents | Organización que estandarizó protocolos de comunicación inter-agente a finales de los noventa. |
| **SR** | Self-Reflection | Patrón en el que un agente revisor evalúa la salida y la devuelve al productor para refinarla en un bucle acotado por número máximo de iteraciones. |
| **PM** | Product Manager | Rol del pipeline secuencial encargado de redactar el PRD a partir del enunciado. |
| **LangGraph** | — | Framework de orquestación de agentes basado en grafos de estado tipados, mantenido por LangChain Inc. (LangChain Inc., 2024). |

## D.2. Modelos de lenguaje

| Sigla | Expansión | Definición breve |
|---|---|---|
| **LLM** | Large Language Model | Modelo de lenguaje preentrenado con un número grande de parámetros (típicamente miles de millones). |
| **RAG** | Retrieval-Augmented Generation | Patrón que combina un LLM con una recuperación de documentos relevantes para fundamentar la generación. |
| **CoT** | Chain-of-Thought | Técnica de prompting que pide al modelo razonar paso a paso (Wei et al., 2022). |
| **ReAct** | Reasoning + Acting | Patrón que intercala razonamiento y llamadas a herramientas (Yao et al., 2023). |
| **TPM** | Tokens Per Minute | Límite de tasa habitual en proveedores de inferencia comercial. |
| **RPM** | Requests Per Minute | Límite de tasa por número de solicitudes. |

## D.3. Generación y evaluación de código

| Sigla | Expansión | Definición breve |
|---|---|---|
| **pass@k** | Pass at k | Probabilidad de que al menos una de `k` muestras independientes generadas por el modelo supere todos los tests del problema. Estimador insesgado de Chen et al. (2021). |
| **HumanEval** | Human-curated Evaluation | Benchmark de 164 problemas de generación de funciones Python con tests unitarios (Chen et al., 2021). |
| **MBPP** | Mostly Basic Python Problems | Benchmark complementario de problemas introductorios y de complejidad media (Austin et al., 2021). |
| **APPS** | Automated Programming Progress Standard | Benchmark de >10 000 problemas de competitive programming (Hendrycks et al., 2021). |
| **SWE-bench** | Software Engineering Benchmark | Benchmark de tareas reales de ingeniería del software a partir de issues de GitHub (Jiménez et al., 2024). |
| **ClassEval** | Class-level Evaluation | Benchmark de generación de código a nivel de clase con dependencias entre métodos (Du et al., 2023). |
| **PRD** | Product Requirements Document | Artefacto de especificación de requisitos producido por el agente Product Manager. |
| **QA** | Quality Assurance | Rol responsable de la verificación funcional del código mediante pruebas. |
| **CI** | Confidence Interval | Intervalo de confianza estadístico, reportado al 95 % en este trabajo. |
| **IC** | Intervalo de Confianza | Equivalente en castellano de CI. |

## D.4. Stack tecnológico

| Sigla | Expansión | Definición breve |
|---|---|---|
| **API** | Application Programming Interface | Interfaz programática de un servicio. |
| **CSV** | Comma-Separated Values | Formato de fichero de los resultados experimentales. |
| **JSON** | JavaScript Object Notation | Formato de serialización de datos estructurados usado para prompts, logs y trazas. |
| **TypedDict** | Typed Dictionary | Tipo de Python para diccionarios con campos tipados; usado para el `AgentState`. |
| **CPU** | Central Processing Unit | Procesador del equipo de cómputo. |
| **GPU** | Graphics Processing Unit | Unidad de procesamiento gráfico, utilizada para acelerar la inferencia del LLM. |
| **GPL** | GNU General Public License | Familia de licencias de software libre. |
| **MIT** | Massachusetts Institute of Technology | Licencia de software permisiva, utilizada en este proyecto. |

## D.5. Métodos estadísticos

| Sigla | Expansión | Definición breve |
|---|---|---|
| **McNemar** | Test de McNemar | Prueba estadística para clasificadores pareados sobre los mismos sujetos. |
| **Bootstrap** | Bootstrap percentil | Método de re-muestreo para estimar intervalos de confianza sin asumir distribución. |
| **TFG** | Trabajo Fin de Grado | Documento académico final de la titulación. |
| **UAX** | Universidad Alfonso X el Sabio | Institución académica del autor. |

## D.6. Configuraciones del sistema

| Etiqueta | Descripción |
|---|---|
| `baseline` | Configuración 1, monolítica: un único LLM produce el código directamente. |
| `sequential` | Configuración 2, pipeline de cinco roles sin ciclos: PM → Arquitecto → Developer → QA → Reviewer. |
| `self_reflection_rN` | Configuración 3, pipeline secuencial con bucle Reviewer → Developer, hasta N iteraciones. N ∈ {1, 2, 3}. |
| `ablation_no_pm` | Variante que omite el Product Manager. |
| `ablation_no_architect` | Variante que omite el Arquitecto. |
| `ablation_no_reviewer` | Variante que omite el Code Reviewer. |

## D.7. Términos del estado compartido

| Campo del `AgentState` | Productor | Descripción |
|---|---|---|
| `problem_statement` | Benchmark | Enunciado en lenguaje natural del problema. |
| `function_signature` | Benchmark | Identificador (función o clase) que se debe implementar. |
| `test_cases` | Benchmark | Aserciones unitarias asociadas al problema. |
| `prd` | Product Manager | Documento de requisitos estructurado. |
| `design_doc` | Arquitecto | Documento de diseño técnico. |
| `code_artifact` | Developer | Código fuente Python generado. |
| `test_results` | QA Tester | Mapa booleano por test que indica éxito o fallo. |
| `review_comments` | Code Reviewer | Veredicto APPROVE/REQUEST_CHANGES y comentarios cualitativos. |
| `revision_count` | Grafo SR | Número de iteraciones de revisión consumidas. |
| `tokens_input` | Todos | Tokens de entrada acumulados durante la ejecución. |
| `tokens_output` | Todos | Tokens de salida acumulados. |
| `latency_seconds` | Runner | Tiempo de wall-clock de la ejecución. |

## D.8. Marco normativo y ético

| Sigla | Expansión | Definición breve |
|---|---|---|
| **IA** | Inteligencia Artificial | Término genérico para los sistemas tratados en este TFG. Equivalente al AI anglosajón. |
| **RGPD** | Reglamento General de Protección de Datos | Reglamento (UE) 2016/679 que regula el tratamiento de datos personales en la Unión Europea. |
| **GPAI** | General-Purpose AI | Categoría de modelos de propósito general definida por el Reglamento de IA de la UE (AI Act). |
| **UE** | Unión Europea | Marco jurisdiccional aplicable al desarrollo y despliegue del sistema. |
| **UNESCO** | Organización de las Naciones Unidas para la Educación, la Ciencia y la Cultura | Organismo cuyas recomendaciones éticas sobre IA se citan en el análisis ético. |
