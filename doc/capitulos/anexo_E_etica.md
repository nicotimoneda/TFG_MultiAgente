# Anexo E: Aspectos éticos, legales y de sostenibilidad

Este anexo recoge las consideraciones éticas, legales y de
sostenibilidad relativas al trabajo. La extensión es deliberadamente
breve y operativa: identifica los aspectos relevantes, documenta las
decisiones tomadas y señala los marcos normativos aplicables. No
sustituye un dictamen jurídico ni una revisión ética formal por
parte de un comité institucional.

## E.1. Naturaleza de los datos utilizados

El estudio no procesa datos personales ni datos sensibles. Los dos
benchmarks empleados —HumanEval y MBPP— están compuestos por
problemas sintéticos de programación, redactados por sus autores
para evaluar modelos. No contienen información identificable, ni
metadatos personales, ni texto extraído de comunicaciones privadas.
HumanEval se publica bajo licencia MIT por OpenAI; MBPP está
publicado por Google Research bajo Apache 2.0. La inclusión de
ambos benchmarks en este trabajo respeta los términos de sus
respectivas licencias.

En consecuencia, este TFG queda fuera del alcance del Reglamento
General de Protección de Datos (RGPD, Reglamento UE 2016/679) por
ausencia de tratamiento de datos personales. No se ha requerido por
tanto la intervención del Delegado de Protección de Datos de la
UAX.

## E.2. Modelos utilizados y sus licencias

El modelo principal utilizado en la corrida experimental es
**Qwen 2.5 Coder 7B Instruct** (Hui et al., 2024), publicado por
Alibaba Cloud bajo licencia Apache 2.0. La cuantización Q4_K_M y la
distribución a través de Ollama mantienen las condiciones de la
licencia original.

El proyecto soporta como backend alternativo **Cerebras Inference**,
que ofrece acceso gratuito a varios modelos open-weight (entre
ellos las familias Qwen y Llama de Meta) bajo sus términos de
servicio públicos. La verificación cruzada que en su momento se
planteó con Qwen-3 235B o Llama 3.1 se contempló dentro de los
límites del tier gratuito, sin recurrir a ningún plan de pago.

Todos los modelos utilizados son **open-weight**: los pesos están
publicados y son auditables. Esta elección se ajusta a las
recomendaciones de transparencia para investigación académica
recogidas en el Código de Buenas Prácticas Científicas de la UAX y
en la Recomendación de la UNESCO sobre la Ética de la IA (2021).

## E.3. Generación automática de código y responsabilidad

El sistema produce código fuente Python. Aunque el alcance de este
TFG se limita a problemas algorítmicos de benchmark, la tecnología
subyacente es la misma que se está integrando en herramientas de
desarrollo profesionales. Procede señalar tres consideraciones:

**Autoría y trazabilidad.** Todo código generado por el sistema en
los experimentos queda registrado en los CSV de resultados junto a
los artefactos intermedios (PRD, design doc, review). En un
hipotético despliegue en un entorno profesional, esa trazabilidad
permite auditar la procedencia de cada fragmento de código y
distinguirla de los aportes humanos.

**Verificación funcional como salvaguarda.** El sandbox de
ejecución y el agente QA Tester son piezas centrales del diseño,
no añadidos opcionales. Su función es precisamente garantizar que
ningún código generado se acepta sin verificación contra pruebas
objetivas, lo que mitiga la categoría más común de error en la
generación automática: la apariencia de corrección sin verificación
real (Chen et al., 2021).

**Limitaciones reconocidas.** El estudio se limita a generación de
funciones aisladas. No aborda generación de código en contextos de
seguridad crítica, ni código que interactúe con datos personales
en tiempo de ejecución, ni código que se despliegue sin revisión
humana. Estas son extensiones reconocidas como fuera de alcance en
el capítulo 8.

## E.4. Marco legal: Reglamento Europeo de IA

El Reglamento Europeo de Inteligencia Artificial (Reglamento (UE)
2024/1689, conocido como **AI Act**), aprobado el 13 de junio de
2024 y con entrada en vigor escalonada hasta 2026-2027, establece
un marco regulatorio para los sistemas de IA en función de su nivel
de riesgo. El sistema multi-agente desarrollado en este TFG se
clasifica como **sistema de IA de propósito general aplicado a la
generación de código**, categoría que el AI Act trata bajo los
artículos relativos a modelos fundacionales (general-purpose AI,
GPAI).

Las obligaciones aplicables en este nivel son:

- **Transparencia sobre el sistema.** El presente documento, junto
  al repositorio público y los prompts reproducidos en el Anexo A,
  cumplen el requisito de documentación técnica que el AI Act
  exige a los desarrolladores de sistemas basados en GPAI.
- **Trazabilidad de las salidas generadas.** Los CSV de
  experimentación registran qué modelo y qué configuración produjo
  cada artefacto, lo que satisface el principio de trazabilidad
  recogido en el artículo 53 del AI Act.
- **Evaluación de riesgo de uso.** El sistema se limita a
  generación de código en un entorno controlado de evaluación
  académica; no procesa interacciones con usuarios finales ni
  toma decisiones automatizadas con efecto jurídico, por lo que no
  se sitúa en las categorías de alto riesgo definidas en el Anexo
  III del AI Act.

No se utilizan en este TFG sistemas clasificados como de riesgo
inaceptable (artículo 5) ni de alto riesgo (Anexo III), de manera
que las obligaciones materiales del AI Act se limitan a las de
transparencia y documentación recogidas más arriba.

## E.5. Sostenibilidad y huella de cómputo

La corrida experimental se ejecuta sobre un único MacBook Air con
chip Apple M2, un equipo de consumo energético reducido (potencia
máxima nominal de 30 W, típicamente <20 W durante inferencia con
modelos cuantizados). La duración estimada de la corrida principal
es de aproximadamente 9-10 días de cómputo desatendido.

Una estimación conservadora del consumo total es:

- Potencia media bajo carga continua: ≈ 18 W
- Duración: ≈ 220 horas
- Energía total: ≈ 4 kWh
- Huella de carbono equivalente (mix eléctrico español 2024,
  ~0.16 kgCO₂/kWh): ≈ 0.6 kg CO₂eq

La elección deliberada de un modelo cuantizado de 7 B parámetros y
de hardware local de bajo consumo —frente a alternativas como
modelos de frontera servidos en datacenters comerciales— sitúa el
coste energético del experimento en el orden de magnitud de la
carga de un par de smartphones durante un año. Esta consideración
forma parte del razonamiento de scope documentado en el capítulo 3
y en el Anexo B.

## E.6. Reproducibilidad como obligación ética

La publicación del código fuente bajo licencia MIT, la inclusión
verbatim de los prompts en el Anexo A, la documentación de cada
decisión técnica en el Anexo B, y el listado de comandos de
reproducción en el Anexo C responden no sólo a un criterio
metodológico sino a una obligación ética: cualquier afirmación
empírica de esta memoria es **falsable por terceros con acceso al
equipo equivalente y al modelo público**. Esto es coherente con
los principios de la Recomendación de la UNESCO sobre la Ética de
la IA (2021), que sitúa la verificabilidad como uno de los pilares
del desarrollo responsable de sistemas de IA.

## E.7. Conflictos de interés

El autor declara no tener conflictos de interés económicos, ni
relaciones contractuales con ninguno de los proveedores de modelos
o infraestructura mencionados en este trabajo (Alibaba Cloud,
Meta, Cerebras, Ollama, LangChain Inc., HuggingFace). Todas las
referencias a estos proveedores se realizan a título exclusivamente
técnico.
