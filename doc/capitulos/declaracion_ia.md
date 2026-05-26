# Declaración sobre el uso de herramientas de inteligencia artificial

En cumplimiento de las directrices institucionales de la Universidad
Alfonso X el Sabio sobre el uso responsable de herramientas de
inteligencia artificial en trabajos académicos, y en línea con la
Recomendación de la UNESCO sobre la Ética de la IA (2021) y con las
obligaciones de transparencia del Reglamento Europeo de IA (Reglamento
(UE) 2024/1689), el autor declara lo siguiente:

## Herramientas utilizadas

Durante la elaboración de esta memoria se ha empleado el asistente
conversacional **Claude** (Anthropic) como herramienta de apoyo a la
redacción, revisión estilística y depuración del código fuente del
sistema. La herramienta se ha utilizado en los siguientes contextos:

- **Redacción asistida.** Borradores iniciales de secciones técnicas,
  reescritura de fragmentos para mejorar la claridad expositiva y
  revisión de coherencia entre capítulos.
- **Revisión y refactorización de código.** Sugerencias de mejora
  sobre el código del sistema multi-agente, depuración de errores y
  generación de pruebas unitarias.
- **Documentación técnica.** Apoyo en la redacción del README,
  comentarios docstring y comandos de reproducción.

## Carácter del uso

El uso de la herramienta ha sido **asistencial, nunca sustitutivo**.
Todas las decisiones de diseño del sistema, las hipótesis del estudio,
la metodología experimental, la interpretación de los resultados y la
estructura argumental de la memoria son responsabilidad intelectual
del autor. El asistente ha intervenido en la forma del texto y en la
exploración de implementaciones, pero no en la formulación de las
contribuciones originales del trabajo.

Los siguientes elementos del trabajo han sido producidos
**íntegramente sin asistencia de IA**:

- La elección del tema y la formulación de la pregunta de
  investigación, recogidas en la propuesta original aprobada por el
  Jefe de Estudios.
- Las decisiones de scope sobre qué configuraciones evaluar, qué
  benchmarks utilizar y qué hipótesis contrastar.
- La interpretación cualitativa del estudio piloto exploratorio
  (capítulo 7, sección 7.2).
- El veredicto interpretativo de los resultados (rechazo de H1 y
  H3, no concluyente para H2) y la identificación de las tres
  causas plausibles del hallazgo negativo (propagación de errores,
  sobrecarga del prompt en modelos pequeños, inadecuación de
  HumanEval para evaluar pipelines multi-agente).

Los siguientes elementos han sido **asistidos por IA pero revisados,
editados y validados por el autor**:

- Borradores de los capítulos 1 a 8 y los anexos A a F, incluyendo
  la redacción final de la discusión por hipótesis del capítulo 8 y
  del análisis cualitativo del capítulo 7.7 a partir de los puntos
  interpretativos definidos por el autor.
- El código fuente bajo `src/`, `experiments/`, `tests/` y
  `scripts/`.
- La generación de las figuras de los grafos y del diagrama de
  arquitectura mediante renderizado de descripciones Mermaid.

## Trazabilidad

Las conversaciones de asistencia con la herramienta han quedado
registradas localmente durante el desarrollo del proyecto, en
cumplimiento del principio de auditabilidad de la Recomendación
UNESCO 2021. El autor mantiene la responsabilidad última sobre cada
afirmación contenida en este documento, sobre el correcto
funcionamiento del código entregado, y sobre la defensa oral del
trabajo ante el tribunal evaluador.

## Capacidad de defensa

El autor declara expresamente su capacidad de **explicar, justificar
y defender** cada decisión técnica y conceptual recogida en esta
memoria sin asistencia externa, en la defensa oral del trabajo ante
el tribunal. La asistencia de IA durante el desarrollo no implica
delegación de la competencia evaluable.

---

*Madrid, junio de 2026*

*Fdo: Nicolás Timoneda*
