# Outline de la defensa del TFG

**Duración estimada:** 15 minutos de exposición + 10-15 minutos de
preguntas del tribunal.

**Audiencia:** tribunal académico de la UAX (Computación e IA). Es razonable
asumir que conocen LLMs como concepto pero no la literatura específica de
sistemas multi-agente.

## Estructura sugerida (12-14 slides)

### Slide 1 — Portada (15 s)

- Título completo del TFG, autor, tutor, titulación, fecha.

### Slide 2 — Pregunta de investigación (45 s)

- Una sola frase: ¿la especialización por roles mejora la generación
  automática de código frente a un LLM monolítico, y a qué coste?
- Dos viñetas que sitúen por qué interesa: división del trabajo es
  natural en ingeniería de software humana; los LLM monolíticos
  alucinan sin verificación interna.
- **Foto/diagrama**: figura del Anexo C "arquitectura_sistema.png".

### Slide 3 — Las tres configuraciones (90 s)

- Diagrama compacto de las 3 topologías una al lado de otra:
  baseline (1 nodo), sequential (5 nodos), self_reflection (5 + bucle).
- Una viñeta que justifique el orden incremental: aislar primero el
  efecto de los roles, después el del ciclo de revisión.
- Apoyarse en `figures/graph_baseline.png`, `graph_sequential.png`,
  `graph_self_reflection_r1.png`.

### Slide 4 — El estado compartido (60 s)

- Snippet recortado del `AgentState` (caben los 13 campos en un slide
  si se reduce el tamaño de fuente).
- Una viñeta: cada campo tiene un único productor, lo que evita
  condiciones de carrera lógicas.
- Una viñeta sobre la comunicación estructurada vs. conversacional:
  esta es la afirmación que se mide con la métrica de adherencia.

### Slide 5 — Metodología (60 s)

- Benchmark: HumanEval (164 problemas, evalplus extended).
- Modelo: Qwen 2.5 Coder 7B Instruct Q4_K_M sobre Ollama local en M2.
- 3 semillas por problema y por configuración para varianza.
- Métricas: pass@1 con IC 95% bootstrap, pass@3 (estimador Chen 2021),
  test pareado de McNemar, adherencia estructural.

### Slide 6 — Las ablaciones (45 s)

- Tres variantes que eliminan exactamente un rol del pipeline.
- Justificación: cuantificar la **contribución marginal de cada rol**,
  no sólo la del sistema completo.
- Posicionar como innovación del trabajo respecto a la literatura, que
  típicamente compara el sistema completo contra un baseline.

### Slide 7 — Resultados: pass@1 por configuración (90 s)

- `figures/pass_at_1.png` ocupando ¾ del slide.
- Mensaje principal en una frase grande: "X% sequential vs. Y% baseline,
  diferencia significativa al 95%" (o "no significativa", según el
  resultado real al final de la corrida).
- Bullet con el p-valor de McNemar.

### Slide 8 — Resultados: coste-calidad (90 s)

- `figures/cost_quality_pareto.png` ocupando ¾ del slide.
- Mensaje: a partir de un determinado nivel de revisiones la mejora
  marginal de pass@1 deja de compensar el coste en tokens.
- Identificar visualmente la configuración en la frontera Pareto.

### Slide 9 — Ablaciones (90 s)

- Tabla compacta: pass@1 de sequential vs cada ablación, con diferencia
  porcentual.
- Mensaje: cuál es el rol crítico (el que más reduce el pass@1 al
  retirarlo) y cuál el menos crítico.
- Si los resultados lo permiten, una hipótesis: "el QA Tester aporta
  el mayor incremento por su papel determinista de verificación".

### Slide 10 — Adherencia y telemetría (60 s)

- `doc/tables/adherence.md` reducido a 4 filas (baseline, sequential,
  SR_r3, ablation_no_reviewer).
- Mensaje: el protocolo estructurado mantiene 100% de adherencia incluso
  con modelos pequeños, lo que justifica la elección frente a
  conversación libre.

### Slide 11 — Conclusiones por hipótesis (90 s)

- H1 — Especialización: aceptada / rechazada / parcial con datos.
- H2 — Auto-revisión: aceptada / rechazada / parcial con datos.
- H3 — Trade-off: confirmado con la frontera Pareto del slide 8.
- Una frase sintetizando la pregunta de investigación con la respuesta.

### Slide 12 — Limitaciones y trabajo futuro (60 s)

- Modelo de 7 B sólo; conclusiones no extrapolables automáticamente.
- HumanEval mide funciones aisladas, no proyectos reales.
- SWE-bench fuera de alcance por presupuesto Docker.
- Líneas futuras: ClassEval real (scaffolding listo), Dynamic Task
  Decomposition, comparativa heterogénea por rol.

### Slide 13 — Aportaciones operativas (45 s)

- Banco experimental resumible y atómico.
- Pipeline de análisis con figuras, tablas y test pareado.
- Tres anexos reproducibles (prompts, decisiones, comandos).
- Repositorio público en GitHub.

### Slide 14 — Cierre y gracias (15 s)

- Una frase ejecutiva del veredicto del trabajo.
- Datos de contacto.

## Preguntas previsibles del tribunal (preparación)

1. **¿Por qué Qwen 7B y no GPT-4 o Claude?** Respuesta: restricción de
   hardware (16 GB de memoria unificada), reproducibilidad sin coste
   marginal por inferencia, y la pregunta de investigación es sobre
   la **diferencia entre arquitecturas con un modelo fijado**, no
   sobre el modelo en sí. El factory de clientes permite repetir con
   un modelo mayor a través de Cerebras como verificación cruzada.

2. **¿Por qué no SWE-bench?** Respuesta: requiere reproducir entornos
   Docker por instancia con dependencias variables, infraestructura
   fuera del alcance del TFG. El loader y el stub están preparados
   para una extensión futura.

3. **¿Cómo escalaría el pipeline secuencial a proyectos multi-fichero?**
   Respuesta: el `AgentState` actual modela un único `code_artifact`;
   la extensión natural exige modificarlo a una lista tipada de
   artefactos y adaptar los prompts del Developer y del Reviewer
   para mantener coherencia entre módulos. ClassEval ya es un paso
   intermedio en esa dirección.

4. **¿Qué hace al diseño multi-agente más robusto que prompting
   chain-of-thought sobre un LLM único?** Respuesta: el estado tipado
   y la separación de responsabilidades garantizan que cada artefacto
   tiene un productor único y un formato verificable, lo que CoT
   sobre un prompt monolítico no permite. Además, el ciclo de
   self-reflection es condicional sobre el resultado de las pruebas,
   no sobre auto-evaluación del modelo —es decir, está fundamentado
   en evidencia externa, no en la propia confianza del LLM.

5. **¿Qué validez tiene comparar configuraciones con muestras de
   tamaño diferente?** Respuesta: el bootstrap percentil con 2 000
   remuestras y el test pareado de McNemar son apropiados para
   muestras de tamaño variable. Los intervalos de confianza
   reportados ya capturan la incertidumbre asociada a la cardinalidad
   disponible al cierre de la corrida.

6. **¿Por qué no comparativa heterogénea de LLMs por rol?** Respuesta:
   espacio combinatorio de configuraciones inmanejable con presupuesto
   de cómputo limitado; controlar la variable "modelo" facilita la
   atribución causal de las diferencias observadas al diseño del
   pipeline, que es el objeto de estudio principal.

7. **¿El veredicto del Reviewer no es entonces redundante respecto al
   QA Tester?** Respuesta: el veredicto APPROVE/REQUEST_CHANGES sí
   se deriva determinísticamente del QA, pero los **comentarios**
   que el Reviewer emite alimentan el Developer reflexivo en el
   ciclo de self-reflection. La ablación `no_reviewer` cuantifica
   exactamente cuánto aporta esa señal cualitativa al pass@1 final.

## Material de apoyo

Si el tribunal pide demo en vivo:

```bash
# 1. Quick check (10 problemas, 5 configs, ~15 min)
LLM_BACKEND=ollama python experiments/quick_check.py

# 2. Single-problem demo en cada configuración
LLM_BACKEND=ollama python - <<'EOF'
from src.evaluation.humaneval_loader import get_problem
from src.graph.baseline_graph import run_baseline
from src.graph.sequential_graph import run_sequential
p = get_problem("HumanEval/38")  # decode/encode pair, mostrado en el piloto
b = run_baseline(p, "qwen2.5-coder:7b-instruct-q4_K_M")
s = run_sequential(p, "qwen2.5-coder:7b-instruct-q4_K_M")
print("baseline tokens:", b["tokens_input"] + b["tokens_output"])
print("sequential tokens:", s["tokens_input"] + s["tokens_output"])
EOF
```

El problema HumanEval/38 (sección 7.2.1) es bueno para la demo porque
en el piloto monolítico el LLM produjo un `decode` incorrecto; con la
configuración secuencial el QA Tester detecta el fallo y el Reviewer
lo señala, demostrando el valor del pipeline en un caso concreto.
