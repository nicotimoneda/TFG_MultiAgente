# Hallazgos preliminares (borrador automático)

El análisis cubre 2 configuración(es) con datos (Baseline, Sequential), un total de 510 ejecuciones válidas y los benchmarks HE.

La configuración con mayor pass@1 es Baseline, que alcanza 80.08% (IC 95% [76.42%, 83.54%]) sobre 492 ejecuciones. En el extremo opuesto, Sequential obtiene 72.22% (IC 95% [50.00%, 94.44%]), lo que delimita el rango observado entre configuraciones.

En el plano coste-calidad, la frontera de Pareto está formada por Baseline (283 tokens, 80.08%). Estas configuraciones no son dominadas por ninguna otra: cualquier alternativa con menos tokens medios obtiene un pass@1 inferior.

El contraste pareado de McNemar con menor p-valor enfrenta a Baseline y Sequential sobre 18 pares emparejados (b=5, c=0, p=0.0625, método exact); Baseline supera a Sequential, lo que constituye una diferencia no significativa al 5%.

El análisis de adherencia estructural muestra una tasa del 100% en todas las configuraciones evaluadas: no se han registrado fallos de formato ni avisos en el contrato de salida JSON, lo que respalda la robustez del pipeline de validación.

Esta redacción ha sido generada automáticamente a partir de datos parciales del experimento y debe revisarse manualmente antes de la entrega final: las cifras dependen del estado actual de los CSV y pueden variar a medida que se completen las ejecuciones pendientes.
