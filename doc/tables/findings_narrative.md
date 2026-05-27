# Hallazgos preliminares (borrador automático)

El análisis cubre 3 configuración(es) con datos (Baseline, Sequential, SR (r=1)), un total de 1476 ejecuciones válidas y los benchmarks HE.

La configuración con mayor pass@1 es Baseline, que alcanza 80.08% (IC 95% [76.42%, 83.54%]) sobre 492 ejecuciones. En el extremo opuesto, Sequential obtiene 58.33% (IC 95% [53.86%, 62.60%]), lo que delimita el rango observado entre configuraciones.

En el plano coste-calidad, la frontera de Pareto está formada por Baseline (283 tokens, 80.08%). Estas configuraciones no son dominadas por ninguna otra: cualquier alternativa con menos tokens medios obtiene un pass@1 inferior.

El contraste pareado de McNemar con menor p-valor enfrenta a Baseline y Sequential sobre 492 pares emparejados (b=128, c=21, p=0.0000, método chi2); Baseline supera a Sequential, lo que constituye una diferencia estadísticamente significativa al 5%.

En las configuraciones de self-reflection, el número medio de revisiones es 0.41 (mediana 0); el 58.74% de las ejecuciones aprueba sin necesidad de ninguna revisión (r=0), lo que sugiere que el revisor sólo interviene en una fracción acotada de los problemas.

La adherencia estructural no es uniforme entre configuraciones: alguna combinación presenta fallos o avisos en el contrato de salida, según refleja la tabla de adherencia generada.

Esta redacción ha sido generada automáticamente a partir de datos parciales del experimento y debe revisarse manualmente antes de la entrega final: las cifras dependen del estado actual de los CSV y pueden variar a medida que se completen las ejecuciones pendientes.
