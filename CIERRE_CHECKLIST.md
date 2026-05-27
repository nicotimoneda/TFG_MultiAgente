# Checklist de cierre — sustituciones de cifras finales

**Generado:** 2026-05-27, durante S23. **Para ejecutar cuando**
`progress.json` marque `completed == 1476`.

## 0. Comandos previos (sustituyen las cifras automáticas)

```bash
cd ~/UNI/TFG/TFG_MultiAgente
.venv/bin/python experiments/analyze_results.py     # tablas + figuras + McNemar
.venv/bin/python experiments/adherence_metric.py    # adherencia
```

Esto deja **automáticamente actualizadas** (no requieren acción manual):

- `doc/tables/summary.md`, `summary.tex`, `pairwise_mcnemar.md`,
  `per_benchmark.md`, `problem_difficulty.md`, `findings_narrative.md`
- `doc/tables/adherence.md`, `experiments/results/adherence.json`
- `figures/pass_at_1.png`, `cost_quality_pareto.png`, `latency_box.png`,
  `revision_distribution.png`

Luego, leer las cifras finales reales en `doc/tables/summary.md` y
`pairwise_mcnemar.md` y aplicar las sustituciones de abajo.

## 1. Cifras observadas vs. cifras finales esperadas

Las **dos primeras filas (Baseline, Sequential) ya están cerradas** y no
cambian. La fila **SR_r1** y todas las cifras pareadas que la implican
sí cambian. Lectura del dry-run a 1380 filas (SR_r1 a 396/492):

| Métrica SR_r1 | Documento actual (n=211) | Dry-run actual (n=396) | Final (n=492) |
|---|---:|---:|---:|
| n_runs | 211 | 396 | **492** |
| pass@1 | 67,30 % | 66,16 % | leer de `summary.md` |
| IC 95 % | [60,7 ; 73,5] | [61,4 ; 71,0] | leer de `summary.md` |
| pass@3 | 84,51 % | 83,33 % | leer de `summary.md` |
| Tokens | 13 719 | 14 005 | leer de `summary.md` |
| Latencia (s) | 386,79 | 403,75 | leer de `summary.md` |
| Revisiones | 0,39 | 0,40 | leer de `summary.md` |
| McNemar Baseline vs SR (b/c/p) | 51/4/<0,0001 | 85/21/<0,0001 | leer de `pairwise_mcnemar.md` |
| McNemar Sequential vs SR (b/c/p) | 19/29/0,194 | 40/64/**0,0241** | leer de `pairwise_mcnemar.md` |

**Aviso H2.** Ya con n=396 cruza p<0,05 (0,0241). Con n=492 muy
probablemente seguirá significativa. **Hay que cambiar de "no
concluyente" a "respaldada con tendencia favorable".**

## 2. Ficheros a editar a mano (con line numbers)

### `doc/capitulos/06_experimentos.md`

- **L184** — "mientras el experimento principal todavía está en curso"
  → "tanto durante la corrida como tras su cierre" o equivalente.
- **L254-255** — "resultados parciales disponibles" → "resultados al
  cierre de la corrida".

### `doc/capitulos/07_resultados.md`

- **L110-112** — "1 195 ejecuciones, 80,9 % de la matriz" → "1 476
  ejecuciones, 100 % de la matriz"; "self_reflection_r1 está en curso y
  aporta 211 ejecuciones" → texto que refleje que está cerrado con 492.
- **L117-119** (tabla 7.2) — fila SR (r=1): sustituir n, pass@1, IC,
  pass@3, tokens, latencia, revisiones medias.
- **L142-144** — "67,30 %" → cifra final; recalcular "cae 12,8 puntos".
- **L222** — fila tabla SR_r1 (n=211 → 492; revisiones).
- **L232** — "sobre 211 ejecuciones" → "sobre 492 ejecuciones".
- **L287-288** (tabla 7.5 McNemar) — recalcular b, c, p para Baseline
  vs SR (r=1) y **Sequential vs SR (r=1)**; con datos finales esta
  última fila puede cruzar la significancia.
- **L305-310 (§7.6.2 H2)** — reescribir veredicto:
  - Si p<0,05 al cierre: "H2 queda **respaldada**: SR_r1 mejora pass@1
    sobre sequential en X puntos (McNemar p=Y)".
  - Si sigue p≥0,05: mantener no concluyente con n=492.

### `doc/capitulos/08_conclusiones.md`

- **L17** — "algunos números del capítulo 7 son parciales por
  construcción" → reformular o eliminar (al cierre ya no son parciales).
- **L31** — "Baseline y sequential completos; SR_r1 al 43 % (211/492) y
  avanzando" → "Las tres configuraciones completas (492 runs c/u)".
- **L51-53** — "80,9 % de la matriz principal (1 195 de 1 476
  ejecuciones) permite ya emitir conclusiones firmes para dos de las
  tres hipótesis" → "100 % de la matriz (1 476 ejecuciones)"; ajustar
  "dos de las tres hipótesis" según cierre H2.
- **L65-74 (Sobre H2)** — bloque entero. Cambiar "67,30 %" por cifra
  final; "no concluyente" → "respaldada" si p<0,05; reformular
  "con los 211 pares disponibles al cierre" → "con los 492 pares".
- **L195** — "Entregado parcialmente" (SR) sigue siendo correcto porque
  r2/r3 quedan fuera, **no tocar**.

### `doc/capitulos/00_resumen.md`

- **L34-36** (ES) — sustituir "67,30 %" por cifra final.
- **L102-104** (EN, abstract) — sustituir "67.30 %" por cifra final
  (punto decimal en inglés).
- Verificar si el bloque "Resultado principal" menciona el estado de H2;
  actualizar si procede.

### `doc/entregas/ResumenEjecutivo_EntregaFinal.md`

- **L28-30** (tabla) — fila SR (r=1): n=211 → 492; 67,30 % → final;
  IC, tokens, latencia.
- **L32** — narrativa "cae 12,8 puntos" → recalcular con cifra final;
  revisar el factor "40×/77× más caro" según latencia y tokens finales;
  adherencia final (lectura de `adherence.md`).

### `README.md`

- **L23-25** (tabla) — fila SR (r=1) idéntica a la del resumen ejecutivo;
  eliminar el asterisco "SR_r1 en curso".
- **L27** — eliminar "\* SR_r1 en curso al cierre del documento"; añadir
  la línea con los tres McNemar pareados finales.

### `doc/capitulos/anexo_E_etica.md`

- **L131-133** (tabla CO₂e) — fila SR (r=1) con n=492 y latencia
  final. Recalcular kWh y kg CO₂eq:
  - Energía (kWh) = n × latencia(s) × P_med(W) / 3 600 000.
  - L133 actual: 211 × 386,79 = 81 612 s ≈ 22,67 Wh × algo.
- **L148-150** — "≈ 0,30 kg CO₂eq" total: recalcular con la suma de
  las tres configs cerradas. Con factor 0,16 kg CO₂eq/kWh.

## 3. Verificación final

```bash
bash scripts/build_memoria.sh   # regenera .docx
```

Comprobar visualmente en `build/2526_TFG_GCIA_NP147254_Memoria.docx`:

- Que ninguna mención a "211", "1 195", "1380", "80,9 %", "n=396",
  "en curso" o "parcial" quede sin actualizar.
- Que el verdict de H2 sea coherente entre cap 7 §7.6.2, cap 8 §8.3
  H2, resumen ES, abstract EN, ResumenEjecutivo y README.
- Que la frontera de Pareto del cap 7 siga reflejando que la ocupa el
  baseline.

## 4. Commit recomendado al cierre

```
git add doc/ figures/ README.md experiments/results/  # CSVs incluidos
git commit -m "S24: cierre experimental (1476 runs) + actualización cifras finales"
```

(El handoff prohíbe commitear CSVs **mientras corre**; al cerrar, sí
deben commitearse junto al resto de artefactos.)
