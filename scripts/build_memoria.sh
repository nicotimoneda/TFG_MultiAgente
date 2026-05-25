#!/usr/bin/env bash
#
# build_memoria.sh — assemble the TFG memoria into a single .docx + .pdf
#
# Concatenates chapters 1..8, anexos A..C and the global bibliography in
# canonical order, then invokes pandoc to produce both .docx and .pdf
# outputs. Idempotent: previous outputs are overwritten in place.
#
# Requirements:
#   - pandoc (brew install pandoc)
#   - For .pdf: a working LaTeX distribution (brew install --cask basictex,
#     then sudo tlmgr install collection-fontsrecommended).
#
# Output:
#   build/2526_TFG_GCIA_NP147254_Memoria.docx
#   build/2526_TFG_GCIA_NP147254_Memoria.pdf
#
# The .docx is the entrega-final deliverable; the .pdf is a quick-read
# rendering for review.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD="build"
mkdir -p "$BUILD"

OUT_BASE="2526_TFG_GCIA_NP147254_Memoria"
DOCX="$BUILD/${OUT_BASE}.docx"
PDF="$BUILD/${OUT_BASE}.pdf"
COMBINED_MD="$BUILD/${OUT_BASE}_full.md"

# Order of inclusion. The bibliography goes last by tutor's instruction:
# no per-chapter references, only the global list at the end.
CHAPTERS=(
    "doc/capitulos/00_resumen.md"
    "doc/capitulos/declaracion_ia.md"
    "doc/capitulos/01_introduccion.md"
    "doc/capitulos/02_estado_del_arte.md"
    "doc/capitulos/03_objetivos.md"
    "doc/capitulos/04_metodologia.md"
    "doc/capitulos/05_desarrollo.md"
    "doc/capitulos/06_experimentos.md"
    "doc/capitulos/07_resultados.md"
    "doc/capitulos/08_conclusiones.md"
    "doc/capitulos/anexo_A_prompts.md"
    "doc/capitulos/anexo_B_decisiones.md"
    "doc/capitulos/anexo_C_reproducir.md"
    "doc/capitulos/anexo_D_glosario.md"
    "doc/capitulos/anexo_E_etica.md"
    "doc/capitulos/anexo_F_agradecimientos.md"
    "doc/referencias/bibliografia.md"
)

echo "[build] concatenating ${#CHAPTERS[@]} sources → $COMBINED_MD"
{
    cat <<'EOF'
---
title: "Orquestación de Equipos de Agentes LLM con Roles Especializados para Resolución Colaborativa de Tareas Complejas de Ingeniería de Software"
author: "Nicolás Timoneda"
date: "2026"
lang: es
toc: true
toc-depth: 2
lof: true
lot: true
numbersections: true
geometry: margin=2.5cm
---

EOF
    for f in "${CHAPTERS[@]}"; do
        if [ ! -f "$f" ]; then
            echo "[build] WARNING: missing $f, skipping" >&2
            continue
        fi
        echo "<!-- source: $f -->"
        echo
        cat "$f"
        echo
        echo
    done
} > "$COMBINED_MD"

echo "[build] pandoc → $DOCX"
pandoc "$COMBINED_MD" \
    --from=gfm+yaml_metadata_block \
    --to=docx \
    --output="$DOCX" \
    --standalone \
    --resource-path=".:figures:doc"

if pandoc --list-output-formats | grep -q "^pdf$"; then
    echo "[build] pandoc → $PDF"
    pandoc "$COMBINED_MD" \
        --from=gfm+yaml_metadata_block \
        --to=pdf \
        --output="$PDF" \
        --pdf-engine=xelatex \
        --resource-path=".:figures:doc" \
        || echo "[build] WARNING: PDF generation failed — install a LaTeX engine to enable"
else
    echo "[build] PDF not generated (no LaTeX engine detected)"
fi

echo "[build] done."
echo "  docx: $DOCX"
[ -f "$PDF" ] && echo "  pdf:  $PDF"
