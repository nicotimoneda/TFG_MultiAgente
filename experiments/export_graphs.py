#!/usr/bin/env python3
"""Export LangGraph diagrams (Mermaid + PNG) for each configuration.

Generates static figures for the thesis document (chapter 5):
- figures/graph_baseline.{mmd,png}
- figures/graph_sequential.{mmd,png}
- figures/graph_self_reflection.{mmd,png}

PNG rendering uses the public mermaid.ink service via
`draw_mermaid_png()`; the .mmd source is also saved so the figures
can be re-rendered or edited later without network access.

Usage:
    python experiments/export_graphs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Avoid touching real LLM clients during graph construction.
os.environ.setdefault("LLM_BACKEND", "ollama")

from src.graph.baseline_graph import build_baseline_graph  # noqa: E402
from src.graph.sequential_graph import build_sequential_graph  # noqa: E402
from src.graph.self_reflection_graph import build_self_reflection_graph  # noqa: E402


_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
_OUT = Path("figures")


def _export(name: str, graph) -> None:
    g = graph.get_graph()
    mmd = g.draw_mermaid()
    (_OUT / f"graph_{name}.mmd").write_text(mmd, encoding="utf-8")
    print(f"  ✓ figures/graph_{name}.mmd ({len(mmd)} chars)")
    try:
        png = g.draw_mermaid_png()
        (_OUT / f"graph_{name}.png").write_bytes(png)
        print(f"  ✓ figures/graph_{name}.png ({len(png)//1024} KB)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ PNG render failed for {name}: {exc}")
        print("    (the .mmd source was saved; render manually at mermaid.live)")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print("Exporting graph diagrams...\n")

    print("baseline:")
    _export("baseline", build_baseline_graph(_MODEL))

    print("\nsequential:")
    _export("sequential", build_sequential_graph(_MODEL))

    for r in (1, 2, 3):
        print(f"\nself_reflection (max_revisions={r}):")
        _export(f"self_reflection_r{r}", build_self_reflection_graph(_MODEL, max_revisions=r))

    print("\nDone.")


if __name__ == "__main__":
    main()
