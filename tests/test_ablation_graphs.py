"""Smoke tests for the role-ablation graphs.

We verify the topology (which nodes exist and how they connect) without
making any LLM calls. The compile step is enough to validate that the
graph definition is well-formed.
"""

from __future__ import annotations

import os

import pytest

# These graphs build a real LLM client at compile time. Force Ollama so we
# don't hit the Cerebras key check; the client object is created but never
# invoked here.
os.environ.setdefault("LLM_BACKEND", "ollama")

from src.graph.ablation_graphs import (  # noqa: E402
    build_no_architect_graph,
    build_no_pm_graph,
    build_no_reviewer_graph,
    run_ablation,
)

_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"


def _nodes(graph):
    return set(graph.get_graph().nodes)


def test_no_pm_topology():
    g = build_no_pm_graph(_MODEL)
    nodes = _nodes(g)
    assert {"architect", "developer", "qa", "reviewer", "seed_prd"} <= nodes
    assert "pm" not in nodes


def test_no_architect_topology():
    g = build_no_architect_graph(_MODEL)
    nodes = _nodes(g)
    assert {"pm", "developer", "qa", "reviewer", "seed_design"} <= nodes
    assert "architect" not in nodes


def test_no_reviewer_topology():
    g = build_no_reviewer_graph(_MODEL)
    nodes = _nodes(g)
    assert {"pm", "architect", "developer", "qa"} <= nodes
    assert "reviewer" not in nodes


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        run_ablation("nonexistent_role", {"task_id": "x"}, _MODEL)
