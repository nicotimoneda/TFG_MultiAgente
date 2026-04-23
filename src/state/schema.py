"""Shared state schema for all LangGraph agent configurations.

This module defines the AgentState TypedDict that flows through every node
in the graph, regardless of configuration (baseline, sequential, self_reflection).
"""

from typing import TypedDict


class AgentState(TypedDict):
    """Central state object passed between LangGraph nodes.

    Fields are populated progressively as the graph executes. For the
    baseline configuration most artifact fields remain empty strings.
    """

    # --- Problem identity ---
    problem_id: str
    """HumanEval/MBPP problem identifier, e.g. 'HumanEval/0'."""

    problem_statement: str
    """Natural language description of the coding problem."""

    function_signature: str
    """Expected function name and signature extracted from the prompt."""

    test_cases: list[str]
    """List of assert statements taken directly from the benchmark."""

    # --- Agent artifacts ---
    prd: str
    """Product Manager artifact: requirements document."""

    design_doc: str
    """Architect artifact: technical design document."""

    code_artifact: str
    """Developer artifact: the Python code string to be evaluated."""

    test_results: dict[str, bool]
    """Mapping of each test-case string to pass (True) / fail (False)."""

    review_comments: str
    """Code Reviewer artifact: structured feedback on the generated code."""

    # --- Iteration control ---
    revision_count: int
    """Number of self-reflection iterations completed (starts at 0)."""

    # --- Telemetry ---
    tokens_input: int
    """Cumulative input tokens consumed across all LLM calls."""

    tokens_output: int
    """Cumulative output tokens consumed across all LLM calls."""

    latency_seconds: float
    """Wall-clock time from graph invocation to completion, in seconds."""

    # --- Run metadata ---
    config_name: str
    """Which configuration produced this state: 'baseline' | 'sequential' | 'self_reflection'."""
