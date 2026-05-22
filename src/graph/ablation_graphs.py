"""Role-ablation variants of the sequential pipeline.

These graphs deliberately remove one role at a time so we can quantify the
contribution of each agent — a key research question from the propuesta:
"¿el reviewer mejora la calidad? ¿el tester encuentra bugs que el developer
no ve? ¿qué combinaciones de roles aportan más valor?".

Three variants are provided:

* ``no_pm``        — Architect reads the raw problem statement (no PRD).
* ``no_architect`` — Developer reads the PRD directly (no design doc).
* ``no_reviewer``  — Pipeline ends after QA; no LLM-side review feedback.

All variants share the same AgentState schema as the full sequential graph
so the runner, sandbox, and CSV writer keep working unchanged.
"""

from __future__ import annotations

import time
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from src.state.schema import AgentState
from src.llm.client_factory import create_chat_client
from src.agents.roles.product_manager import ProductManagerAgent
from src.agents.roles.architect import ArchitectAgent
from src.agents.roles.developer import DeveloperAgent
from src.agents.roles.qa_tester import QATesterAgent
from src.agents.roles.code_reviewer import CodeReviewerAgent
from src.graph.baseline_graph import _extract_assert_lines

logger = logging.getLogger(__name__)


def _initial_state(problem: dict, config_name: str) -> AgentState:
    test_cases = _extract_assert_lines(
        problem.get("test", ""), problem.get("entry_point", "")
    )
    return {
        "problem_id": problem["task_id"],
        "problem_statement": problem["prompt"],
        "function_signature": problem.get("entry_point", ""),
        "test_cases": test_cases,
        "prd": "",
        "design_doc": "",
        "code_artifact": "",
        "test_results": {},
        "review_comments": "",
        "revision_count": 0,
        "tokens_input": 0,
        "tokens_output": 0,
        "latency_seconds": 0.0,
        "config_name": config_name,
    }


def _seed_prd_from_problem(state: AgentState) -> AgentState:
    return {**state, "prd": state["problem_statement"]}


def _seed_design_from_prd(state: AgentState) -> AgentState:
    return {**state, "design_doc": state["prd"] or state["problem_statement"]}


def build_no_pm_graph(model_name: str) -> CompiledGraph:
    """Sequential without PM: architect → developer → qa → reviewer.

    A trivial seed node copies ``problem_statement`` into ``prd`` so the
    architect's prompt template stays valid without spending tokens on PM.
    """
    llm_client = create_chat_client(model_name)

    architect = ArchitectAgent(model_name=model_name, llm_client=llm_client)
    developer = DeveloperAgent(model_name=model_name, llm_client=llm_client)
    qa = QATesterAgent(model_name=model_name, llm_client=llm_client)
    reviewer = CodeReviewerAgent(model_name=model_name, llm_client=llm_client)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("seed_prd", _seed_prd_from_problem)
    builder.add_node("architect", architect.run)
    builder.add_node("developer", developer.run)
    builder.add_node("qa", qa.run)
    builder.add_node("reviewer", reviewer.run)

    builder.add_edge(START, "seed_prd")
    builder.add_edge("seed_prd", "architect")
    builder.add_edge("architect", "developer")
    builder.add_edge("developer", "qa")
    builder.add_edge("qa", "reviewer")
    builder.add_edge("reviewer", END)
    return builder.compile()


def build_no_architect_graph(model_name: str) -> CompiledGraph:
    """Sequential without architect: pm → developer → qa → reviewer.

    The developer reads the PRD as if it were the design document.
    """
    llm_client = create_chat_client(model_name)

    pm = ProductManagerAgent(model_name=model_name, llm_client=llm_client)
    developer = DeveloperAgent(model_name=model_name, llm_client=llm_client)
    qa = QATesterAgent(model_name=model_name, llm_client=llm_client)
    reviewer = CodeReviewerAgent(model_name=model_name, llm_client=llm_client)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("pm", pm.run)
    builder.add_node("seed_design", _seed_design_from_prd)
    builder.add_node("developer", developer.run)
    builder.add_node("qa", qa.run)
    builder.add_node("reviewer", reviewer.run)

    builder.add_edge(START, "pm")
    builder.add_edge("pm", "seed_design")
    builder.add_edge("seed_design", "developer")
    builder.add_edge("developer", "qa")
    builder.add_edge("qa", "reviewer")
    builder.add_edge("reviewer", END)
    return builder.compile()


def build_no_reviewer_graph(model_name: str) -> CompiledGraph:
    """Sequential without reviewer: pm → architect → developer → qa.

    QA still runs the sandbox; ``review_comments`` stays empty.
    """
    llm_client = create_chat_client(model_name)

    pm = ProductManagerAgent(model_name=model_name, llm_client=llm_client)
    architect = ArchitectAgent(model_name=model_name, llm_client=llm_client)
    developer = DeveloperAgent(model_name=model_name, llm_client=llm_client)
    qa = QATesterAgent(model_name=model_name, llm_client=llm_client)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("pm", pm.run)
    builder.add_node("architect", architect.run)
    builder.add_node("developer", developer.run)
    builder.add_node("qa", qa.run)

    builder.add_edge(START, "pm")
    builder.add_edge("pm", "architect")
    builder.add_edge("architect", "developer")
    builder.add_edge("developer", "qa")
    builder.add_edge("qa", END)
    return builder.compile()


_BUILDERS = {
    "no_pm": build_no_pm_graph,
    "no_architect": build_no_architect_graph,
    "no_reviewer": build_no_reviewer_graph,
}


def run_ablation(variant: str, problem: dict, model_name: str) -> AgentState:
    """Run a named ablation variant on a single problem dict.

    Args:
        variant: One of ``"no_pm"``, ``"no_architect"``, ``"no_reviewer"``.
        problem: Problem dict in the HumanEval/MBPP schema.
        model_name: Model identifier.

    Returns:
        Final ``AgentState`` with ``latency_seconds`` set to wall-clock time.
    """
    if variant not in _BUILDERS:
        raise ValueError(
            f"Unknown ablation variant: {variant!r}. "
            f"Choose from {sorted(_BUILDERS)}."
        )
    graph = _BUILDERS[variant](model_name)
    state = _initial_state(problem, f"ablation_{variant}")

    t0 = time.perf_counter()
    final: AgentState = graph.invoke(state)  # type: ignore[assignment]
    final["latency_seconds"] = time.perf_counter() - t0
    return final
