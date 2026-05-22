"""ClassEval-aware sequential graph.

Mirrors the standard sequential pipeline (PM → Architect → Developer → QA →
Reviewer) but swaps the function-level ``DeveloperAgent`` for the
class-aware ``ClassEvalDeveloperAgent`` (see ``src/agents/roles/
developer_classeval.py``). Use this graph when the benchmark provides class
skeletons rather than single function signatures.

The remaining four roles are reused unchanged — the PRD, design document
and code review do not depend on whether the artifact is a function or a
class, only on what the Developer emitted.
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
from src.agents.roles.developer_classeval import ClassEvalDeveloperAgent
from src.agents.roles.qa_tester import QATesterAgent
from src.agents.roles.code_reviewer import CodeReviewerAgent
from src.graph.baseline_graph import _extract_assert_lines

logger = logging.getLogger(__name__)


def build_classeval_sequential_graph(model_name: str) -> CompiledGraph:
    """Build the sequential pipeline with a class-aware Developer."""
    llm_client = create_chat_client(model_name)

    pm = ProductManagerAgent(model_name=model_name, llm_client=llm_client)
    architect = ArchitectAgent(model_name=model_name, llm_client=llm_client)
    developer = ClassEvalDeveloperAgent(model_name=model_name, llm_client=llm_client)
    qa = QATesterAgent(model_name=model_name, llm_client=llm_client)
    reviewer = CodeReviewerAgent(model_name=model_name, llm_client=llm_client)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("pm", pm.run)
    builder.add_node("architect", architect.run)
    builder.add_node("developer", developer.run)
    builder.add_node("qa", qa.run)
    builder.add_node("reviewer", reviewer.run)

    builder.add_edge(START, "pm")
    builder.add_edge("pm", "architect")
    builder.add_edge("architect", "developer")
    builder.add_edge("developer", "qa")
    builder.add_edge("qa", "reviewer")
    builder.add_edge("reviewer", END)
    return builder.compile()


def run_classeval_sequential(problem: dict, model_name: str) -> AgentState:
    """Run the class-aware sequential graph on a single ClassEval problem.

    The ``test`` field of ClassEval problems carries the full pytest module
    that exercises the generated class. The sandbox harness treats it as a
    single test case (see ``_extract_assert_lines`` fallback path).
    """
    graph = build_classeval_sequential_graph(model_name)

    test_cases = _extract_assert_lines(
        problem.get("test", ""), problem.get("entry_point", "")
    )
    # ClassEval test fields rarely match the ``check(candidate)`` pattern, so
    # the fallback extracts ``assert`` lines. If neither pattern fires, fall
    # back to running the whole test text as a single block.
    if not test_cases and problem.get("test"):
        test_cases = [problem["test"]]

    initial_state: AgentState = {
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
        "config_name": "classeval_sequential",
    }

    t0 = time.perf_counter()
    final: AgentState = graph.invoke(initial_state)  # type: ignore[assignment]
    final["latency_seconds"] = time.perf_counter() - t0
    return final
