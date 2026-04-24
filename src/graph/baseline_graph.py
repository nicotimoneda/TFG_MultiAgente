"""LangGraph graph for Config 1: single-node baseline solver."""

import time
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from src.state.schema import AgentState
from src.agents.baseline_agent import BaselineAgent
from src.llm.client_factory import create_chat_client

logger = logging.getLogger(__name__)


def build_baseline_graph(model_name: str) -> CompiledGraph:
    """Construct and compile the baseline (monolithic) LangGraph.

    The graph contains a single node — "solver" — backed by BaselineAgent.
    Control flow: START → solver → END.

    Args:
        model_name: HuggingFace model repo ID
            (e.g. ``'meta-llama/Llama-3.1-70B-Instruct'``).

    Returns:
        Compiled LangGraph ready to be invoked with an AgentState.
    """
    llm_client = create_chat_client(model_name)
    agent = BaselineAgent(model_name=model_name, llm_client=llm_client)

    def solver_node(state: AgentState) -> AgentState:
        return agent.run(state)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("solver", solver_node)
    builder.add_edge(START, "solver")
    builder.add_edge("solver", END)

    return builder.compile()


def run_baseline(problem: dict, model_name: str) -> AgentState:
    """Run the baseline graph on a single HumanEval/MBPP problem dict.

    Initialises AgentState from the problem dict returned by
    ``humaneval_loader.load_humaneval()`` (or equivalent MBPP loader),
    invokes the compiled graph, and returns the final state.

    Args:
        problem: Dict with keys ``task_id``, ``prompt``, ``entry_point``,
                 ``test``, ``canonical_solution`` (HumanEval schema).
        model_name: Groq model identifier.

    Returns:
        Final AgentState after the graph has completed.
    """
    graph = build_baseline_graph(model_name)

    # Extract individual assert lines from the HumanEval test harness so the
    # sandbox can run each test case independently.
    raw_test: str = problem.get("test", "")
    test_cases = _extract_assert_lines(raw_test, problem.get("entry_point", ""))

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
        "config_name": "baseline",
    }

    t_wall_start = time.perf_counter()
    final_state: AgentState = graph.invoke(initial_state)  # type: ignore[assignment]
    elapsed = time.perf_counter() - t_wall_start

    # Overwrite latency with the true wall-clock duration so it includes any
    # graph overhead beyond what the agent itself measures.
    final_state["latency_seconds"] = elapsed
    return final_state


def _extract_assert_lines(test_source: str, entry_point: str = "") -> list[str]:
    """Build executable test cases from a HumanEval/evalplus test string.

    evalplus wraps all assertions inside ``check(candidate)`` and never calls
    ``check`` at module level. When that pattern is detected the whole test
    source is returned as a single string with ``check(<entry_point>)`` appended,
    so the sandbox can exec the solution, then exec this string to run every
    assertion correctly. Falls back to extracting bare ``assert`` lines for
    other formats (e.g. plain MBPP tests).

    Args:
        test_source: Full source of the test field from the dataset.
        entry_point: Name of the function defined by the solution code.

    Returns:
        List of executable test-case strings (usually one element for evalplus).
    """
    if "def check(candidate):" in test_source and entry_point:
        return [test_source + f"\ncheck({entry_point})"]
    lines = []
    for line in test_source.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            lines.append(stripped)
    return lines
