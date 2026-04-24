"""LangGraph graph for Config 2: sequential five-role pipeline."""

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


def build_sequential_graph(model_name: str) -> CompiledGraph:
    """Construct and compile the sequential multi-agent LangGraph.

    Node order: pm → architect → developer → qa → reviewer.
    No conditional edges, no cycles. All agents share the same HuggingFace
    client (and therefore the same model).

    Args:
        model_name: HuggingFace model repo ID
            (e.g. ``'meta-llama/Llama-3.1-70B-Instruct'``).

    Returns:
        Compiled LangGraph ready to be invoked with an AgentState.
    """
    llm_client = create_chat_client(model_name)

    pm = ProductManagerAgent(model_name=model_name, llm_client=llm_client)
    architect = ArchitectAgent(model_name=model_name, llm_client=llm_client)
    developer = DeveloperAgent(model_name=model_name, llm_client=llm_client)
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


def run_sequential(problem: dict, model_name: str) -> AgentState:
    """Run the sequential graph on a single HumanEval/MBPP problem dict.

    Initialises AgentState from the problem dict, invokes the compiled graph,
    and returns the final state with wall-clock latency set.

    Args:
        problem: Dict with keys ``task_id``, ``prompt``, ``entry_point``,
                 ``test``, ``canonical_solution`` (HumanEval schema).
        model_name: Groq model identifier.

    Returns:
        Final AgentState after all five pipeline stages have completed.
    """
    graph = build_sequential_graph(model_name)

    test_cases = _extract_assert_lines(problem.get("test", ""))

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
        "config_name": "sequential",
    }

    t_wall_start = time.perf_counter()
    final_state: AgentState = graph.invoke(initial_state)  # type: ignore[assignment]
    final_state["latency_seconds"] = time.perf_counter() - t_wall_start

    return final_state
