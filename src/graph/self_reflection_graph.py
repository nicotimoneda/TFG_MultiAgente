"""LangGraph graph for Config 3: sequential pipeline with conditional self-reflection loop."""

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

_SYSTEM_PROMPT = (
    "You are a Python developer. Given a design document and function signature, "
    "implement the function. Rules:\n"
    "(1) Output ONLY the Python function inside ```python``` fences\n"
    "(2) Match the exact function signature provided\n"
    "(3) Include inline comments for non-obvious logic\n"
    "(4) No imports unless strictly necessary\n"
    "(5) No test code in the output"
)


class ReflectiveDeveloperAgent(DeveloperAgent):
    """Developer agent variant for Config 3 that incorporates reviewer feedback.

    Overrides ``run()`` to prepend ``review_comments`` to the user prompt when
    feedback is present, enabling iterative self-correction. Uses temperature=0.4
    to introduce mild diversity during revision passes.
    """

    def run(self, state: AgentState) -> AgentState:
        """Implement or revise the function, incorporating review feedback if present.

        When ``state["review_comments"]`` is non-empty (i.e. a revision pass),
        the feedback is prepended to the prompt so the LLM addresses it explicitly.

        Args:
            state: Current shared agent state. Reads ``design_doc``,
                ``function_signature``, ``problem_statement``, and
                ``review_comments``.

        Returns:
            Updated state with ``code_artifact``, ``tokens_input``,
            ``tokens_output`` populated.
        """
        base_prompt = (
            f"Problem statement:\n{state['problem_statement']}\n\n"
            f"Function signature: {state['function_signature']}\n\n"
            f"Design document:\n{state['design_doc']}\n\n"
            "Implement the function now."
        )

        if state["review_comments"]:
            user_prompt = (
                f"Previous review feedback:\n{state['review_comments']}\n\n"
                f"Revise the implementation accordingly.\n\n{base_prompt}"
            )
        else:
            user_prompt = base_prompt

        import re as _re
        _CODE_FENCE_RE = _re.compile(r"```python\s*\n(.*?)```", _re.DOTALL)

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)

        match = _CODE_FENCE_RE.search(response_text)
        code = match.group(1).strip() if match else ""
        if not code:
            logger.warning(
                "No ```python``` fence in reflective developer response for %s; using raw text.",
                state["problem_id"],
            )
            code = response_text.strip()

        return {
            **state,
            "code_artifact": code,
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
        }


def build_self_reflection_graph(model_name: str, max_revisions: int = 1) -> CompiledGraph:
    """Construct and compile the self-reflection multi-agent LangGraph.

    Node order: pm → architect → developer → qa → reviewer, with a conditional
    edge from reviewer back to developer when verdict is REQUEST_CHANGES and
    revision_count has not reached max_revisions.

    Args:
        model_name: Groq model identifier (e.g. ``'llama-3.3-70b-versatile'``).
        max_revisions: Maximum number of developer revision cycles allowed.
            When this limit is reached the graph terminates regardless of verdict.

    Returns:
        Compiled LangGraph ready to be invoked with an AgentState.
    """
    llm_client = create_chat_client(model_name)
    llm_client_dev = create_chat_client(model_name, temperature=0.4)

    pm = ProductManagerAgent(model_name=model_name, llm_client=llm_client)
    architect = ArchitectAgent(model_name=model_name, llm_client=llm_client)
    reflective_developer = ReflectiveDeveloperAgent(
        model_name=model_name, llm_client=llm_client_dev
    )
    qa = QATesterAgent(model_name=model_name, llm_client=llm_client)
    reviewer = CodeReviewerAgent(model_name=model_name, llm_client=llm_client)

    def _developer_node(state: AgentState) -> AgentState:
        """Wrap the reflective developer: increments revision_count on re-entry."""
        if state["review_comments"]:
            state = {**state, "revision_count": state["revision_count"] + 1}
        return reflective_developer.run(state)

    def _reflection_router(state: AgentState) -> str:
        """Routes after reviewer: 'end' if approved or max revisions reached, 'developer' otherwise."""
        comments = state.get("review_comments", "")
        first_line = comments.split("\n")[0].strip() if comments else ""

        if first_line == "VERDICT: APPROVE":
            return END

        # REQUEST_CHANGES path
        if state["revision_count"] < max_revisions:
            return "developer"

        # Forced termination: max_revisions reached
        logger.info(
            "Max revisions (%d) reached for %s — terminating.",
            max_revisions,
            state["problem_id"],
        )
        return END

    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("pm", pm.run)
    builder.add_node("architect", architect.run)
    builder.add_node("developer", _developer_node)
    builder.add_node("qa", qa.run)
    builder.add_node("reviewer", reviewer.run)

    builder.add_edge(START, "pm")
    builder.add_edge("pm", "architect")
    builder.add_edge("architect", "developer")
    builder.add_edge("developer", "qa")
    builder.add_edge("qa", "reviewer")
    builder.add_conditional_edges("reviewer", _reflection_router)

    return builder.compile()


def run_self_reflection(
    problem: dict,
    model_name: str,
    max_revisions: int = 1,
) -> AgentState:
    """Run the self-reflection graph on a single HumanEval/MBPP problem dict.

    Initialises AgentState from the problem dict, invokes the compiled graph,
    and returns the final state with wall-clock latency set.

    Args:
        problem: Dict with keys ``task_id``, ``prompt``, ``entry_point``,
                 ``test``, ``canonical_solution`` (HumanEval schema).
        model_name: Groq model identifier.
        max_revisions: Maximum number of revision cycles. Defaults to 1.

    Returns:
        Final AgentState after all pipeline stages (and any revision cycles) complete.
    """
    graph = build_self_reflection_graph(model_name=model_name, max_revisions=max_revisions)

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
        "config_name": "self_reflection",
    }

    t_wall_start = time.perf_counter()
    final_state: AgentState = graph.invoke(initial_state)  # type: ignore[assignment]
    final_state["latency_seconds"] = time.perf_counter() - t_wall_start

    return final_state
