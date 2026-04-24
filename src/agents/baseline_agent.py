"""Baseline monolithic agent: solves coding problems in a single LLM call."""

import re
import time
import logging

from langchain_core.language_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Given a coding problem, output ONLY valid Python code inside a single "
    "```python\\n...\\n``` fenced block. "
    "Do NOT include any explanation, comments outside the code, or prose. "
    "The code must define the requested function and be self-contained."
)

_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


class BaselineAgent(BaseAgent):
    """Single-node agent that produces code directly from the problem statement.

    This is Config 1 of the TFG evaluation: a monolithic solver with no
    intermediate artifacts (no PRD, no design doc, no review).
    """

    def __init__(self, model_name: str, llm_client: BaseChatModel) -> None:
        """Initialise the baseline agent.

        Args:
            model_name: HuggingFace model repo ID.
            llm_client: Configured LangChain chat model.
        """
        super().__init__(
            role="monolithic_solver",
            model_name=model_name,
            llm_client=llm_client,
        )

    def run(self, state: AgentState) -> AgentState:
        """Solve the problem and populate ``code_artifact`` in the state.

        Builds a user prompt from the problem statement and function signature,
        calls the LLM, extracts the code block, and writes it to state.

        Args:
            state: Current shared agent state.

        Returns:
            Updated state with ``code_artifact``, ``tokens_input``,
            ``tokens_output``, and ``latency_seconds`` populated.
        """
        t_start = time.perf_counter()

        user_prompt = (
            f"Problem:\n{state['problem_statement']}\n\n"
            f"Function signature:\n{state['function_signature']}\n\n"
            "Write the complete Python implementation."
        )

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)

        code = _extract_code(response_text)
        if not code:
            logger.warning(
                "No ```python``` fence found in response for %s; using raw response.",
                state["problem_id"],
            )
            code = response_text.strip()

        elapsed = time.perf_counter() - t_start

        return {
            **state,
            "code_artifact": code,
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
            "latency_seconds": state["latency_seconds"] + elapsed,
        }


def _extract_code(text: str) -> str:
    """Extract the first ```python``` fenced block from *text*.

    Args:
        text: Raw LLM response string.

    Returns:
        The code inside the fence, or an empty string if none is found.
    """
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else ""
