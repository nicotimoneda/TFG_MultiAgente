"""Developer agent: implements the function from the design document."""

import re
import logging

from langchain_core.language_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState

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

_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


class DeveloperAgent(BaseAgent):
    """Third node in the sequential pipeline: produces the code artifact.

    Reads ``design_doc``, ``function_signature``, and ``problem_statement``
    from state; writes Python source code to ``state["code_artifact"]``.
    """

    def __init__(self, model_name: str, llm_client: BaseChatModel) -> None:
        """Initialise the DeveloperAgent.

        Args:
            model_name: HuggingFace model repo ID.
            llm_client: Configured LangChain chat model.
        """
        super().__init__(
            role="developer",
            model_name=model_name,
            llm_client=llm_client,
        )

    def run(self, state: AgentState) -> AgentState:
        """Implement the function according to the design document.

        Args:
            state: Current shared agent state. Reads ``design_doc``,
                ``function_signature``, and ``problem_statement``.

        Returns:
            Updated state with ``code_artifact``, ``tokens_input``,
            ``tokens_output``, and ``latency_seconds`` populated.
        """
        user_prompt = (
            f"Problem statement:\n{state['problem_statement']}\n\n"
            f"Function signature: {state['function_signature']}\n\n"
            f"Design document:\n{state['design_doc']}\n\n"
            "Implement the function now."
        )

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)

        code = _extract_code(response_text)
        if not code:
            logger.warning(
                "No ```python``` fence in developer response for %s; using raw text.",
                state["problem_id"],
            )
            code = response_text.strip()

        return {
            **state,
            "code_artifact": code,
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
        }


def _extract_code(text: str) -> str:
    """Extract the first ```python``` fenced block from *text*.

    Args:
        text: Raw LLM response string.

    Returns:
        Code inside the fence, or empty string if none found.
    """
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else ""
