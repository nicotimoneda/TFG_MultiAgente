"""Architect agent: converts a PRD into a concrete technical design document."""

from langchain_core.language_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState

_SYSTEM_PROMPT = (
    "You are a software architect. Given a PRD and a function signature, produce a "
    "technical design document with:\n"
    "(1) Algorithm choice and justification\n"
    "(2) Data structures used\n"
    "(3) Step-by-step implementation plan in plain English\n"
    "(4) Known failure modes and how to handle them\n"
    "Output only the design document."
)


class ArchitectAgent(BaseAgent):
    """Second node in the sequential pipeline: produces the design document.

    Reads ``prd`` and ``function_signature`` from state, writes a technical
    design document to ``state["design_doc"]``.
    """

    def __init__(self, model_name: str, llm_client: BaseChatModel) -> None:
        """Initialise the ArchitectAgent.

        Args:
            model_name: HuggingFace model repo ID.
            llm_client: Configured LangChain chat model.
        """
        super().__init__(
            role="architect",
            model_name=model_name,
            llm_client=llm_client,
        )

    def run(self, state: AgentState) -> AgentState:
        """Generate the technical design document from the PRD.

        Args:
            state: Current shared agent state. Reads ``prd`` and
                ``function_signature``.

        Returns:
            Updated state with ``design_doc``, ``tokens_input``,
            ``tokens_output``, and ``latency_seconds`` populated.
        """
        user_prompt = (
            f"PRD:\n{state['prd']}\n\n"
            f"Function signature: {state['function_signature']}\n\n"
            "Write the technical design document now."
        )

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)

        return {
            **state,
            "design_doc": response_text.strip(),
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
        }
