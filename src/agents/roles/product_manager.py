"""Product Manager agent: translates a coding problem into a structured PRD."""

from langchain_core.language_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState

_SYSTEM_PROMPT = (
    "You are a software product manager. Given a coding problem, produce a structured "
    "PRD with exactly these sections:\n"
    "(1) Problem Summary\n"
    "(2) Acceptance Criteria as a numbered list of testable conditions\n"
    "(3) Edge Cases to handle\n"
    "(4) Out of scope\n"
    "Be precise. Output only the PRD, no preamble."
)


class ProductManagerAgent(BaseAgent):
    """First node in the sequential pipeline: produces the PRD artifact.

    Reads the problem statement, function signature, and test cases from state,
    then writes a structured PRD to ``state["prd"]``.
    """

    def __init__(self, model_name: str, llm_client: BaseChatModel) -> None:
        """Initialise the ProductManagerAgent.

        Args:
            model_name: HuggingFace model repo ID.
            llm_client: Configured LangChain chat model.
        """
        super().__init__(
            role="product_manager",
            model_name=model_name,
            llm_client=llm_client,
        )

    def run(self, state: AgentState) -> AgentState:
        """Generate the PRD from the problem description and write it to state.

        Args:
            state: Current shared agent state. Reads ``problem_statement``,
                ``function_signature``, and ``test_cases``.

        Returns:
            Updated state with ``prd``, ``tokens_input``, ``tokens_output``,
            and ``latency_seconds`` populated.
        """
        test_cases_block = "\n".join(state["test_cases"]) if state["test_cases"] else "(none provided)"

        user_prompt = (
            f"Problem statement:\n{state['problem_statement']}\n\n"
            f"Function signature: {state['function_signature']}\n\n"
            f"Known test cases:\n{test_cases_block}\n\n"
            "Write the PRD now."
        )

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)

        return {
            **state,
            "prd": response_text.strip(),
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
        }
