"""QA Tester agent: runs existing test cases through the sandbox (no LLM call)."""

from langchain_core.language_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.evaluation.sandbox import execute_code_safely
from src.state.schema import AgentState


class QATesterAgent(BaseAgent):
    """Fourth node in the sequential pipeline: deterministic test executor.

    This agent does NOT call the LLM. It runs the test cases already present
    in ``state["test_cases"]`` against ``state["code_artifact"]`` using the
    subprocess sandbox, then writes a results dict (plus a summary entry) to
    ``state["test_results"]``.

    Token counts are not modified because no LLM call is made.
    """

    def __init__(self, model_name: str, llm_client: BaseChatModel) -> None:
        """Initialise the QATesterAgent.

        Args:
            model_name: Model identifier (unused but required by BaseAgent).
            llm_client: BaseChatModel instance (unused but required by BaseAgent).
        """
        super().__init__(
            role="qa_tester",
            model_name=model_name,
            llm_client=llm_client,
        )

    def run(self, state: AgentState) -> AgentState:
        """Execute test cases in the sandbox and record pass/fail results.

        Runs each assertion in ``state["test_cases"]`` against
        ``state["code_artifact"]`` inside an isolated subprocess. Appends a
        ``"qa_summary"`` key to the results dict containing aggregate counts.

        Args:
            state: Current shared agent state. Reads ``code_artifact`` and
                ``test_cases``.

        Returns:
            Updated state with ``test_results`` populated. Token counts and
            latency are not modified (no LLM call).

        Note:
            If ``test_cases`` is empty, ``test_results`` will contain only
            the ``qa_summary`` with all counts at zero.
        """
        test_cases = state["test_cases"]
        code = state["code_artifact"]

        # Prepend the original problem prompt (imports + signature context) to
        # the generated code before execution. This follows the canonical
        # HumanEval evaluation pattern (Chen et al., 2021), where pass@k is
        # computed on `prompt + completion`. Without this, a Developer that
        # omits e.g. `from typing import List` would fail every test even when
        # the function body is correct — a frequent failure mode with smaller
        # local models. The duplicated function signature is harmless: Python
        # binds the last definition.
        full_code = f"{state['problem_statement']}\n\n{code}" if code else code

        raw_results: dict[str, bool] = {}
        if test_cases and full_code:
            raw_results = execute_code_safely(full_code, test_cases)

        passed = sum(1 for v in raw_results.values() if v)
        failed = sum(1 for v in raw_results.values() if not v)
        total = passed + failed
        pass_rate = round(passed / total, 4) if total > 0 else 0.0

        test_results: dict[str, bool | dict] = {**raw_results}  # type: ignore[assignment]
        test_results["qa_summary"] = {  # type: ignore[assignment]
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
        }

        return {
            **state,
            "test_results": test_results,  # type: ignore[typeddict-item]
        }
