"""Code Reviewer agent: produces a structured verdict on the generated code."""

import json
import logging

from langchain_core.language_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior Python code reviewer. You receive the implemented code, "
    "the test results (pass/fail per test), and the original design document. "
    "Produce a concise structured review of the CODE (not the design document):\n"
    "(1) Issues found: numbered list, each tagged [CRITICAL|MAJOR|MINOR]\n"
    "(2) Suggested fix for each issue\n"
    "If everything is fine, say so briefly in one sentence.\n"
    "Do NOT include any verdict line — the verdict is derived automatically "
    "from the test results."
)

_VALID_VERDICTS = {"VERDICT: APPROVE", "VERDICT: REQUEST_CHANGES"}


class CodeReviewerAgent(BaseAgent):
    """Fifth (final) node in the sequential pipeline: structured code review.

    Reads ``code_artifact``, ``test_results``, and ``design_doc`` from state.
    Writes a review string to ``state["review_comments"]`` whose first line is
    a canonical verdict (``VERDICT: APPROVE`` or ``VERDICT: REQUEST_CHANGES``)
    derived deterministically from the test outcomes, followed by the LLM's
    qualitative commentary.
    """

    def __init__(self, model_name: str, llm_client: BaseChatModel) -> None:
        """Initialise the CodeReviewerAgent.

        Args:
            model_name: Model identifier (e.g. an Ollama tag or a Cerebras model ID).
            llm_client: Configured LangChain chat model.
        """
        super().__init__(
            role="code_reviewer",
            model_name=model_name,
            llm_client=llm_client,
        )

    def run(self, state: AgentState) -> AgentState:
        """Review the generated code and emit verdict + commentary.

        Args:
            state: Current shared agent state. Reads ``code_artifact``,
                ``test_results``, and ``design_doc``.

        Returns:
            Updated state with ``review_comments`` (verdict line followed by
            the LLM commentary), ``tokens_input`` and ``tokens_output``
            populated.
        """
        test_summary = _format_test_results(state["test_results"])

        user_prompt = (
            f"Code to review:\n```python\n{state['code_artifact']}\n```\n\n"
            f"Test results:\n{test_summary}\n\n"
            f"Original design document:\n{state['design_doc']}\n\n"
            "Write your review now."
        )

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)
        commentary = response_text.strip()

        # Verdict is derived deterministically from test results, not asked of
        # the LLM. Rationale: APPROVE/REQUEST_CHANGES is by definition a
        # function of the tests, and small local models do not follow strict
        # format constraints reliably. The LLM's contribution is the
        # qualitative commentary that feeds the self-reflection revision step.
        verdict = _derive_verdict(state["test_results"])
        review = f"{verdict}\n\n{commentary}" if commentary else verdict

        return {
            **state,
            "review_comments": review,
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_verdict(test_results: dict) -> str:
    """Derive the verdict deterministically from test outcomes.

    APPROVE iff every per-test-case entry passed; REQUEST_CHANGES otherwise
    (including the empty-results case, which we treat as inconclusive).

    Args:
        test_results: ``state["test_results"]``. May contain a ``qa_summary``
            metadata entry which is excluded from the pass/fail tally.

    Returns:
        ``"VERDICT: APPROVE"`` or ``"VERDICT: REQUEST_CHANGES"``.
    """
    per_test = {k: v for k, v in (test_results or {}).items() if k != "qa_summary"}
    if per_test and all(per_test.values()):
        return "VERDICT: APPROVE"
    return "VERDICT: REQUEST_CHANGES"


def _format_test_results(test_results: dict) -> str:
    """Render the test_results dict as a readable string for the LLM prompt.

    Separates the ``qa_summary`` entry (which is a dict) from the
    per-test-case bool entries.

    Args:
        test_results: The ``state["test_results"]`` dict, which may contain
            a nested ``"qa_summary"`` dict alongside bool-valued entries.

    Returns:
        Human-readable string of results.
    """
    if not test_results:
        return "(no test results available)"

    lines: list[str] = []
    summary = test_results.get("qa_summary")
    if summary:
        lines.append(f"Summary: {json.dumps(summary)}")

    for test_case, result in test_results.items():
        if test_case == "qa_summary":
            continue
        status = "PASS" if result else "FAIL"
        lines.append(f"  [{status}] {test_case}")

    return "\n".join(lines)
