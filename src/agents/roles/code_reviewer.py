"""Code Reviewer agent: produces a structured verdict on the generated code."""

import json
import logging

from langchain_groq import ChatGroq

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior Python code reviewer. You receive: the implemented code, "
    "the test results (pass/fail per test), and the original design document. "
    "Produce a structured review with:\n"
    "(1) VERDICT: APPROVE or REQUEST_CHANGES on the first line\n"
    "(2) Issues found (if any): numbered list, each with severity "
    "[CRITICAL|MAJOR|MINOR]\n"
    "(3) Suggested fixes for each issue\n"
    "If all tests pass and the design is followed: output "
    "\"VERDICT: APPROVE\" and nothing else."
)

_VALID_VERDICTS = {"VERDICT: APPROVE", "VERDICT: REQUEST_CHANGES"}


class CodeReviewerAgent(BaseAgent):
    """Fifth (final) node in the sequential pipeline: structured code review.

    Reads ``code_artifact``, ``test_results``, and ``design_doc`` from state;
    writes a review string starting with a verdict line to
    ``state["review_comments"]``.
    """

    def __init__(self, model_name: str, groq_client: ChatGroq) -> None:
        """Initialise the CodeReviewerAgent.

        Args:
            model_name: Groq model identifier.
            groq_client: Configured ChatGroq instance.
        """
        super().__init__(
            role="code_reviewer",
            model_name=model_name,
            groq_client=groq_client,
        )

    def run(self, state: AgentState) -> AgentState:
        """Review the generated code and produce a verdict.

        Args:
            state: Current shared agent state. Reads ``code_artifact``,
                ``test_results``, and ``design_doc``.

        Returns:
            Updated state with ``review_comments``, ``tokens_input``,
            and ``tokens_output`` populated.

        Raises:
            ValueError: If the LLM response does not start with a valid verdict
                line (``VERDICT: APPROVE`` or ``VERDICT: REQUEST_CHANGES``).
        """
        test_summary = _format_test_results(state["test_results"])

        user_prompt = (
            f"Code to review:\n```python\n{state['code_artifact']}\n```\n\n"
            f"Test results:\n{test_summary}\n\n"
            f"Original design document:\n{state['design_doc']}\n\n"
            "Write your review now."
        )

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)
        review = response_text.strip()

        _parse_verdict(review)  # raises ValueError if verdict line is missing

        return {
            **state,
            "review_comments": review,
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_verdict(review_text: str) -> str:
    """Extract and validate the verdict from the first line of the review.

    Args:
        review_text: Full review string produced by the LLM.

    Returns:
        The verdict string: ``"VERDICT: APPROVE"`` or
        ``"VERDICT: REQUEST_CHANGES"``.

    Raises:
        ValueError: If the first non-empty line is not a recognised verdict.
    """
    for line in review_text.splitlines():
        first_line = line.strip()
        if first_line:
            if first_line in _VALID_VERDICTS:
                return first_line
            raise ValueError(
                f"Review does not start with a valid verdict line. "
                f"Got: {first_line!r}. "
                f"Expected one of: {sorted(_VALID_VERDICTS)}"
            )
    raise ValueError("Review text is empty — no verdict found.")


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
