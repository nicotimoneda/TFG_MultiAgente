"""Class-aware Developer agent variant for the ClassEval benchmark.

ClassEval problems require generating **classes** with multiple methods rather
than single functions. The default ``DeveloperAgent`` prompts the model to
emit a function body inside a ``python`` fence; reusing it for ClassEval
yields code that passes the fence check but fails the class-level test
harness.

This subclass overrides the system prompt and the user-prompt template so the
Developer emits ``class X:`` and complete method implementations. The
``code_artifact`` is still extracted from the ``python`` fence and the
sandbox concatenates the original problem prompt before execution, so the
class skeleton from the benchmark is in scope at evaluation time.

This module is wired only when the runner is invoked with the ClassEval
benchmark. The existing HumanEval/MBPP pipeline keeps using the function-level
``DeveloperAgent`` unchanged.
"""

from __future__ import annotations

import re
import logging

from langchain_core.language_models import BaseChatModel

from src.agents.roles.developer import DeveloperAgent
from src.state.schema import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Python developer. Given a design document, a class skeleton, "
    "and the class name, implement every method of the class. Rules:\n"
    "(1) Output ONLY the complete Python class inside a single ```python``` "
    "fenced block.\n"
    "(2) Use the exact class name provided.\n"
    "(3) Implement every method declared in the skeleton; do not omit any.\n"
    "(4) Keep method signatures identical to the skeleton's.\n"
    "(5) Include inline comments only for non-obvious logic.\n"
    "(6) No test code, no usage examples, no prose outside the fence."
)

_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


class ClassEvalDeveloperAgent(DeveloperAgent):
    """Developer variant that emits a class instead of a function.

    Drop-in replacement for ``DeveloperAgent`` inside the sequential or
    self-reflection graphs when running on ClassEval.
    """

    def __init__(self, model_name: str, llm_client: BaseChatModel) -> None:
        super().__init__(model_name=model_name, llm_client=llm_client)

    def run(self, state: AgentState) -> AgentState:
        user_prompt = (
            f"Class skeleton (problem statement):\n{state['problem_statement']}\n\n"
            f"Class name: {state['function_signature']}\n\n"
            f"Design document:\n{state['design_doc']}\n\n"
            "Implement the full class now."
        )

        response_text, in_tok, out_tok = self._call_llm(_SYSTEM_PROMPT, user_prompt)

        match = _CODE_FENCE_RE.search(response_text)
        code = match.group(1).strip() if match else ""
        if not code:
            logger.warning(
                "No ```python``` fence in ClassEval developer response for %s; using raw text.",
                state["problem_id"],
            )
            code = response_text.strip()

        return {
            **state,
            "code_artifact": code,
            "tokens_input": state["tokens_input"] + in_tok,
            "tokens_output": state["tokens_output"] + out_tok,
        }
