"""Abstract base class for all agents in the multi-agent system."""

import os
import threading
import time
import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.state.schema import AgentState

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2.0

# Optional global pacing between LLM calls. When the backend imposes a
# per-minute rate limit (e.g. Cerebras free tier) we space requests apart
# proactively to avoid 429 responses. The value is read once at import
# time from the ``LLM_MIN_INTERVAL_S`` environment variable; defaults to
# zero (no pacing) so the local Ollama path is unaffected.
_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "0"))
_pacing_lock = threading.Lock()
_last_call_ts: float = 0.0


def _pace_request() -> None:
    """Block until ``_MIN_INTERVAL_S`` seconds have passed since the last call."""
    global _last_call_ts
    if _MIN_INTERVAL_S <= 0:
        return
    with _pacing_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL_S - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


class BaseAgent(ABC):
    """Abstract base for all LangGraph node agents.

    Subclasses implement `run` to transform an AgentState. LLM calls go
    through `_call_llm`, which handles retries and token accumulation.
    """

    def __init__(self, role: str, model_name: str, llm_client: BaseChatModel) -> None:
        """Initialise the agent.

        Args:
            role: Human-readable role label (e.g. 'monolithic_solver').
            model_name: Model identifier (Ollama tag or Cerebras model ID).
            llm_client: Configured LangChain chat model to use for inference.
        """
        self.role = role
        self.model_name = model_name
        self._client = llm_client

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Execute the agent's task and return the updated state.

        Args:
            state: Current shared agent state.

        Returns:
            Updated AgentState with any new fields populated.
        """
        ...

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call the LLM with retry logic and return the response with token counts.

        Retries up to `_MAX_RETRIES` times with exponential backoff on any
        exception (rate limits, transient network errors, etc.).

        Args:
            system_prompt: Instruction context for the model.
            user_prompt: The actual user-facing message.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                _pace_request()
                response = self._client.invoke(messages)
                response_text: str = response.content  # type: ignore[assignment]

                usage = getattr(response, "usage_metadata", None) or {}
                input_tokens: int = usage.get("input_tokens", 0)
                output_tokens: int = usage.get("output_tokens", 0)

                return response_text, input_tokens, output_tokens

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait = _BASE_BACKOFF_SECONDS ** (attempt + 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d) for role=%s: %s. Retrying in %.1fs.",
                    attempt + 1,
                    _MAX_RETRIES,
                    self.role,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"All {_MAX_RETRIES} LLM attempts failed for role={self.role}."
        ) from last_error
