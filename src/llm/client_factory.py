"""Factory for creating chat clients used across all graph configurations.

Backend is selected by the ``LLM_BACKEND`` environment variable:

- ``cerebras`` (default): Cerebras Inference API, requires ``CEREBRAS_API_KEY``.
- ``ollama``: local Ollama server (OpenAI-compatible) at
  ``OLLAMA_BASE_URL`` (default ``http://localhost:11434/v1``).

Per-backend default model can be overridden with ``LLM_MODEL``.
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


_DEFAULT_MODELS = {
    "cerebras": "qwen-3-235b-a22b-instruct-2507",
    "ollama": "qwen2.5-coder:7b-instruct-q4_K_M",
}


def get_llm_client(
    model_name: str | None = None,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Create a chat client for the active backend.

    Args:
        model_name: Override model ID. If ``None``, uses ``LLM_MODEL`` env var
            or the per-backend default.
        temperature: Sampling temperature. 0.2 for deterministic outputs;
            0.4 for the reflective developer pass.

    Returns:
        A LangChain ``BaseChatModel`` compatible with ``.invoke(messages)``.
    """
    backend = os.getenv("LLM_BACKEND", "cerebras").lower()
    model = model_name or os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(backend, "")
    if not model:
        raise ValueError(f"No model resolved for backend={backend!r}; set LLM_MODEL.")

    if backend == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr("ollama"),
            model=model,
            temperature=temperature,
        )

    if backend == "cerebras":
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError(
                "CEREBRAS_API_KEY not found. "
                "Get a free key at cloud.cerebras.ai → API Keys (no credit card needed). "
                "Then: export CEREBRAS_API_KEY=your_key_here"
            )
        return ChatOpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=SecretStr(api_key),
            model=model,
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_BACKEND={backend!r}. Use 'cerebras' or 'ollama'.")


create_chat_client = get_llm_client
