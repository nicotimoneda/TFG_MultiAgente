"""Factory for creating Cerebras chat clients used across all graph configurations."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_llm_client(
    model_name: str = "qwen-3-235b-a22b-instruct-2507",
    temperature: float = 0.2,
) -> BaseChatModel:
    """Create a Cerebras chat client for the given model.

    Uses the Cerebras Inference API (OpenAI-compatible) with the key from
    ``CEREBRAS_API_KEY`` in the environment. All graph configurations call this
    factory so the backend can be swapped in a single place.

    Args:
        model_name: Cerebras model ID, e.g. ``'qwen-3-235b-a22b-instruct-2507'``.
        temperature: Sampling temperature. Defaults to 0.2 for deterministic
            outputs; use 0.4 for the reflective developer pass.

    Returns:
        A LangChain ``BaseChatModel`` compatible with ``.invoke(messages)``.

    Raises:
        ValueError: If ``CEREBRAS_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError(
            "CEREBRAS_API_KEY not found. "
            "Get a free key at cloud.cerebras.ai → API Keys (no credit card needed). "
            "Then: export CEREBRAS_API_KEY=your_key_here"
        )
    return ChatOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )


# Alias for backwards compatibility with graph modules
create_chat_client = get_llm_client
