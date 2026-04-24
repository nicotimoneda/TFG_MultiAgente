"""Factory for creating HuggingFace chat clients used across all graph configurations."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def create_chat_client(model_name: str, temperature: float = 0.2) -> BaseChatModel:
    """Create a HuggingFace chat client for the given model.

    Uses the HuggingFace Inference API (serverless) with the token from
    ``HF_TOKEN`` in the environment. All graph configurations call this
    factory so the backend can be swapped in a single place.

    Args:
        model_name: HuggingFace repo ID, e.g.
            ``'meta-llama/Llama-3.1-70B-Instruct'``.
        temperature: Sampling temperature. Defaults to 0.2 for deterministic
            outputs; use 0.4 for the reflective developer pass.

    Returns:
        A LangChain ``BaseChatModel`` compatible with ``.invoke(messages)``.

    Raises:
        KeyError: If ``HF_TOKEN`` is not set in the environment.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        temperature=temperature,
        max_new_tokens=2048,
    )
    return ChatHuggingFace(llm=endpoint)
