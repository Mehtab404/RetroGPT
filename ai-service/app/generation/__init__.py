"""Generation package: LLM abstraction and prompt templates."""

from app.generation.llm import (
    GeminiLLM,
    GroqLLM,
    LLMError,
    LLMProvider,
    OpenAILLM,
    get_llm_provider,
)
from app.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

__all__ = [
    # LLM providers
    "LLMProvider",
    "LLMError",
    "GeminiLLM",
    "OpenAILLM",
    "GroqLLM",
    "get_llm_provider",
    # Prompts
    "RAG_SYSTEM_PROMPT",
    "build_rag_prompt",
]
