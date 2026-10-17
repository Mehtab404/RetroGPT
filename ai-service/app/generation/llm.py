"""LLM provider abstraction for the RetroGPT generation pipeline.

Provides a uniform :class:`LLMProvider` interface for generating text from
a prompt, with concrete implementations for:

* :class:`GeminiLLM`  — Google Generative AI (``gemini-1.5-flash`` default)
* :class:`OpenAILLM`  — OpenAI chat completions (``gpt-4o-mini`` default)

Both implementations support:

* **Retry with exponential backoff** — transient HTTP errors (429, 5xx) and
  network timeouts are retried up to *max_retries* times.
* **Configurable temperature / max tokens** — sensible defaults for RAG.

Typical usage::

    from app.generation.llm import get_llm_provider

    provider = get_llm_provider()          # reads provider from settings
    answer = await provider.generate("Your prompt here")

All provider methods are ``async`` and use ``httpx.AsyncClient`` internally.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT: float = 60.0
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# Gemini REST API base URL
_GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
# OpenAI REST API base URL
_OPENAI_BASE_URL: str = "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when an LLM API call fails permanently after all retries."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement :meth:`generate`.

    Args:
        model:        Name of the model to use.
        temperature:  Sampling temperature (0.0 = deterministic).
        max_tokens:   Maximum number of tokens to generate.
        max_retries:  Number of retry attempts for transient errors.
        base_delay:   Initial retry delay in seconds (doubled each attempt).
        jitter:       If ``True``, add random jitter to retry delays.
        timeout:      HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.jitter = jitter
        self.timeout = timeout

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response for *prompt*.

        Args:
            prompt: The full prompt string (including any system/context text).

        Returns:
            The generated text as a plain string.

        Raises:
            LLMError: If the API returns a non-retryable error after all
                      retry attempts are exhausted.
        """

    # ------------------------------------------------------------------
    # Shared retry helper
    # ------------------------------------------------------------------

    async def _with_retry(self, coro_fn, *args: Any, **kwargs: Any) -> Any:
        """Execute *coro_fn* with exponential-backoff retry.

        Args:
            coro_fn: Async callable that may raise :class:`httpx.HTTPStatusError`
                     or :class:`httpx.RequestError`.
            *args:   Positional arguments forwarded to *coro_fn*.
            **kwargs: Keyword arguments forwarded to *coro_fn*.

        Returns:
            The return value of *coro_fn* on success.

        Raises:
            LLMError: After all retry attempts are exhausted.
        """
        delay = self.base_delay
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise LLMError(
                        f"LLM API error {exc.response.status_code}: " f"{exc.response.text[:200]}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "LLM API returned %d on attempt %d/%d; retrying…",
                    exc.response.status_code,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "LLM request error on attempt %d/%d: %s; retrying…",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

            if attempt < self.max_retries:
                wait = delay + (random.uniform(0, 0.5) if self.jitter else 0)
                await asyncio.sleep(wait)
                delay *= 2

        raise LLMError(
            f"LLM API failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


class GeminiLLM(LLMProvider):
    """Google Gemini text generation via REST API.

    Uses the ``generateContent`` endpoint with ``text/plain`` parts.

    Args:
        model:        Gemini model name. Defaults to ``settings.llm_model``
                      (``"gemini-1.5-flash"``).
        api_key:      Google API key. Defaults to ``settings.google_api_key``.
        temperature:  Sampling temperature.
        max_tokens:   Maximum output tokens.
        max_retries:  Retry attempts for transient errors.
        base_delay:   Base retry delay in seconds.
        jitter:       Whether to add jitter to retry delays.
        timeout:      HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(
            model=model or settings.llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_delay=base_delay,
            jitter=jitter,
            timeout=timeout,
        )
        self._api_key = api_key or settings.google_api_key

    async def generate(self, prompt: str) -> str:
        """Generate text using the Gemini REST API.

        Args:
            prompt: Full prompt string.

        Returns:
            Generated text.

        Raises:
            LLMError: On API errors after retries.
        """
        return await self._with_retry(self._call_api, prompt)

    async def _call_api(self, prompt: str) -> str:
        """Make a single Gemini API call (no retry logic here)."""
        url = f"{_GEMINI_BASE_URL}/models/{self.model}:generateContent" f"?key={self._api_key}"
        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response structure: {data}") from exc


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAILLM(LLMProvider):
    """OpenAI chat completion via REST API.

    Uses the ``/chat/completions`` endpoint with a single user message.

    Args:
        model:        OpenAI model name. Defaults to ``"gpt-4o-mini"``.
        api_key:      OpenAI API key. Defaults to ``settings.openai_api_key``.
        temperature:  Sampling temperature.
        max_tokens:   Maximum output tokens.
        max_retries:  Retry attempts for transient errors.
        base_delay:   Base retry delay in seconds.
        jitter:       Whether to add jitter to retry delays.
        timeout:      HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_delay=base_delay,
            jitter=jitter,
            timeout=timeout,
        )
        self._api_key = api_key or settings.openai_api_key

    async def generate(self, prompt: str) -> str:
        """Generate text using the OpenAI chat completions API.

        Args:
            prompt: Full prompt string (sent as a user message).

        Returns:
            Generated text.

        Raises:
            LLMError: On API errors after retries.
        """
        return await self._with_retry(self._call_api, prompt)

    async def _call_api(self, prompt: str) -> str:
        """Make a single OpenAI API call (no retry logic here)."""
        url = f"{_OPENAI_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected OpenAI response structure: {data}") from exc


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_llm_provider(
    provider: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Return an :class:`LLMProvider` instance based on configuration.

    Args:
        provider: ``"gemini"`` or ``"openai"``. Defaults to
                  ``settings.llm_provider``.
        **kwargs: Additional keyword arguments forwarded to the provider
                  constructor (e.g. ``temperature``, ``max_tokens``).

    Returns:
        A configured :class:`LLMProvider` instance.

    Raises:
        ValueError: If *provider* is not ``"gemini"`` or ``"openai"``.
    """
    chosen = provider or settings.llm_provider
    if chosen == "gemini":
        return GeminiLLM(**kwargs)
    if chosen == "openai":
        return OpenAILLM(**kwargs)
    raise ValueError(f"Unknown LLM provider {chosen!r}. Expected 'gemini' or 'openai'.")
