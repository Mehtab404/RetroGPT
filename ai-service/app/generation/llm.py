"""LLM provider abstraction for the RetroGPT generation pipeline.

Provides a uniform :class:`LLMProvider` interface for generating text from
a prompt, with concrete implementations for:

* :class:`GeminiLLM`  — Google Generative AI (``gemini-1.5-flash`` default)
* :class:`OpenAILLM`  — OpenAI chat completions (``gpt-4o-mini`` default)

Both implementations support:

* **Retry with exponential backoff** — transient HTTP errors (429, 5xx) and
  network timeouts are retried up to *max_retries* times.
* **Configurable temperature / max tokens** — sensible defaults for RAG.
* **Native tool calling** — :meth:`LLMProvider.generate_with_tools` maps a
  provider-neutral transcript to Gemini ``functionDeclarations`` / OpenAI
  function ``tools`` (Issue #35, backlog #30).

Typical usage::

    from app.generation.llm import get_llm_provider

    provider = get_llm_provider()          # reads provider from settings
    answer = await provider.generate("Your prompt here")

All provider methods are ``async`` and use ``httpx.AsyncClient`` internally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
# Tool-calling contracts (Issue #35 / backlog #30)
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool invocation requested by the model.

    Attributes:
        name:      Registered tool name.
        arguments: Parsed JSON arguments for the tool.
        call_id:   Provider-assigned call identifier (``None`` for Gemini,
                   which matches calls to results by name and order).
    """

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class AgentTurn:
    """One model turn in a tool-calling conversation.

    Attributes:
        text:       Optional assistant text produced alongside tool calls.
        tool_calls: Zero or more tool invocations requested by the model.
    """

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


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

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream the response for *prompt* chunk by chunk (Issue #25).

        The default implementation falls back to the non-streaming
        :meth:`generate` and yields the complete answer as a single chunk;
        providers with native streaming support override this method.

        Args:
            prompt: The full prompt string.

        Yields:
            Successive text chunks of the generated answer.

        Raises:
            LLMError: If generation fails.
        """
        yield await self.generate(prompt)

    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """Run one tool-calling turn of an agent conversation (Issue #35).

        *messages* uses a provider-neutral transcript format:

        * ``{"role": "system", "content": str}``
        * ``{"role": "user", "content": str}``
        * ``{"role": "assistant", "content": str | None,
          "tool_calls": [{"name": str, "arguments": dict, "call_id": str | None}]}``
        * ``{"role": "tool", "name": str, "tool_call_id": str | None,
          "content": str}`` — *content* is the JSON-serialised tool payload.

        *tools* are JSON-schema specifications: ``{"name", "description",
        "parameters"}``.  The default implementation raises :class:`LLMError`;
        providers with native function-calling support override it.

        Args:
            messages: Provider-neutral transcript (see above).
            tools:    Tool specifications the model may call.

        Returns:
            An :class:`AgentTurn` with optional text and tool calls.

        Raises:
            LLMError: When the provider does not support tool calling, or the
                API call fails after all retries.
        """
        raise LLMError(f"{type(self).__name__} does not support tool calling")

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

    async def _with_stream_retry(
        self, coro_fn: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Execute streaming *coro_fn* with retry until the first chunk.

        Retrying a stream that has already yielded text would duplicate
        tokens, so transient failures are only retried while nothing has
        been yielded yet.

        Args:
            coro_fn: Async generator function yielding text chunks.
            *args:   Positional arguments forwarded to *coro_fn*.
            **kwargs: Keyword arguments forwarded to *coro_fn*.

        Yields:
            Text chunks produced by *coro_fn*.

        Raises:
            LLMError: On a non-retryable error, on a failure after streaming
                has started, or after all retry attempts are exhausted.
        """
        delay = self.base_delay
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            yielded = False
            try:
                async for chunk in coro_fn(*args, **kwargs):
                    yielded = True
                    yield chunk
                return
            except httpx.HTTPStatusError as exc:
                if yielded:
                    raise LLMError(
                        f"LLM stream interrupted after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                if exc.response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise LLMError(
                        f"LLM API error {exc.response.status_code}: " f"{exc.response.text[:200]}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "LLM API returned %d on streaming attempt %d/%d; retrying…",
                    exc.response.status_code,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except httpx.RequestError as exc:
                if yielded:
                    raise LLMError(
                        f"LLM stream interrupted after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "LLM streaming request error on attempt %d/%d: %s; retrying…",
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
# SSE helpers (shared by streaming providers)
# ---------------------------------------------------------------------------


async def _iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the payload of each ``data:`` line in an SSE response."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data:
            yield data


# ---------------------------------------------------------------------------
# Tool-calling translation helpers (shared by providers)
# ---------------------------------------------------------------------------


def _tool_content_to_response(content: str) -> dict[str, Any]:
    """Parse a tool-result JSON string into an object (Gemini requires one)."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"result": content}
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def _gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the neutral transcript into Gemini ``contents`` entries.

    Consecutive same-role entries are merged because the REST API expects
    alternating ``user``/``model`` roles — multiple function responses for one
    model turn become several parts of a single ``user`` entry.
    """
    contents: list[dict[str, Any]] = []

    def append(role: str, parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for message in messages:
        role = message.get("role")
        if role == "user":
            append("user", [{"text": str(message.get("content", ""))}])
        elif role == "assistant":
            parts: list[dict[str, Any]] = []
            if message.get("content"):
                parts.append({"text": str(message["content"])})
            for call in message.get("tool_calls") or []:
                parts.append(
                    {
                        "functionCall": {
                            "name": str(call.get("name", "")),
                            "args": call.get("arguments") or {},
                        }
                    }
                )
            append("model", parts)
        elif role == "tool":
            append(
                "user",
                [
                    {
                        "functionResponse": {
                            "name": str(message.get("name", "")),
                            "response": _tool_content_to_response(str(message.get("content", ""))),
                        }
                    }
                ],
            )
    return contents


def _safe_json_args(raw: Any) -> dict[str, Any]:
    """Decode JSON-encoded tool arguments defensively."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the neutral transcript into OpenAI ``messages`` entries."""
    converted: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in ("system", "user"):
            converted.append({"role": role, "content": str(message.get("content", ""))})
        elif role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
            }
            calls = message.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.get("call_id") or f"call_{index + 1}_{pos + 1}",
                        "type": "function",
                        "function": {
                            "name": str(call.get("name", "")),
                            "arguments": json.dumps(call.get("arguments") or {}),
                        },
                    }
                    for pos, call in enumerate(calls)
                ]
            if entry["content"] is None and "tool_calls" not in entry:
                continue
            converted.append(entry)
        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id") or "",
                    "content": str(message.get("content", "")),
                }
            )
    return converted


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

    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """Run one Gemini ``generateContent`` tool-calling turn (Issue #35).

        Sends the neutral transcript as ``contents`` plus
        ``tools[0].functionDeclarations``; the first candidate's parts are
        parsed into text and ``functionCall`` invocations.
        """
        turn: AgentTurn = await self._with_retry(self._call_with_tools_api, messages, tools)
        return turn

    async def _call_with_tools_api(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """Make a single Gemini tool-calling call (no retry logic here)."""
        url = f"{_GEMINI_BASE_URL}/models/{self.model}:generateContent" f"?key={self._api_key}"
        payload: dict[str, Any] = {
            "contents": _gemini_contents(messages),
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get(
                                "parameters",
                                {"type": "object", "properties": {}},
                            ),
                        }
                        for tool in tools
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        system_text = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response structure: {data}") from exc

        text = "".join(str(part["text"]) for part in parts if "text" in part)
        tool_calls = [
            ToolCall(
                name=str(part["functionCall"].get("name", "")),
                arguments=_safe_json_args(part["functionCall"].get("args")),
            )
            for part in parts
            if "functionCall" in part
        ]
        return AgentTurn(text=text or None, tool_calls=tool_calls)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream text from Gemini's ``streamGenerateContent`` endpoint.

        Args:
            prompt: Full prompt string.

        Yields:
            Text chunks as they are produced.

        Raises:
            LLMError: On API errors (retried before the first chunk).
        """
        async for chunk in self._with_stream_retry(self._stream_api, prompt):
            yield chunk

    async def _stream_api(self, prompt: str) -> AsyncIterator[str]:
        """Make a single streaming Gemini API call (no retry logic here)."""
        url = (
            f"{_GEMINI_BASE_URL}/models/{self.model}:streamGenerateContent"
            f"?alt=sse&key={self._api_key}"
        )
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
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for raw in _iter_sse_data(response):
                    try:
                        data = json.loads(raw)
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        # Keep-alives and finish-reason-only frames carry no text.
                        continue
                    if text:
                        yield text


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

    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """Run one OpenAI chat-completions tool-calling turn (Issue #35).

        Sends the neutral transcript as ``messages`` plus function ``tools``;
        ``tool_calls`` in the assistant message are parsed into invocations.
        """
        turn: AgentTurn = await self._with_retry(self._call_with_tools_api, messages, tools)
        return turn

    async def _call_with_tools_api(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """Make a single OpenAI tool-calling call (no retry logic here)."""
        url = f"{_OPENAI_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _openai_messages(messages),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        ),
                    },
                }
                for tool in tools
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response structure: {data}") from exc

        content = message.get("content")
        tool_calls = [
            ToolCall(
                name=str((raw.get("function") or {}).get("name", "")),
                arguments=_safe_json_args((raw.get("function") or {}).get("arguments")),
                call_id=raw.get("id"),
            )
            for raw in message.get("tool_calls") or []
        ]
        return AgentTurn(text=str(content) if content else None, tool_calls=tool_calls)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream chat-completion deltas from OpenAI.

        Args:
            prompt: Full prompt string (sent as a user message).

        Yields:
            Content deltas as they are produced.

        Raises:
            LLMError: On API errors (retried before the first chunk).
        """
        async for chunk in self._with_stream_retry(self._stream_api, prompt):
            yield chunk

    async def _stream_api(self, prompt: str) -> AsyncIterator[str]:
        """Make a single streaming OpenAI API call (no retry logic here)."""
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
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for raw in _iter_sse_data(response):
                    if raw == "[DONE]":
                        return
                    try:
                        data = json.loads(raw)
                        text = data["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                        continue
                    if text:
                        yield text


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
