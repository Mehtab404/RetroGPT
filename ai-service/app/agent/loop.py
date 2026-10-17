"""Bounded tool-calling loop for ``POST /agent`` (Issue #35 / backlog #30).

The loop is provider-agnostic: it drives
:meth:`LLMProvider.generate_with_tools` with the neutral transcript from
:mod:`app.generation.llm`, executes requested tools through the
:class:`~app.agent.tools.ToolRegistry`, and appends results back into the
transcript until the model answers or the step budget runs out.

When the budget is exhausted the loop closes with a deterministic grounded
answer built from the collected evidence (standard RAG prompt), so a
tool-happy model can never loop unboundedly or answer ungrounded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent.tools import ToolRegistry
from app.generation.llm import AgentTurn, LLMError, LLMProvider
from app.generation.prompts import AGENT_SYSTEM_PROMPT, build_rag_prompt
from app.retrieval.vector_store import SearchResult

#: Exact refusal used by the standard RAG path when no evidence exists.
NO_CONTEXT_ANSWER = "I don't know — this information is not available in the provided sources."


@dataclass
class AgentResult:
    """Outcome of one agent run.

    Attributes:
        answer:     Final answer text (with ``[N]`` citations when grounded).
        sources:    Evidence passages collected by tools (first-seen order).
        tools_used: Tool invocations in call order (first occurrence).
        steps:      Number of model turns taken.
    """

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    tools_used: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0


async def run_agent(
    llm: LLMProvider,
    question: str,
    registry: ToolRegistry,
    max_steps: int = 3,
) -> AgentResult:
    """Run the bounded tool-calling loop and return the final answer.

    Args:
        llm:       Provider implementing :meth:`generate_with_tools`.
        question:  The user's question.
        registry:  Tool registry bound to the service's retrieval callable.
        max_steps: Maximum model turns (the loop never exceeds this).

    Returns:
        An :class:`AgentResult`; evidence and tool calls come from the
        registry, so they reflect exactly what the model requested.

    Raises:
        LLMError: If the provider fails after retries or returns an empty
            turn (neither text nor tool calls).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        turn = await llm.generate_with_tools(messages, registry.specs())
        if not turn.tool_calls:
            if turn.text is None:
                raise LLMError("Model returned neither text nor tool calls")
            return AgentResult(
                answer=turn.text,
                sources=registry.evidence,
                tools_used=list(registry.calls),
                steps=step,
            )

        messages.append(_assistant_message(turn))
        for position, call in enumerate(turn.tool_calls, start=1):
            payload = await registry.execute(call.name, call.arguments)
            messages.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "tool_call_id": call.call_id or f"call_{step}_{position}",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

    # Step budget exhausted while the model kept requesting tools: close with
    # a deterministic, grounded answer so the run still terminates usefully.
    evidence = registry.evidence
    if not evidence:
        return AgentResult(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            tools_used=list(registry.calls),
            steps=max_steps,
        )
    prompt = build_rag_prompt(
        question=question,
        context_passages=[
            {"id": result.chunk_id, "content": result.content} for result in evidence
        ],
    )
    answer = await llm.generate(prompt)
    return AgentResult(
        answer=answer,
        sources=evidence,
        tools_used=list(registry.calls),
        steps=max_steps,
    )


def _assistant_message(turn: AgentTurn) -> dict[str, Any]:
    """Render a model turn back into the neutral transcript format."""
    return {
        "role": "assistant",
        "content": turn.text,
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments, "call_id": call.call_id}
            for call in turn.tool_calls
        ],
    }
