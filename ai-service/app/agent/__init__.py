"""Native tool-calling agent mode (Issue #35 / backlog #30).

Public surface:

* :class:`~app.agent.tools.ToolRegistry` — tool specs + execution with
  stable citation numbering for collected evidence.
* :func:`~app.agent.loop.run_agent` — bounded tool-calling loop.

Implementation choice (recorded in ADR 0009): native function calling on the
existing dependency-free httpx provider abstraction instead of adding
LangChain to the stack.
"""

from app.agent.loop import AgentResult, run_agent
from app.agent.tools import GAME_CATALOG, GAME_SLUGS, ToolRegistry

__all__ = [
    "GAME_CATALOG",
    "GAME_SLUGS",
    "AgentResult",
    "ToolRegistry",
    "run_agent",
]
