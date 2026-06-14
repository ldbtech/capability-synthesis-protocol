"""
csp.llm.base
~~~~~~~~~~~~~~
Abstract LLM interface. Both planner and synthesizer depend on this
abstraction — never on a concrete provider. Developers pass their
preferred implementation into Orchestrator.

Design:
- Single async complete() method — everything goes through it
- Messages follow the same role/content shape as Anthropic + OpenAI
  so implementations are trivial to write
- LLMResponse is a simple frozen dataclass — no provider leakage
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """A single message in a conversation."""
    role:    str   # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Response from any LLM provider."""
    content:       str
    input_tokens:  int            = 0
    output_tokens: int            = 0
    stop_reason:   Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"<LLMResponse tokens={self.input_tokens}+{self.output_tokens} "
            f"stop={self.stop_reason!r} content={self.content!r:.80}>"
        )


class BaseLLM(ABC):
    """
    Abstract base for all LLM providers.

    Implement complete() — everything else in CSP uses this interface.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM and return the response.

        Parameters
        ----------
        messages:
            Conversation history in role/content pairs.
        max_tokens:
            Maximum tokens to generate.
        temperature:
            0.0 for deterministic (planning/synthesis), higher for creative.
        system:
            System prompt override. If None, use the provider default.
        """
        ...

    async def complete_once(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Convenience: single user message, no history."""
        return await self.complete(
            [LLMMessage(role="user", content=prompt)],
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )