"""
csp.llm.anthropic
~~~~~~~~~~~~~~~~~~~
Anthropic Claude implementation of BaseLLM.

Usage:
    from csp.llm import AnthropicLLM
    llm = AnthropicLLM(api_key="sk-ant-...")
    app = Orchestrator("my-app", llm=llm)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import BaseLLM, LLMMessage, LLMResponse

log = logging.getLogger("csp.llm.anthropic")

# Default model — fast and capable enough for planning + synthesis
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude LLM provider.

    Parameters
    ----------
    api_key:
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model:
        Model string. Defaults to claude-3-5-haiku for speed.
    default_system:
        Default system prompt injected on every call unless overridden.
    """

    __slots__ = ("_api_key", "_model", "_default_system", "_client")

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: Optional[str] = None,
        default_system: Optional[str] = None,
    ) -> None:
        self._api_key        = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model          = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        self._default_system = default_system
        self._client         = None   # lazy init — avoids import cost at startup

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                )
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Convert to Anthropic message format
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"   # system goes in the system param
        ]

        system_prompt = system or self._default_system

        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
            temperature=temperature,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        log.debug("llm call model=%s messages=%d", self._model, len(anthropic_messages))

        response = await client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )