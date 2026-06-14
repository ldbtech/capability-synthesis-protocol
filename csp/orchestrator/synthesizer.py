"""
csp.orchestrator.synthesizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesizer — turns a capability name + goal context into a full
JSON-RPC 2.0 capability spec using the LLM.

This is what makes csp different from MCP:
- MCP: developer pre-defines every tool
- csp: synthesizer generates new capability specs on demand

The output is a SynthesizedCapability whose spec IS the JSON-RPC 2.0
artifact. For MVP it is mock-executed. Future runtimes (Docker, K8s,
Terraform) will read this spec and know exactly how to run it.

The LLM is prompted to return ONLY valid JSON — no prose, no markdown.
Response is parsed, validated against required fields, and stored.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from ..llm.base import BaseLLM, LLMMessage
from .capability import SynthesizedCapability

log = logging.getLogger("csp.synthesizer")

# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """\
You are the CSP capability synthesizer.
You generate a capability as REAL, RUNNABLE Python plus a small metadata block.

The code you write is executed in a sandboxed subprocess. It MUST define
exactly one entrypoint:

    def run(args: dict) -> dict:
        ...
        return {...}   # a JSON-serializable dict

Rules for the code:
- Define `def run(args):` taking a single dict of parameters.
- Read inputs from the args keys described in the context you are given.
- Return a JSON-serializable dict (no numpy/pandas objects — convert to
  plain float/int/list/dict before returning).
- You may use the Python standard library freely (math, re, statistics,
  datetime, csv, io, collections, etc.).
- You may use pandas / numpy if helpful, but convert results to plain types.
- Do NOT read files, make network calls, or use input(). Work only from `args`.
- Keep it self-contained and deterministic. Handle missing keys gracefully.

Respond in EXACTLY this format — a python code block, then a json block.
Nothing before, between (except a newline), or after them.

```python
def run(args):
    # real implementation
    return {...}
```
```json
{
  "description": "<one sentence>",
  "params_schema": {
    "<param>": {"type": "<string|number|boolean|object|array>", "description": "<...>", "required": true}
  },
  "result_schema": {
    "<field>": {"type": "<type>", "description": "<...>"}
  },
  "steps": ["<short progress line 1>", "<short progress line 2>"]
}
```

Putting the code in its own block (not inside JSON) avoids escaping bugs.
"""


class Synthesizer:
    """
    Generates SynthesizedCapability objects on demand.

    Parameters
    ----------
    llm:
        LLM provider for spec generation.
    max_retries:
        How many times to retry if LLM returns invalid JSON.
    """

    __slots__ = ("_llm", "_max_retries", "_guidance")

    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_retries: int = 2,
        guidance: str = "",
    ) -> None:
        self._llm         = llm
        self._max_retries = max_retries
        # App-supplied, domain-specific guidance injected into every synthesis
        # prompt. The library stays generic; the developer describes their
        # domain's conventions (data shapes, output formats, etc.) here.
        self._guidance    = guidance

    async def synthesize(
        self,
        capability_name: str,
        goal: str,
        context: Optional[str] = None,
    ) -> SynthesizedCapability:
        """
        Generate a JSON-RPC 2.0 capability spec for the given name and goal.

        Parameters
        ----------
        capability_name:
            The name the planner asked for (e.g. "train_churn_model").
        goal:
            The original user goal — gives the LLM context.
        context:
            Optional extra context (available resources, existing caps).
        """
        prompt = _build_prompt(capability_name, goal, context, self._guidance)
        spec   = await self._generate_spec(prompt, capability_name)

        # Ensure capability_id matches what the planner asked for
        spec["params"]["capability_id"] = capability_name

        cap = SynthesizedCapability(
            name=capability_name,
            spec=spec,
            description=spec["params"].get("description", ""),
            version=spec["params"].get("version", "1.0.0"),
        )

        log.info("synthesized capability %r steps=%d", capability_name, len(cap.steps))
        return cap

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _generate_spec(
        self,
        prompt: str,
        capability_name: str,
    ) -> dict[str, Any]:
        """Call LLM and parse JSON, retrying on parse failure."""
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                log.warning(
                    "synthesizer retry %d/%d for %r: %s",
                    attempt, self._max_retries, capability_name, last_error,
                )

            try:
                response = await self._llm.complete(
                    [LLMMessage(role="user", content=prompt)],
                    system=_SYNTHESIS_SYSTEM,
                    temperature=0.2,    # slight creativity for step descriptions
                    max_tokens=1500,
                )
                spec = _assemble_spec(response.content, capability_name)
                _validate_spec(spec)
                return spec

            except (json.JSONDecodeError, ValueError, KeyError, SyntaxError) as exc:
                last_error = exc

        # All retries exhausted — return a minimal fallback spec
        log.error(
            "synthesizer failed for %r after %d attempts, using fallback",
            capability_name, self._max_retries + 1,
        )
        return _fallback_spec(capability_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(
    capability_name: str,
    goal: str,
    context: Optional[str],
    guidance: str = "",
) -> str:
    parts = [
        f"Synthesize a capability named: {capability_name!r}",
        f"User goal: {goal}",
    ]
    if context:
        parts.append(f"Available context:\n{context}")
    if guidance:
        parts.append(f"App-specific conventions you must follow:\n{guidance}")
    parts.append(
        "Generate the capability. Make the steps concrete and specific to the goal."
    )
    return "\n\n".join(parts)


def _assemble_spec(raw: str, capability_name: str) -> dict[str, Any]:
    """
    Parse the two-block response (```python ... ``` and ```json ... ```) and
    assemble a full JSON-RPC 2.0 capability spec. Keeping code out of the JSON
    string avoids the escaping bugs that break single-blob synthesis.
    """
    code = _extract_block(raw, "python")
    if not code:
        raise ValueError("synthesis response missing ```python code block")

    meta_raw = _extract_block(raw, "json")
    if not meta_raw:
        raise ValueError("synthesis response missing ```json metadata block")
    meta = json.loads(meta_raw)

    return {
        "jsonrpc": "2.0",
        "method": "csp.capability.invoke",
        "params": {
            "capability_id": capability_name,
            "version": "1.0.0",
            "description": meta.get("description", ""),
            "kind": "synthesized",
            "params_schema": meta.get("params_schema", {}),
            "result_schema": meta.get("result_schema", {}),
            "execution": {
                "target": "python",
                "transport": "stdio",
                "entrypoint": "run",
                "code": code,
                "steps": meta.get("steps", [f"executing {capability_name}"]),
            },
        },
    }


def _extract_block(raw: str, lang: str) -> str:
    """Extract the contents of a ```<lang> ... ``` fenced block."""
    fence = f"```{lang}"
    start = raw.find(fence)
    if start == -1:
        return ""
    start += len(fence)
    end = raw.find("```", start)
    if end == -1:
        return ""
    return raw[start:end].strip()


def _validate_spec(spec: dict[str, Any]) -> None:
    """Raise ValueError if required fields are missing."""
    if spec.get("jsonrpc") != "2.0":
        raise ValueError("spec missing jsonrpc: 2.0")
    if spec.get("method") != "csp.capability.invoke":
        raise ValueError("spec missing correct method")
    params = spec.get("params", {})
    for required in ("capability_id", "execution"):
        if required not in params:
            raise ValueError(f"spec missing params.{required}")

    execution = params["execution"]
    if execution.get("target") == "python":
        code = execution.get("code", "")
        if "def run" not in code:
            raise ValueError("python execution must define a run(args) entrypoint")
        # Reject code that obviously won't compile before we ever run it
        compile(code, f"<{params['capability_id']}>", "exec")


def _fallback_spec(name: str) -> dict[str, Any]:
    """Minimal valid spec used when LLM synthesis fails — echoes its args."""
    code = (
        "def run(args):\n"
        "    return {\n"
        f"        'capability': {name!r},\n"
        "        'note': 'fallback implementation — synthesis failed',\n"
        "        'received_args': args,\n"
        "    }\n"
    )
    return {
        "jsonrpc": "2.0",
        "method": "csp.capability.invoke",
        "params": {
            "capability_id": name,
            "version": "1.0.0",
            "description": f"Auto-synthesized capability: {name}",
            "kind": "synthesized",
            "params_schema": {},
            "result_schema": {"result": {"type": "object"}},
            "execution": {
                "target": "python",
                "transport": "stdio",
                "entrypoint": "run",
                "code": code,
                "steps": [f"executing {name}"],
            },
        },
    }