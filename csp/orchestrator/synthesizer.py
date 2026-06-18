"""
csp.orchestrator.synthesizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesizer — turns a capability name + goal context into a full
JSON-RPC 2.0 capability spec using the LLM.

This is what makes csp different from MCP:
- MCP: developer pre-defines every tool
- csp: synthesizer generates new capability specs on demand

The output is a SynthesizedCapability whose spec IS the JSON-RPC 2.0
artifact and carries the real Python that the executor runs in a sandbox.

The LLM returns the implementation as a ```python code block plus a ```json
metadata block. Both are parsed, the code is compile-checked, and the
assembled spec is validated before it is stored and run.
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

╔══════════════════════════════════════════════════════════════════════╗
║ GENERALITY CONTRACT — THE MOST IMPORTANT RULE                          ║
╚══════════════════════════════════════════════════════════════════════╝
You are building a REUSABLE capability for an ENTIRE CLASS of tasks — NOT a
one-off answer to the request that triggered this synthesis. The SAME capability
will be invoked again later with DIFFERENT args, and it must work then too.

- Read EVERY specific from `args`: column names, chart kind, group-by keys,
  filters, thresholds, URLs, units, sort order. NEVER hardcode a value taken from
  the current request into the code.
- Design `params_schema` FIRST as the COMPLETE interface for the whole class.
  A charting capability exposes kind, x, y, series, agg, title — not just the one
  chart asked for now. An aggregation capability exposes group_by, metric, agg.
- Provide sensible DEFAULTS via args.get(key, default) so the capability still
  runs if a knob is omitted.
- Litmus test: if asked to "plot average salary by department as a bar chart",
  do NOT write code that only plots salary-by-department bars. Write a general
  charting capability that reads kind/x/y/agg from args and would EQUALLY render
  "age distribution as a histogram" on its next invocation without any change.
- The capability NAME you were given is a general verb (plot_chart,
  aggregate_table). Honor it — implement the whole verb, not the single instance.

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
- You MAY make network/API calls using `requests` or `urllib` when the goal
  requires real external data (weather, stock prices, images, etc.).
- When your code needs an API key or token, read it from os.environ:
      import os
      key = os.environ["SERVICE_API_KEY"]
  and declare it in the ##CREDENTIALS block (see format below).
- Handle missing keys in `args` gracefully.

Respond in EXACTLY this format:

```python
def run(args):
    # real implementation
    return {...}
```

If and ONLY IF the code needs external API credentials, add this block
BETWEEN the python block and the json block:

##CREDENTIALS
ENV_VAR_NAME: Service Name · get at https://example.com/api-keys

One line per credential. Format exactly as shown (ENV_VAR: Service · get at URL).
Omit the ##CREDENTIALS block entirely if no credentials are needed.

```json
{
  "description": "<one sentence>",
  "params_schema": {
    "<param>": {"type": "<string|number|boolean|object|array>", "description": "<...>", "required": true}
  },
  "result_schema": {
    "<field>": {"type": "<type>", "description": "<...>"}
  },
  "tags": ["<3-6 lowercase keywords describing what this capability DOES>"],
  "steps": ["<short progress line 1>", "<short progress line 2>"]
}
```

`tags` are how this capability gets found later: include the function/category
words a future goal might use (e.g. for a group-by table:
["aggregation","group-by","statistics","table"]). Think synonyms, not just the
exact name — this is paid once, here, so selection stays cheap forever.

Putting the code in its own block (not inside JSON) avoids escaping bugs.
"""


_EVOLVE_SYSTEM = """\
You are the CSP capability evolver.
You are given the CURRENT code of a synthesized capability and an instruction
describing how to modify it.

Rules:
- Keep everything that works. Change only what the instruction requires.
- The output must still define `def run(args): ... return {...}`.
- Return the COMPLETE updated code (not a diff — the full function).
- Same output format as synthesis: ```python block then optional ##CREDENTIALS
  block then ```json block.
- Update the description in the json block to reflect the changes.
- If the modification adds a new credential requirement, add a ##CREDENTIALS
  line. If an existing credential is no longer needed, remove it.
- Do NOT add features beyond what the instruction asks for.
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
        spec, credentials = await self._generate_spec(prompt, capability_name)

        # Ensure capability_id matches what the planner asked for
        spec["params"]["capability_id"] = capability_name

        cap = SynthesizedCapability(
            name=capability_name,
            spec=spec,
            description=spec["params"].get("description", ""),
            version=spec["params"].get("version", "1.0.0"),
            credentials=credentials,
        )

        log.info("synthesized capability %r steps=%d", capability_name, len(cap.steps))
        return cap

    async def evolve(
        self,
        cap: SynthesizedCapability,
        instruction: str,
    ) -> SynthesizedCapability:
        """
        Modify an existing synthesized capability in place.

        Passes the current code + a natural-language instruction to the LLM,
        which patches only what's needed and returns the complete updated code.
        The evolved capability keeps the same name but gets a new version stamp.
        """
        prompt = (
            f"Capability name: {cap.name!r}\n"
            f"Current description: {cap.description}\n\n"
            f"Current code:\n```python\n{cap.code}\n```\n\n"
            f"Modification instruction: {instruction}\n\n"
            "Apply the requested change. Return the full updated code."
        )

        spec, credentials = await self._generate_spec(
            prompt, cap.name, system_override=_EVOLVE_SYSTEM
        )
        spec["params"]["capability_id"] = cap.name

        evolved = SynthesizedCapability(
            name=cap.name,
            spec=spec,
            description=spec["params"].get("description", cap.description),
            version=cap.version,
            credentials=credentials,
        )
        log.info("evolved capability %r: %s", cap.name, instruction[:60])
        return evolved

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _generate_spec(
        self,
        prompt: str,
        capability_name: str,
        system_override: Optional[str] = None,
    ) -> tuple[dict[str, Any], list[dict]]:
        """Call LLM and parse JSON, retrying on parse failure."""
        system = system_override or _SYNTHESIS_SYSTEM
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
                    system=system,
                    temperature=0.2,
                    max_tokens=8000,
                )
                spec, credentials = _assemble_spec(response.content, capability_name)
                _validate_spec(spec)
                return spec, credentials

            except (json.JSONDecodeError, ValueError, KeyError, SyntaxError) as exc:
                last_error = exc

        # All retries exhausted — return a minimal fallback spec
        log.error(
            "synthesizer failed for %r after %d attempts, using fallback",
            capability_name, self._max_retries + 1,
        )
        return _fallback_spec(capability_name), []


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


def _assemble_spec(raw: str, capability_name: str) -> tuple[dict[str, Any], list[dict]]:
    """
    Parse the LLM response:
      - ```python ... ```   — the executable code
      - ##CREDENTIALS ...   — optional credential declarations (between blocks)
      - ```json ... ```     — metadata (description, params_schema, steps, ...)

    Returns (spec_dict, credentials_list).
    Keeping code out of the JSON string avoids the escaping bugs that break
    single-blob synthesis.
    """
    code = _extract_block(raw, "python")
    if not code:
        raise ValueError("synthesis response missing ```python code block")

    credentials = _extract_credentials(raw)

    meta_raw = _extract_block(raw, "json")
    if not meta_raw:
        raise ValueError("synthesis response missing ```json metadata block")
    meta = json.loads(meta_raw)

    spec = {
        "jsonrpc": "2.0",
        "method": "csp.capability.invoke",
        "params": {
            "capability_id": capability_name,
            "version": "1.0.0",
            "description": meta.get("description", ""),
            "kind": "synthesized",
            # Semantic tags the LLM attached — consumed by SelectionStrategy to
            # shortlist this capability for future goals (see selection.py).
            "tags": meta.get("tags", []),
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
    return spec, credentials


def _extract_credentials(raw: str) -> list[dict]:
    """
    Parse an optional ##CREDENTIALS block from the LLM response.

    Expected format (one line per credential):
        ##CREDENTIALS
        ENV_VAR_NAME: Service Name · get at https://example.com
    """
    creds: list[dict] = []
    in_block = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "##CREDENTIALS":
            in_block = True
            continue
        if in_block:
            if stripped.startswith("```") or stripped.startswith("##"):
                break
            if not stripped or stripped.startswith("#"):
                continue
            # ENV_VAR_NAME: Service · get at URL
            if ":" in stripped:
                env_key, rest = stripped.split(":", 1)
                env_key = env_key.strip()
                rest = rest.strip()
                # parse "Service · get at URL"
                service = rest
                get_it_at = ""
                if "get at" in rest.lower():
                    parts = rest.lower().split("get at", 1)
                    service = rest[: len(parts[0])].rstrip("·· ").strip()
                    get_it_at = rest[len(parts[0]) + len("get at"):].strip()
                    # Restore original case from raw rest
                    idx = rest.lower().find("get at")
                    get_it_at = rest[idx + len("get at"):].strip()
                creds.append({
                    "env_key":    env_key,
                    "service":    service,
                    "get_it_at":  get_it_at,
                    "description": f"{service} API credential",
                })
    return creds


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