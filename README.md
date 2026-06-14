# CSP — Capability Synthesis Protocol

CSP is a Python library for building AI orchestrators that **plan**, **execute**, and **synthesize** capabilities at runtime using LLMs.

Inspired by the MCP (Model Context Protocol) wire format — you register Python functions as capabilities, submit natural-language goals, and CSP figures out which capabilities to run and in what order. If a capability doesn't exist yet, CSP synthesizes a JSON-RPC 2.0 spec for it on the fly using an LLM.

---

## Install

```bash
pip install csp-sdk
```

Or install from source:

```bash
git clone https://github.com/ldbtech/csp
cd csp
pip install -e .
```

---

## Quickstart

**server.py** — define your capabilities:

```python
from csp import Orchestrator, ElicitRequired, AnthropicLLM

app = Orchestrator(
    "my-server",
    llm=AnthropicLLM(),          # reads ANTHROPIC_API_KEY + ANTHROPIC_MODEL from env
    # or pass inline:
    # llm=AnthropicLLM(api_key="sk-ant-...", model="claude-sonnet-4-6"),
)

@app.capability("greet")
async def greet(name: str, language: str = "english") -> dict:
    """Greet a person in their preferred language."""
    greetings = {"english": "Hello", "spanish": "Hola", "japanese": "こんにちは"}
    return {"message": f"{greetings.get(language, 'Hello')}, {name}!"}

@app.capability("send_report")
async def send_report(recipient: str) -> dict:
    """Send a report. Requires approval first."""
    raise ElicitRequired(
        kind="approval",
        question=f"Send report to {recipient}?",
    )

if __name__ == "__main__":
    app.run()   # listens on stdin, writes to stdout — identical to MCP
```

**Run it:**

```bash
ANTHROPIC_API_KEY=sk-ant-... python server.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `ANTHROPIC_MODEL` | No | Claude model to use (default: `claude-haiku-4-5-20251001`) |

Available models:

```
claude-haiku-4-5-20251001   # fast, cheapest — default
claude-sonnet-4-6           # balanced speed + quality
claude-opus-4-8             # most capable
```

---

## How it works

```
Developer submits goal: "greet Alice in Spanish"
        ↓
Planner (LLM) → checks registry → found: greet
        ↓
Executor → calls greet(name="Alice", language="spanish")
        ↓
LLM summarizes result → streams back to client
```

If the capability is **not registered**, the Synthesizer generates a JSON-RPC 2.0 spec via LLM and mock-executes it. The spec is stored in the registry for reuse.

---

## Wire protocol

CSP uses **JSON-RPC 2.0 over stdio** (NDJSON, one message per line) — the same transport as MCP. Each message is a JSON object terminated by `\n`.

Key methods:

| Method | Direction | Description |
|---|---|---|
| `initialize` | client → server | Handshake |
| `csp.goal.submit` | client → server | Submit a natural-language goal |
| `csp.capability.list` | client → server | List registered capabilities |
| `csp.stream.event` | server → client | Streaming progress event |
| `csp.elicit.request` | server → client | Human-in-the-loop pause |
| `csp.elicit.respond` | client → server | Answer an elicitation |
| `csp.result` | server → client | Terminal result |

---

## Project structure

```
csp/
├── csp/
│   ├── __init__.py          # public API: Orchestrator, ElicitRequired, AnthropicLLM
│   ├── llm/
│   │   ├── base.py          # BaseLLM abstract interface
│   │   └── anthropic.py     # AnthropicLLM implementation
│   ├── orchestrator/
│   │   ├── server.py        # Orchestrator class + stdio transport
│   │   ├── planner.py       # LLM-based planner
│   │   ├── synthesizer.py   # capability synthesis
│   │   ├── executor.py      # plan execution + ElicitRequired
│   │   ├── registry.py      # capability registry
│   │   ├── capability.py    # RegisteredCapability / SynthesizedCapability
│   │   └── elicitation.py   # ElicitationManager
│   └── client/
│       └── types.py         # StreamEvent, ElicitRequest, Result, ...
├── helloworld/              # example developer project
├── tests/
├── pyproject.toml
└── LICENSE
```

---

## Bring your own LLM

Implement `BaseLLM` to use any provider:

```python
from csp.llm import BaseLLM, LLMMessage, LLMResponse

class MyLLM(BaseLLM):
    async def complete(self, messages, *, max_tokens=4096, temperature=0.0, system=None) -> LLMResponse:
        # call your provider here
        return LLMResponse(content="...", input_tokens=0, output_tokens=0)

app = Orchestrator("my-app", llm=MyLLM())
```

---

## License

MIT — see [LICENSE](LICENSE).
