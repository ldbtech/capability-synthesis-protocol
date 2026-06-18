# CSP — Capability Synthesis Protocol

CSP is a Python library for building AI orchestrators that **plan**, **execute**, and **synthesize** capabilities at runtime.

You register Python functions as capabilities and submit natural-language goals. CSP plans which capabilities to run. **If a capability doesn't exist, CSP writes real Python for it on the fly, runs that code in a sandbox, and reuses it forever after.** The wire format and consumption model mirror MCP (Model Context Protocol).

---

## Install

This project is managed with [uv](https://docs.astral.sh/uv/). Install uv once
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
git clone https://github.com/ldbtech/capability-synthesis-protocol
cd csp
uv sync                          # core lib + dev tools, into .venv
```

Optional extras:

```bash
uv sync --extra langgraph        # LangGraph adapter
uv run pytest                    # run the test suite (no activation needed)
```

> Prefer plain pip? `pip install -e ".[langgraph]"` still works — uv just adds a
> committed `uv.lock` for reproducible installs.

---

## Quickstart

```python
from csp import Orchestrator, ElicitRequired, AnthropicLLM

app = Orchestrator(
    "my-server",
    llm=AnthropicLLM(),          # reads ANTHROPIC_API_KEY + ANTHROPIC_MODEL from env
    # llm=AnthropicLLM(api_key="sk-ant-...", model="claude-sonnet-4-6"),
)

@app.capability("greet")
async def greet(name: str, language: str = "english") -> dict:
    """Greet a person in their preferred language."""
    greetings = {"english": "Hello", "spanish": "Hola", "japanese": "こんにちは"}
    return {"message": f"{greetings.get(language, 'Hello')}, {name}!"}

@app.capability("send_report")
async def send_report(recipient: str, _elicit_response: str = "") -> dict:
    """Send a report — asks for approval first."""
    if not _elicit_response:
        raise ElicitRequired(kind="approval", question=f"Send report to {recipient}?")
    return {"sent": _elicit_response.lower() == "yes"}

if __name__ == "__main__":
    app.run()   # stdio JSON-RPC server — identical to MCP
```

```bash
ANTHROPIC_API_KEY=sk-ant-... python server.py
```

---

## How it works

```
Goal: "average salary by department"
        │
   ┌────▼─────┐   no matching capability?
   │ Planner  │──────────────────────────────┐
   └────┬─────┘                               │
        │ found a registered capability       │ needs synthesis
   ┌────▼─────────┐                    ┌───────▼────────────┐
   │  Executor    │                    │   Synthesizer      │
   │ runs your fn │                    │ LLM writes real    │
   └────┬─────────┘                    │ Python (run(args)) │
        │                              └───────┬────────────┘
        │                              ┌───────▼────────────┐
        │                              │  PythonSandbox      │
        │                              │ runs the code in a  │
        │                              │ subprocess (timeout)│
        │                              └───────┬────────────┘
        └──────────────► result ◄──────────────┘
```

A synthesized capability's spec **and** its generated `.py` are written to a
`planner/` folder in your project and **reloaded on the next run** — so a
capability is synthesized at most once.

> Full diagrams (MCP-style architecture, request lifecycle, CSP↔MCP comparison):
> see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## API surface

The same `Orchestrator` can be driven however you need — CSP keeps its core
(plan → synthesize → execute) separate from how you consume it.

| Call | What it does |
|---|---|
| `app.run()` | Start a **stdio JSON-RPC** server (MCP-style host/subprocess). |
| `async for ev in app.submit(goal, ambient=…)` | Plan + execute, **streaming** event dicts (FastAPI/SSE, live UIs). |
| `await app.run_goal(goal, ambient=…)` | Headless **one-shot** → final result dict (scripts, adapters). |
| `await app.call_capability(name, **args)` | **Direct** call of one capability — no planner. CSP's `tools/call`. |
| `async with app.borrow(name) as cap:` | **Borrow** an existing capability (Rust-like) — never synthesizes. |
| `await app.list_capabilities()` | All capabilities (registered + synthesized, with generated code). |
| `await app.forget(name)` | Drop a synthesized capability so it regenerates (blocked while borrowed). |

### Borrowing (reuse, the Rust way)

Synthesis *creates* a capability; **borrowing takes a shared, read-only handle
to one that already exists** — it never creates a duplicate. Like `&T` in Rust:

```python
async with app.borrow("detect_anomalies") as cap:   # KeyError if it doesn't exist
    result = await cap.invoke(rows=rows)             # read-only handle
    # while borrowed, app.forget("detect_anomalies") raises BorrowError
```

- Borrowing a capability that doesn't exist **raises** (never silently
  synthesizes a new one).
- Many services can hold **shared** borrows of the same capability at once.
- A capability **cannot be forgotten or replaced while it's borrowed** — the
  registry enforces it, like Rust won't free a value that's still borrowed.
- Borrows are **scoped**: released automatically at the end of the `async with`.

`ambient` is a dict (e.g. `{"rows": [...], "columns": [...]}`) merged into every
step's args, so synthesized code can compute over your data.

### Orchestrator options

```python
Orchestrator(
    name, llm,
    planner_dir="planner",          # where specs/logs/plans persist (None to disable)
    synthesis_guidance="",          # app-specific conventions for generated code
    sandbox_env={"MPLBACKEND": "Agg"},  # extra env for the sandbox subprocess
    synthesis_timeout=30.0,
    elicitation_timeout=120.0,
)
```

`synthesis_guidance` is how an **app** teaches CSP its domain (data shapes,
output formats like "plots → base64 PNG") without the library knowing anything
domain-specific.

---

## Use it inside LangGraph

```bash
pip install -e ".[langgraph]"
```

```python
from csp.adapters.langgraph import csp_node, csp_tool, build_csp_graph
from langgraph.graph import StateGraph, START, END

# A) CSP as a node in your own graph
g = StateGraph(dict)
g.add_node("csp", csp_node(app, ambient_key="data"))
g.add_edge(START, "csp"); g.add_edge("csp", END)
graph = g.compile()
out = await graph.ainvoke({"goal": "mean of the x values", "data": {"rows": rows}})

# B) CSP as one tool an agent can call (synthesizes code on demand)
tool = csp_tool(app)          # a LangChain StructuredTool

# C) one-line compiled graph
graph = build_csp_graph(app)
```

Adapters import their framework lazily — a plain `csp-sdk` install never pulls in
LangGraph. Other frameworks plug in the same way under [`csp/adapters/`](csp/adapters/).
Runnable example: [`examples/langgraph_integration.py`](examples/langgraph_integration.py).

---

## Demo apps

All demo apps live under [`examples/`](examples/), each with its own backend +
frontend. They run on **different ports**, so several can be up at once. The
simplest way to launch them is the bundled Makefile:

```bash
cd examples
make install          # one-time: npm install for every frontend
make dev              # run csv-rag + algoviz + montage + pitch (Ctrl-C stops all)
make csv-rag          # …or just one app
```

| App | Folder | Shows | Backend | Frontend |
|---|---|---|---|---|
| **CSV-RAG** | [`examples/helloworld/`](examples/helloworld/) | RAG for lookups + synthesized code for analysis/plots | :8000 | :5173 |
| **AlgoViz** | [`examples/algoviz/`](examples/algoviz/) | CSP **inside LangGraph**; invents an algorithm visualizer live | :8001 | :5174 |
| **Montage AI** | [`examples/montage-ai/`](examples/montage-ai/) | Figma-style canvas that **synthesizes its own design capabilities** | :8002 | :5175 |
| **Pitch** | [`examples/pitch/`](examples/pitch/) | Live World Cup copilot — fetches real data + synthesizes predictions | :8003 | :5176 |

### CSV-RAG — ask anything about a CSV

```bash
cd examples && make csv-rag        # → http://localhost:5173
# or manually:
cd examples/helloworld/backend && ../../../.venv/bin/python -m uvicorn app:api --reload --port 8000
cd examples/helloworld/frontend && npm install && npm run dev          # http://localhost:5173
```

Upload `examples/helloworld/sample_data/employees.csv`, then try:

- **Lookup (RAG):** `Who works in Engineering?` · `Find the most experienced person in Seattle`
- **Computed (synthesized):** `Average salary by department` · `Correlation between age and salary` · `Top 5 highest-paid employees` · `What percent earn above the median?`
- **Plots (synthesized matplotlib):** `Plot a histogram of salaries` · `Bar chart of average salary by department` · `Scatter of age vs salary`

### AlgoViz — self-building algorithm visualizer (CSP + LangGraph)

```bash
cd examples && make algoviz        # → http://localhost:5174
# or manually:
cd examples/algoviz/backend && ../../../.venv/bin/python -m uvicorn app:api --reload --port 8001
cd examples/algoviz/frontend && npm install && npm run dev             # http://localhost:5174
```

Type an algorithm — there's no code for it, so CSP writes the visualizer live
and runs it as a node in a LangGraph workflow (`understand → build → narrate`):

- **Sorts/searches:** `visualize quicksort` · `merge sort` · `insertion sort` · `animate binary search`
- **Graphs:** `show BFS on a graph` · `Dijkstra shortest path`
- **Novel:** `visualize the sieve of Eratosthenes` · `animate the Tower of Hanoi`

### Montage AI — self-evolving design canvas

```bash
cd examples && make montage        # → http://localhost:5175
# or manually:
cd examples/montage-ai/backend && ../../../.venv/bin/python -m uvicorn app:app --reload --port 8002
cd examples/montage-ai/frontend && npm install && npm run dev          # http://localhost:5175
```

Type design requests in natural language — CSP synthesizes the layout capability live and renders it on the SVG canvas:

- **Layouts:** `Create a mobile app login screen` · `Make a 3-column pricing card layout`
- **Components:** `Add a navigation bar at the top` · `Create a dashboard with stats cards`
- **Creative:** `Draw a pie chart with 4 segments` · `Make a timeline with 4 steps`

Every new design operation that didn't exist before is synthesized, persisted to `planner/`, and reused on future runs.

---

The library itself only needs `uv sync`. The demo apps need a few extra
runtime libs (FastAPI, RAG embeddings, plotting) — install them into the venv
once, from the repo root:

```bash
uv pip install fastapi "uvicorn[standard]" pandas numpy matplotlib \
               sentence-transformers requests python-multipart
uv sync --extra langgraph        # LangGraph adapter for AlgoViz
```

> Launch backends with `../../../.venv/bin/python -m uvicorn …`, not a bare
> `uvicorn`, so they use the venv's interpreter and dependencies. The
> `examples/Makefile` (`make dev`, `make pitch`, …) already does this for you.

---

## Testing

```bash
uv sync            # dev group (pytest) is installed by default
uv run pytest -q
```

The suite in [`tests/`](tests/) runs **without any network/LLM calls** (a
`FakeLLM` stands in), covering the sandbox (real execution, errors, timeouts,
env), kwargs filtering, two-block synthesis parsing, direct `call_capability`,
`forget`, and `planner/` persistence round-trips.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `ANTHROPIC_MODEL` | No | Model to use (default `claude-haiku-4-5-20251001`) |

---

## Wire protocol

JSON-RPC 2.0 over stdio (NDJSON, one message per line) — the same transport as MCP.

| Method | Direction | Description |
|---|---|---|
| `initialize` | client → server | Handshake |
| `csp.goal.submit` | client → server | Submit a natural-language goal |
| `csp.capability.list` | client → server | List capabilities |
| `csp.capability.invoke` | — | Spec method for a single capability (see `call_capability`) |
| `csp.stream.event` | server → client | Streaming progress event |
| `csp.elicit.request` / `csp.elicit.respond` | both | Human-in-the-loop |
| `csp.result` | server → client | Terminal result |

---

## Project structure

```
csp/
├── csp/                       # the library
│   ├── __init__.py            # public API: Orchestrator, ElicitRequired, AnthropicLLM
│   ├── llm/                   # BaseLLM + AnthropicLLM
│   ├── orchestrator/
│   │   ├── server.py          # Orchestrator: run / submit / run_goal / call_capability / forget
│   │   ├── planner.py         # LLM planner (decides reuse vs synthesize)
│   │   ├── synthesizer.py     # generates real Python (two-block format)
│   │   ├── sandbox.py         # PythonSandbox — runs generated code in a subprocess
│   │   ├── executor.py        # runs the plan; ElicitRequired
│   │   ├── registry.py        # capability registry (+ forget, persistence hook)
│   │   ├── capability.py      # Registered / Synthesized capabilities
│   │   ├── planner_store.py   # planner/ folder: JSON-RPC log, specs, plans
│   │   └── elicitation.py     # human-in-the-loop
│   ├── adapters/
│   │   └── langgraph.py       # csp_node / csp_tool / build_csp_graph
│   └── client/types.py        # StreamEvent, ElicitRequest, Result, …
├── helloworld/                # CSV-RAG demo  (:8000 / :5173)
├── algoviz/                   # AlgoViz demo  (:8001 / :5174)
├── examples/                  # langgraph_integration.py
├── tests/                     # network-free pytest suite
├── pyproject.toml
└── LICENSE
```

---

## Bring your own LLM

```python
from csp.llm import BaseLLM, LLMResponse

class MyLLM(BaseLLM):
    async def complete(self, messages, *, max_tokens=4096, temperature=0.0, system=None) -> LLMResponse:
        ...  # call your provider
        return LLMResponse(content="...", input_tokens=0, output_tokens=0)

app = Orchestrator("my-app", llm=MyLLM())
```

---

## License

MIT — see [LICENSE](LICENSE).
