# CSP Architecture

CSP borrows MCP's shape — a host talks to a server over JSON-RPC 2.0 — but adds
one thing MCP doesn't have: when a capability **doesn't exist, the server writes
the code for it at runtime and runs it.**

---

## 1. The MCP shape (for reference)

```
┌─────────────────────────────┐                          ┌────────────────────┐
│  Host  (e.g. Claude Desktop) │      JSON-RPC 2.0        │     MCP Server      │──► Tools
│  ┌───────────────────────┐  │      (stdio / HTTP)      │                     │──► Resources
│  │      MCP Client        │──┼─────────────────────────►│  exposes a FIXED    │──► Prompts
│  └───────────────────────┘  │                          │  set of tools       │
└─────────────────────────────┘                          └────────────────────┘
        a tool must be pre-written by the developer ───────────┘
```

## 2. The CSP shape (same transport, self-extending server)

```
        CONSUMER  (any of these)                 TRANSPORT                 CSP ORCHESTRATOR
┌──────────────────────────────────┐                              ┌──────────────────────────────┐
│  • stdio host / CLI               │     JSON-RPC 2.0  (stdio)    │                              │
│  • FastAPI + SSE  (web UIs)       │ ───────────────────────────►│   Planner                    │
│  • LangGraph node / tool          │           OR                │     │  decide: reuse or make? │
│  • a script: await app.run_goal() │     in-process  async       │     ▼                        │
└──────────────────────────────────┘     submit()/run_goal()/     │   Registry ◄──── borrow()    │
                 ▲                        call_capability()/borrow │     │   (registered +         │
                 │                                                 │     │    synthesized caps)    │
                 │   events / result                               │     ▼                        │
                 └─────────────────────────────────────────────── │   Executor                   │
                                                                   │     │                        │
                                                                   └─────┼────────────┬───────────┘
                                                                         │            │
                                              registered? run your fn ◄──┘            └──► needs synthesis
                                                                                            │
                                                            ┌───────────────────────────────▼─────────┐
                                                            │  Synthesizer   (asks the LLM for code)   │
                                                            │       │  ```python def run(args): ...```  │
                                                            │       ▼                                   │
                                                            │  PythonSandbox  (subprocess + timeout)    │
                                                            │       runs the generated code for real    │
                                                            └───────────────────────────────────────────┘
                                                                         │ persist
                                                                         ▼
                                                            planner/  (spec.json + generated .py +
                                                                       jsonrpc.ndjson log + plans/)
                                                                         │
                                                                  reloaded next run → never synthesized twice
```

```
                                   ┌──────────────────────┐
   LLM provider (Anthropic / your  │  BaseLLM.complete()  │   used by Planner + Synthesizer only
   own BaseLLM) ──────────────────►│                      │   (capability execution never needs it)
                                   └──────────────────────┘
```

---

## 3. Request lifecycle (a single goal)

```
  user goal: "average salary by department"
        │
        ▼
  ┌───────────┐   "what capabilities exist?"      ┌────────────┐
  │  Planner  │ ────────────────────────────────► │  Registry  │
  └─────┬─────┘ ◄──────────────────────────────── └────────────┘
        │  plan = [ step(capability, args, needs_synthesis?) ]
        ▼
  for each step ───────────────────────────────────────────────────────────┐
        │                                                                    │
        ├── exists & registered ──► Executor runs your async fn ─────────┐   │
        │                                                                │   │
        ├── exists & synthesized ─► borrow + run generated code (sandbox)│   │
        │                                                                │   │
        └── missing ──► Synthesizer writes Python ──► compile-check ──►  │   │
                          store in Registry + persist to planner/ ──► run│   │
                                                       in PythonSandbox ─┘   │
        │                                                                    │
        ▼  step output                                                       │
  collect outputs ◄────────────────────────────────────────────────────────┘
        │
        ▼
  result { status, summary, output }  ──► streamed back to the consumer
```

Key invariant: **a capability is synthesized at most once.** After that it lives
in the registry and on disk (`planner/capabilities/<name>.py`), and is **reused**
— or explicitly **borrowed** (`async with app.borrow(name)`), which guarantees it
can't be replaced while in use.

---

## 4. Ways to drive the same Orchestrator

| API | Planner? | Streams? | Use |
|---|---|---|---|
| `app.run()` | yes | yes (stdio) | MCP-style JSON-RPC server |
| `app.submit(goal)` | yes | yes (dicts) | FastAPI/SSE, live UIs |
| `app.run_goal(goal)` | yes | no | scripts, adapters, tests |
| `app.call_capability(name, **args)` | no | no | direct call (MCP `tools/call`) |
| `async with app.borrow(name)` | no | no | reuse an existing capability safely |

---

## 5. CSP vs MCP

| | MCP | CSP |
|---|---|---|
| Transport | JSON-RPC 2.0 (stdio / HTTP) | JSON-RPC 2.0 (stdio) + in-process |
| Capabilities | fixed, pre-written tools | registered **and** synthesized at runtime |
| Missing capability | not available | **written as real code + run** |
| Reuse | call the tool | call **or** `borrow()` (Rust-like) |
| Persistence | — | `planner/` (specs, generated `.py`, logs, plans) |
| Execution of new logic | n/a | sandboxed subprocess (timeout, isolation) |
| Frameworks | `langchain-mcp-adapters`, … | `csp.adapters.langgraph`, … |

---

## 6. Where each piece lives

```
csp/orchestrator/
  server.py        Orchestrator: run / submit / run_goal / call_capability / borrow / forget
  planner.py       Planner          — reuse vs synthesize decision (LLM)
  registry.py      CapabilityRegistry — owns caps; borrow counting
  synthesizer.py   Synthesizer      — LLM writes real Python (two-block format)
  sandbox.py       PythonSandbox    — runs generated code in a subprocess
  executor.py      Executor         — walks the plan; ElicitRequired (human-in-the-loop)
  capability.py    Registered / Synthesized capability types
  borrow.py        BorrowScope / BorrowedCapability / BorrowError
  planner_store.py planner/ folder  — JSON-RPC log, specs+code, plans
csp/adapters/
  langgraph.py     csp_node / csp_tool / build_csp_graph
```
