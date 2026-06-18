# CSP Architecture

CSP borrows MCP's shape — a host talks to a server over JSON-RPC 2.0 — but adds
two things MCP doesn't have:

1. When a capability **doesn't exist, the server writes the code for it at runtime
   and runs it** (synthesis).
2. What it writes is a **general, reusable verb** — not a one-off task — so the
   *next* goal reuses it instead of generating a near-duplicate (reuse-first), and
   an existing one can be **patched in place** by instruction (evolve).

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

## 2.5 The capability model: general verbs, not one-off tasks

This is the idea the whole system is built around. A **capability is a general,
parameterized verb** — `plot_chart`, `aggregate_table`, `call_rest_api` — that
handles an *entire class* of work. The specifics of any one request (which
columns, which chart kind, which URL, which aggregation) live in the **invocation
args**, never baked into the code.

```
   plot_revenue_by_quarter()          ✗  a TASK — frozen, used once, regenerated next time
   plot_chart(kind, x, y, agg, ...)   ✓  a VERB — reused for every chart, forever
```

So "plot avg salary by department as bars" and "plot age distribution as a
histogram" are **two invocations of one `plot_chart`**, differing only by args
(`kind=bar` vs `kind=histogram`). The JSON-RPC `params_schema` *is* that
interface — the contract that says "this verb's knobs are kind, x, y, agg, …".

Three mechanisms make this real:

| Where | What it does |
|---|---|
| **Registry → planner summary** | exposes each synthesized cap's `params_schema` ("\| params: kind, x, y, agg") so the planner can *see the knobs* and reinvoke with new args. A cap it can't see the interface of can't be reused. |
| **Planner (reuse-first)** | matches a goal on capability *category* (plotting, aggregation, lookup, API call), reuses with new args when one fits, and only synthesizes when no category matches. New caps are named as general verbs; a variation (bar vs scatter, mean vs median) is an **arg, not a new capability**. |
| **Synthesizer (generality contract)** | when it must write a new cap, it designs `params_schema` as the full-class interface first and reads *every* specific from `args` with sensible defaults — never hardcoding the request that triggered it. |

**Why it matters:** task capabilities synthesize without bound (every question = new
code). Verb capabilities form a small, growing, composable toolset — which is what
makes CSP usable for automated data-engineering / analytics **pipelines**
(`ingest_csv → clean_table → aggregate_table → plot_chart`, each a reusable
JSON-RPC node). Because the contract is language-agnostic, the same verb can later
be backed by a different runtime (Python today, TS/d3 later) without the planner
noticing.

Measured on csv-rag (helloworld): 5 varied goals → **2** synthesized verbs
(`plot_chart`, `aggregate_table`), the rest reused; the planner even *composes*
`aggregate_table → plot_chart` for "plot the average salary by department".

---

## 3. Request lifecycle (a single goal)

```
  user goal: "plot the average salary by department as a bar chart"
        │
        ▼
  ┌───────────┐   "what caps exist + their params?"  ┌────────────┐
  │  Planner  │ ───────────────────────────────────► │  Registry  │
  └─────┬─────┘ ◄─────────────────────────────────── └────────────┘
        │  plan = [ step(capability, args, needs_synthesis?) ]
        │  e.g. plot_chart(kind=bar, x=department, y=salary, agg=mean)
        ▼
  for each step ───────────────────────────────────────────────────────────┐
        │                                                                    │
        ├── exists & registered ──► Executor runs your async fn ─────────┐   │
        │                                                                │   │
        ├── exists & synthesized ─► REUSE: run generated code with the   │   │
        │                           new args in the sandbox (no LLM) ────┤   │
        │                                                                │   │
        ├── evolve__<name> ───────► Synthesizer patches the existing cap │   │
        │                           by instruction, persist, re-run ─────┤   │
        │                                                                │   │
        └── missing ──► Synthesizer writes GENERAL Python ──► compile- ──┤   │
                          check ──► store in Registry + persist to       │   │
                          planner/ ──► run in PythonSandbox ─────────────┘   │
        │                                                                    │
        ▼  step output                                                       │
  collect outputs ◄────────────────────────────────────────────────────────┘
        │
        ▼
  result { status, summary, output }  ──► streamed back to the consumer
```

Two key invariants:

- **Reuse over resynthesis.** A *general verb* is synthesized once, then reused
  for the whole class of goals by passing new args (see §2.5). It lives in the
  registry and on disk (`planner/capabilities/<name>.py`), survives restarts
  (reloaded at init), and can be explicitly **borrowed** (`async with
  app.borrow(name)`), which guarantees it can't be replaced while in use.
- **Evolve in place.** When a goal asks to change an existing synthesized cap
  ("add humidity to the weather widget"), the planner emits an `evolve__<name>`
  step: the synthesizer is handed the cap's current code + the instruction,
  patches only what's needed, persists the new version, and re-runs it — instead
  of starting from scratch.

---

## 3.5 API credentials (when synthesized code needs a key)

Some verbs need a real secret — a weather key, a stock API token. CSP handles
this without the developer pre-wiring anything:

```
  Synthesizer emits, between the python and json blocks:
      ##CREDENTIALS
      OPENWEATHER_API_KEY: OpenWeather · get at https://openweathermap.org/api
        │
        ▼
  Before execution, the orchestrator checks the CredentialStore for each declared key
        │
        ├── present ─► injected as an env var into the sandbox subprocess (os.environ)
        │
        └── missing ─► stream a `credential_required` event to the consumer; the UI
                       collects the key, POSTs it back (app.provide_credential), and
                       the goal auto-retries. Keys persist to credentials/ (never logged).
```

The generated code just reads `os.environ["OPENWEATHER_API_KEY"]`. Credentials are
per-key, stored once, reused on every future invocation.

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
| Capabilities | fixed, pre-written tools | registered **and** synthesized at runtime, as **general verbs** |
| Missing capability | not available | **written as real code + run** |
| New request, same class | new tool needed | **reused** with new args (no new code) |
| Changing a capability | edit + redeploy | **`evolve__<name>`** — patched in place by instruction |
| Reuse | call the tool | call **or** `borrow()` (Rust-like) |
| Credentials | configured per server | declared by the code, gated + injected on demand |
| Persistence | — | `planner/` (specs, generated `.py`, logs, plans) |
| Execution of new logic | n/a | sandboxed subprocess (timeout, isolation) |
| Frameworks | `langchain-mcp-adapters`, … | `csp.adapters.langgraph`, … |

---

## 6. Where each piece lives

```
csp/orchestrator/
  server.py        Orchestrator: run / submit / run_goal / call_capability / borrow / forget / evolve
  planner.py       Planner          — reuse-first vs synthesize vs evolve decision (LLM); general-verb naming
  registry.py      CapabilityRegistry — owns caps; borrow counting; exposes params_schema to planner
  synthesizer.py   Synthesizer      — LLM writes/evolves real Python (generality contract, two-block format)
  sandbox.py       PythonSandbox    — runs generated code in a subprocess (timeout, per-call env)
  executor.py      Executor         — walks the plan; ElicitRequired (human-in-the-loop)
  capability.py    Registered / Synthesized capability types (Synthesized carries params_schema)
  credentials.py   CredentialStore / CredentialSpec — API-key declare-gate-inject flow
  elicitation.py   ElicitationManager — pause a goal to ask the user a question
  borrow.py        BorrowScope / BorrowedCapability / BorrowError
  planner_store.py planner/ folder  — JSON-RPC log, specs+code, plans
csp/adapters/
  langgraph.py     csp_node / csp_tool / build_csp_graph
```
