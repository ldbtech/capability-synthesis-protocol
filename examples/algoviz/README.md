# AlgoViz — the self-building algorithm visualizer

A demo of **CSP + LangGraph** together. Type any algorithm. There is **no code
for it** anywhere in this app — CSP writes the visualizer on the fly, runs it as
a node inside a LangGraph workflow, and animates the result.

> "I gave an AI a goal with an empty toolbox. It wrote, tested, and ran its own
> visualization tool inside a live LangGraph — and you watch it happen."

## What you see

1. Type *"visualize quicksort"* (or BFS, merge sort, binary search, Dijkstra…).
2. A **LangGraph workflow** lights up: `understand → build → narrate`.
3. The `build` node hands the goal to **CSP**. No `visualize_quicksort`
   capability exists, so CSP **synthesizes real Python** for it…
4. …the freshly generated code appears in the UI (the "this didn't exist a
   second ago" moment), runs in a sandbox, and produces animation frames.
5. The frames play as an animation, with a plain-language explanation.

If a synthesized capability produces no frames, the graph **self-corrects**:
it forgets the bad capability and re-synthesizes (up to 3 tries).

```
algoviz/
├── backend/
│   ├── app.py        FastAPI + SSE (port 8001)
│   ├── graph.py      LangGraph StateGraph: understand → build(CSP) → narrate
│   └── csp_app.py    CSP Orchestrator + visualization synthesis_guidance
├── frontend/         React + Vite (port 5174)
└── .env              ANTHROPIC_API_KEY (+ optional ANTHROPIC_MODEL)
```

## How CSP plugs into LangGraph

`graph.py` builds a real `StateGraph`. The `build` node drives CSP via
`csp.submit(goal, ambient=...)` and forwards every CSP event (plan, synthesis,
generated code, execution) to the browser. This is the same pattern as
`csp.adapters.langgraph` — CSP is just another node.

## Run

Needs the repo installed once (`pip install -e .` from the repo root) plus:

```bash
pip install fastapi "uvicorn[standard]" langgraph matplotlib networkx
```

**Backend** (port 8001):

```bash
cd algoviz/backend
../../.venv/bin/python -m uvicorn app:api --reload --port 8001
```

**Frontend** (port 5174):

```bash
cd algoviz/frontend
npm install
npm run dev
```

Open http://localhost:5174.

### Runs alongside the CSV-RAG demo

Different ports, so both apps run at once:

| App | Backend | Frontend |
|---|---|---|
| CSV-RAG (`helloworld/`) | :8000 | :5173 |
| AlgoViz (`algoviz/`) | :8001 | :5174 |

## Try

`visualize quicksort` · `merge sort` · `animate binary search` ·
`show BFS on a graph` · `selection sort` · `Dijkstra shortest path`

Ask the same algorithm twice — the second time the `build` node **borrows** the
existing `visualize_<algo>` capability (`csp.borrow(...)`) and invokes it
directly: no planner, no LLM, no synthesis. The UI shows
**🔗 Borrowed (reused)** instead of **⚡ Invented live**, and the synthesized
source is in `backend/planner/capabilities/`.
