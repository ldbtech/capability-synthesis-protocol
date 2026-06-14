"""
backend/graph.py
~~~~~~~~~~~~~~~~~
The LangGraph workflow that drives the visualizer.

    START → understand → build (CSP) → narrate → END

- understand : an LLM node that normalizes the user's request into an
               algorithm name + a concrete sample input.
- build      : the CSP node — submits the goal, CSP discovers no such
               capability exists, synthesizes real Python on the fly, runs it
               in the sandbox, and returns animation frames. Every CSP event
               (planning, synthesizing, the generated code, execution) is
               forwarded live to the browser.
- narrate    : an LLM node that explains the algorithm in plain language.

Live events are pushed through a contextvar-bound emitter so the FastAPI layer
can stream them as SSE while the graph runs.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Awaitable, Callable, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from csp_app import app as csp

log = logging.getLogger("algoviz.graph")

# Per-request event sink. FastAPI sets this; graph nodes emit through it.
emitter: contextvars.ContextVar[Optional[Callable[[dict], Awaitable[None]]]] = (
    contextvars.ContextVar("emitter", default=None)
)


async def emit(event: dict) -> None:
    fn = emitter.get()
    if fn is not None:
        await fn(event)


class VizState(TypedDict, total=False):
    request: str          # raw user request
    algorithm: str        # normalized algorithm name
    sample_input: Any     # concrete input chosen by 'understand'
    goal: str             # goal handed to CSP
    frames: list          # base64 PNG frames
    step_count: int
    capability: str       # name of the synthesized capability
    code: str             # the generated Python
    narration: str
    status: str


# ── Node 1: understand (calls the registered `understand` capability) ─────────
async def understand(state: VizState) -> VizState:
    await emit({"type": "node", "node": "understand", "status": "running"})

    res = await csp.call_capability("understand", request=state["request"])
    algo = res.get("algorithm", state["request"])

    await emit({"type": "node", "node": "understand", "status": "done",
                "detail": f"algorithm = {algo}"})

    # Deterministic, single-capability goal — keeps the plan to one synthesized
    # capability that both runs the algorithm and renders every frame.
    goal = (
        f"Using a SINGLE capability named visualize_{algo}, run the {algo} "
        f"algorithm step by step on the provided input and render each step as "
        f"an animation frame, returning the frames."
    )
    return {
        "algorithm": algo,
        "sample_input": res.get("sample_input"),
        "goal": goal,
    }


# ── Node 2: build (CSP synthesizes + runs the visualizer, self-correcting) ────
_MAX_ATTEMPTS = 3


async def build(state: VizState) -> VizState:
    await emit({"type": "node", "node": "build", "status": "running"})

    ambient = {"data": state.get("sample_input")} if state.get("sample_input") is not None else {}
    cap_name, code, frames, step_count = "", "", [], 0

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if attempt > 1:
            await emit({"type": "csp", "kind": "retry",
                        "message": f"That attempt produced no frames — re-synthesizing (try {attempt})"})

        result_payload: dict = {}
        async for ev in csp.submit(state["goal"], ambient=ambient):
            t = ev.get("type")
            if t == "planning" and "steps" in ev:
                await emit({"type": "csp", "kind": "plan", "steps": ev["steps"]})
            elif t == "event" and ev.get("kind") in ("CAPABILITY", "LOG"):
                await emit({"type": "csp", "kind": "log",
                            "message": ev.get("message", ""), "capability": ev.get("capability")})
            elif t == "result":
                result_payload = ev

        output = result_payload.get("output") or {}

        # Find the synthesized capability that ran + its generated code.
        cap_name, code = "", ""
        for c in await csp.list_capabilities():
            if c.get("kind") == "synthesized" and c.get("code") and c["name"] in output:
                cap_name, code = c["name"], c["code"]

        # Extract frames.
        frames, step_count = [], 0
        for payload in output.values():
            if isinstance(payload, dict) and isinstance(payload.get("frames"), list) and payload["frames"]:
                frames = payload["frames"]
                step_count = payload.get("step_count", len(frames))
                break

        if frames:
            if code:
                await emit({"type": "code", "capability": cap_name, "code": code})
            break

        # Failed: forget the bad capability so the next attempt regenerates it.
        if cap_name:
            await csp.forget(cap_name)

    status = "ok" if frames else "error"
    await emit({"type": "node", "node": "build", "status": "done",
                "detail": f"{len(frames)} frames in {cap_name}" if frames else "no frames after retries"})
    return {"frames": frames, "step_count": step_count,
            "capability": cap_name, "code": code, "status": status}


# ── Node 3: narrate (calls the registered `narrate` capability) ───────────────
async def narrate(state: VizState) -> VizState:
    await emit({"type": "node", "node": "narrate", "status": "running"})
    res = await csp.call_capability("narrate", algorithm=state.get("algorithm", ""))
    text = res.get("explanation", "")
    await emit({"type": "node", "node": "narrate", "status": "done"})
    await emit({"type": "narration", "text": text})
    return {"narration": text}


# ── Build the graph once ──────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(VizState)
    g.add_node("understand", understand)
    g.add_node("build", build)
    g.add_node("narrate", narrate)
    g.add_edge(START, "understand")
    g.add_edge("understand", "build")
    g.add_edge("build", "narrate")
    g.add_edge("narrate", END)
    return g.compile()


GRAPH = build_graph()


async def run_visualization(request: str) -> VizState:
    """Run the full LangGraph workflow for a request and return final state."""
    await emit({"type": "start", "request": request})
    final = await GRAPH.ainvoke({"request": request})
    await emit({"type": "done",
                "frames": final.get("frames", []),
                "step_count": final.get("step_count", 0),
                "algorithm": final.get("algorithm"),
                "capability": final.get("capability"),
                "narration": final.get("narration"),
                "status": final.get("status", "ok")})
    return final
