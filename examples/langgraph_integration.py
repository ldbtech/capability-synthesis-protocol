"""
examples/langgraph_integration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Three ways to use a CSP Orchestrator inside LangGraph — the same Orchestrator,
consumed differently, exactly as MCP servers are consumed across transports.

Run:
    pip install -e ".[langgraph]"
    ANTHROPIC_API_KEY=sk-ant-... python examples/langgraph_integration.py
"""

import asyncio
from typing import Any, TypedDict

from csp import Orchestrator, AnthropicLLM
from csp.adapters.langgraph import csp_node, csp_tool, build_csp_graph

from langgraph.graph import StateGraph, START, END


# A CSP app with one hand-written capability. Anything else the planner needs
# (e.g. statistics) is synthesized as real code at runtime.
app = Orchestrator("langgraph-demo", llm=AnthropicLLM(), planner_dir=None)


@app.capability("greet")
async def greet(name: str = "world") -> dict:
    """Greet a person by name."""
    return {"message": f"Hello, {name}!"}


# ── 1. One-line compiled graph ────────────────────────────────────────────────
async def demo_quick_graph():
    graph = build_csp_graph(app)
    out = await graph.ainvoke({"goal": "greet Alice"})
    print("1) build_csp_graph →", out["csp_result"]["output"])


# ── 2. CSP as a node in your own graph (with ambient data → synthesis) ────────
class State(TypedDict):
    goal: str
    data: dict
    csp_result: Any


async def demo_node_in_graph():
    g = StateGraph(State)
    g.add_node("csp", csp_node(app, ambient_key="data"))
    g.add_edge(START, "csp")
    g.add_edge("csp", END)
    graph = g.compile()

    rows = [{"x": 2}, {"x": 4}, {"x": 6}, {"x": 8}]
    out = await graph.ainvoke({
        "goal": "compute the mean and standard deviation of the x values",
        "data": {"rows": rows},
    })
    print("2) csp_node →", out["csp_result"]["output"])


# ── 3. CSP as a single tool an agent can call ─────────────────────────────────
async def demo_tool():
    tool = csp_tool(app)
    res = await tool.ainvoke({"goal": "greet Charlie"})
    print(f"3) csp_tool ({tool.name}) →", res["output"])


async def main():
    await demo_quick_graph()
    await demo_node_in_graph()
    await demo_tool()


if __name__ == "__main__":
    asyncio.run(main())
