"""
csp.adapters.langgraph
~~~~~~~~~~~~~~~~~~~~~~~~
Use a CSP Orchestrator inside LangGraph — three ways, mirroring how
`langchain-mcp-adapters` lets you drop an MCP server into a graph.

1. csp_node(orchestrator)         → an async graph node that reads a goal from
                                     state, runs CSP, writes the result back.
2. csp_tool(orchestrator)         → a LangChain tool an existing agent can call
                                     ("run anything, synthesizing code if the
                                     capability doesn't exist yet").
3. build_csp_graph(orchestrator)  → a ready-to-run compiled StateGraph with a
                                     single CSP node, for quick embedding.

LangGraph / LangChain are imported lazily so plain CSP installs don't need them.
Install with:  pip install "csp-sdk[langgraph]"

Example
-------
    from csp import Orchestrator, AnthropicLLM
    from csp.adapters.langgraph import csp_node
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Any

    app = Orchestrator("my-app", llm=AnthropicLLM())

    class S(TypedDict):
        goal: str
        csp_result: Any

    g = StateGraph(S)
    g.add_node("csp", csp_node(app))
    g.add_edge(START, "csp")
    g.add_edge("csp", END)
    graph = g.compile()

    out = await graph.ainvoke({"goal": "average salary per department"})
    print(out["csp_result"]["output"])
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from ..orchestrator.server import Orchestrator


def csp_node(
    orchestrator: Orchestrator,
    *,
    goal_key: str = "goal",
    result_key: str = "csp_result",
    ambient_key: Optional[str] = None,
    on_elicit: Optional[Callable[[dict], Awaitable[str]]] = None,
) -> Callable[[dict], Awaitable[dict]]:
    """
    Build an async LangGraph node that runs CSP on ``state[goal_key]``.

    Parameters
    ----------
    orchestrator:
        The CSP Orchestrator to drive.
    goal_key:
        State key holding the natural-language goal.
    result_key:
        State key to write the CSP result dict into.
    ambient_key:
        Optional state key whose value (a dict) is passed as ambient data —
        e.g. {"rows": [...], "columns": [...]} for synthesized code to use.
    on_elicit:
        Optional async callback for human-in-the-loop approvals.

    Returns
    -------
    An ``async def node(state) -> dict`` suitable for ``graph.add_node``.
    """
    async def node(state: dict) -> dict:
        goal = state[goal_key]
        ambient = state.get(ambient_key) if ambient_key else None
        result = await orchestrator.run_goal(goal, ambient=ambient, on_elicit=on_elicit)
        return {result_key: result}

    return node


def csp_tool(
    orchestrator: Orchestrator,
    *,
    name: str = "csp_execute",
    description: Optional[str] = None,
    ambient_provider: Optional[Callable[[], dict]] = None,
):
    """
    Expose CSP as a single LangChain tool an agent can call.

    The agent passes a natural-language ``goal``; CSP plans it, synthesizes and
    runs real code for any capability that doesn't exist yet, and returns the
    result. This gives a LangGraph/LangChain agent CSP's full plan→synthesize→
    execute power as one tool — analogous to mounting an MCP server's tools.

    Requires ``langchain-core``.
    """
    from langchain_core.tools import StructuredTool   # lazy import

    desc = description or (
        "Achieve a goal by planning and executing capabilities. If a needed "
        "capability does not exist, it is synthesized as real code and run. "
        "Input: a natural-language goal string. Returns the execution result."
    )

    async def _run(goal: str) -> dict:
        ambient = ambient_provider() if ambient_provider else None
        return await orchestrator.run_goal(goal, ambient=ambient)

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=desc,
    )


def build_csp_graph(
    orchestrator: Orchestrator,
    *,
    goal_key: str = "goal",
    result_key: str = "csp_result",
    ambient_key: Optional[str] = None,
):
    """
    Build and compile a minimal StateGraph with one CSP node.

    Returns a compiled graph you can ``.ainvoke({goal_key: "..."})``. Handy for
    embedding CSP as a sub-graph or for a quick standalone runner.

    Requires ``langgraph``.
    """
    from langgraph.graph import StateGraph, START, END   # lazy import

    # An open dict state — keeps the adapter schema-agnostic.
    builder = StateGraph(dict)
    builder.add_node("csp", csp_node(
        orchestrator,
        goal_key=goal_key,
        result_key=result_key,
        ambient_key=ambient_key,
    ))
    builder.add_edge(START, "csp")
    builder.add_edge("csp", END)
    return builder.compile()
