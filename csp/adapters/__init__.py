"""
csp.adapters
~~~~~~~~~~~~
Framework adapters for CSP.

CSP's core is framework-neutral: the Orchestrator turns a goal into a stream
of plain event dicts (`submit`) or a single result dict (`run_goal`). Adapters
wrap that core so CSP can be consumed the way each ecosystem expects —
exactly how MCP servers are consumed over stdio, HTTP, or via
`langchain-mcp-adapters`.

The same Orchestrator can therefore be driven as:

  - a stdio JSON-RPC server         →  Orchestrator.run()        (MCP-style)
  - an in-process async stream      →  Orchestrator.submit()     (FastAPI/SSE)
  - a one-shot coroutine            →  Orchestrator.run_goal()   (scripts/tests)
  - a LangGraph node or tool        →  csp.adapters.langgraph
  - (future) CrewAI / AutoGen / …   →  add an adapter here

Adapters import their target framework lazily, so installing CSP never pulls
in LangGraph et al. unless you actually use the adapter.
"""

__all__ = ["langgraph"]
