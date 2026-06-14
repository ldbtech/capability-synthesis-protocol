"""
csp.orchestrator.server
~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestrator server — stdio JSON-RPC 2.0 transport, MCP-style API.

The server reads JSON-RPC 2.0 requests from stdin, processes them,
and writes JSON-RPC 2.0 responses + notifications to stdout.
One message per line (NDJSON). Exactly like MCP's stdio transport.

Supported JSON-RPC methods:
  csp.goal.submit        →  submit a goal, streams events + result
  csp.capability.list    →  list all registered capabilities
  csp.elicit.respond     →  respond to a pending elicitation
  csp.ping               →  health check

Each active goal submission gets its own ElicitationManager so multiple
concurrent goals (different request ids) don't interfere.

Developer usage — identical feel to MCP:

    app = Orchestrator("my-app", llm=AnthropicLLM())

    @app.capability("predict_churn")
    async def predict_churn(dataset: str) -> dict:
        return {"accuracy": 0.87}

    if __name__ == "__main__":
        app.run()   # blocks on stdin, writes to stdout
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Optional

from ..llm.base import BaseLLM
from .capability import AnyCapability, RegisteredCapability, capability_from_function
from .elicitation import ElicitationManager
from .executor import Executor
from .planner import Planner
from .planner_store import PlannerStore
from .registry import CapabilityRegistry
from .sandbox import PythonSandbox
from .synthesizer import Synthesizer

log = logging.getLogger("csp.server")


class Orchestrator:
    """
    csp Orchestrator — the developer-facing server object.

    Mirrors MCP Server in API feel:
      - Instantiate with a name + LLM
      - Register capabilities with @app.capability()
      - Call app.run() to start (blocks on stdin)

    Parameters
    ----------
    name:
        Server name (used in JSON-RPC initialize response).
    llm:
        LLM provider for planning + synthesis.
    version:
        Server version string.
    synthesis_timeout:
        Max seconds to wait for capability synthesis.
    elicitation_timeout:
        Max seconds to wait for human elicitation response.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        *,
        version: str = "0.1.0",
        synthesis_timeout: float = 30.0,
        elicitation_timeout: float = 120.0,
        planner_dir: Optional[str] = "planner",
        synthesis_guidance: str = "",
        sandbox_env: Optional[dict[str, str]] = None,
    ) -> None:
        self._name                = name
        self._version             = version
        self._llm                 = llm
        self._synthesis_timeout   = synthesis_timeout
        self._elicitation_timeout = elicitation_timeout

        # Planner store — auto-creates planner/ in the developer's project.
        # Pass planner_dir=None to disable persistence entirely.
        self._store: Optional[PlannerStore] = (
            PlannerStore(planner_dir) if planner_dir else None
        )

        self._registry    = CapabilityRegistry()
        self._synthesizer = Synthesizer(llm, guidance=synthesis_guidance)
        self._planner     = Planner(llm, self._registry)
        self._sandbox     = PythonSandbox(env=sandbox_env)

        # Reload any capabilities synthesized in previous runs so we never
        # synthesize the same capability twice across restarts, and persist
        # newly synthesized ones as they're created.
        if self._store:
            for cap in self._store.load_capabilities():
                self._registry._synthesized[cap.name] = cap
            self._registry.persist_hook = self._store.save_capability

        # Active elicitation managers keyed by request id
        self._elicitation_managers: dict[str, ElicitationManager] = {}

        log.info("orchestrator %r v%s initialized", name, version)

    # ------------------------------------------------------------------
    # Capability registration — @app.capability() decorator
    # ------------------------------------------------------------------

    def capability(self, name: Optional[str] = None, *, description: str = ""):
        """
        Register a capability.

            @app.capability("predict_churn")
            async def predict_churn(dataset: str, target_column: str) -> dict:
                return {"accuracy": 0.87}

        If name is omitted, the function name is used.
        """
        def decorator(fn):
            cap_name = name or fn.__name__
            cap = capability_from_function(cap_name, fn, description)
            self._registry.register(cap)
            log.debug("registered @capability %r", cap_name)
            return fn
        return decorator

    # ------------------------------------------------------------------
    # In-process API — for embedding CSP in a web server (FastAPI, etc.)
    # ------------------------------------------------------------------

    async def submit(
        self,
        goal: str,
        *,
        session_id: Optional[str] = None,
        on_elicit: Optional[Any] = None,
        ambient: Optional[dict[str, Any]] = None,
    ):
        """
        Submit a goal in-process and async-yield events as plain dicts.

        Unlike run() (which is stdio JSON-RPC), this lets a Python host —
        e.g. a FastAPI route — drive the orchestrator directly and stream
        results to a browser over SSE/WebSocket.

        Each yielded dict has a "type":
          {"type": "planning", "message": ...}
          {"type": "event",    "kind": ..., "message": ..., "capability": ...}
          {"type": "elicit",   "id": ..., "question": ..., ...}
          {"type": "result",   "status": ..., "summary": ..., "output": ...}

        on_elicit: optional async callable taking the elicit params dict and
        returning the user's answer string. If None, elicitations auto-approve.
        """
        import uuid as _uuid
        session_id = session_id or str(_uuid.uuid4())

        elicit_mgr = ElicitationManager(timeout=self._elicitation_timeout)
        self._elicitation_managers[session_id] = elicit_mgr

        try:
            yield {"type": "planning", "message": "Planning your goal..."}

            plan = await self._planner.plan(goal)
            if self._store:
                self._store.save_plan(goal, plan)

            # Inject ambient data (e.g. the uploaded CSV rows) into every step's
            # args so synthesized code can compute over it. Explicit planner args
            # win over ambient on key collisions.
            if ambient:
                for step in plan.steps:
                    step.args = {**ambient, **step.args}

            yield {
                "type": "planning",
                "message": f"Plan ready: {len(plan.steps)} steps",
                "steps": [s.capability for s in plan.steps],
            }

            executor = Executor(
                registry=self._registry,
                synthesizer=self._synthesizer,
                elicitation_manager=elicit_mgr,
                goal=goal,
                sandbox=self._sandbox,
            )

            result_payload: dict[str, Any] = {}

            async for event in executor.execute(plan):
                method = event.get("method", "")
                params = event.get("params", {})

                if method == "csp.result":
                    result_payload = params
                elif method == "csp.elicit.request":
                    yield {"type": "elicit", **params}
                    answer = "yes"
                    if on_elicit is not None:
                        answer = await on_elicit(params)
                    await elicit_mgr.respond(params.get("id", ""), answer)
                else:
                    yield {"type": "event", **params}

            summary = await self._summarize(goal, plan, result_payload)
            result_payload["summary"] = summary
            yield {"type": "result", **result_payload}

        except Exception as exc:
            log.error("submit failed: %s", exc, exc_info=True)
            yield {
                "type": "result",
                "status": "ERROR",
                "summary": f"Execution failed: {exc}",
                "error": str(exc),
                "output": {},
            }
        finally:
            await elicit_mgr.cancel_all()
            self._elicitation_managers.pop(session_id, None)

    async def run_goal(
        self,
        goal: str,
        *,
        session_id: Optional[str] = None,
        on_elicit: Optional[Any] = None,
        ambient: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Headless one-shot: plan + synthesize + execute a goal and return ONLY
        the final result dict (no streaming). This is the framework-neutral
        primitive that adapters (LangGraph, etc.) build on.

        Returns the result payload: {"status", "summary", "output", ...}.
        """
        final: dict[str, Any] = {"status": "ERROR", "summary": "no result", "output": {}}
        async for ev in self.submit(
            goal, session_id=session_id, on_elicit=on_elicit, ambient=ambient
        ):
            if ev.get("type") == "result":
                final = {k: v for k, v in ev.items() if k != "type"}
        return final

    async def call_capability(self, name: str, **args: Any) -> Any:
        """
        Invoke a single capability directly by name — no planner, no LLM.

        This is the direct-call counterpart to submit()/run_goal(): when you
        already know which capability you want (e.g. a LangGraph node calling a
        registered capability), call it straight. Mirrors MCP's `tools/call`.

        Works for both registered (Python function) and synthesized (sandboxed
        code) capabilities. Raises KeyError if the capability doesn't exist.
        """
        cap = await self._registry.resolve(name)
        if cap is None:
            raise KeyError(f"capability not found: {name!r}")

        if isinstance(cap, RegisteredCapability):
            return await cap.invoke(**args)

        # Synthesized — run its generated code in the sandbox.
        if not cap.code:
            raise RuntimeError(f"synthesized capability {name!r} has no executable code")
        sb = await self._sandbox.run(cap.code, args, entrypoint=cap.entrypoint)
        if not sb.ok:
            raise RuntimeError(f"capability {name!r} failed: {sb.error}")
        return sb.result

    async def forget(self, name: str) -> bool:
        """
        Forget a synthesized capability (registry + persisted files) so the next
        request regenerates it. Useful for self-correcting retries when a
        synthesized capability runs but yields a bad result.
        """
        removed = await self._registry.forget_synthesized(name)
        if self._store:
            self._store.delete_capability(name)
        return removed

    async def list_capabilities(self) -> list[dict[str, Any]]:
        """Return all registered + synthesized capabilities as dicts (for a web UI)."""
        all_caps = await self._registry.list_all()
        out = []
        for cap in all_caps:
            entry = {
                "name":        cap.name,
                "kind":        cap.kind.name.lower(),
                "version":     cap.version,
                "description": getattr(cap, "description", ""),
            }
            # Synthesized capabilities expose their generated code
            if hasattr(cap, "code") and cap.code:
                entry["code"] = cap.code
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Run — blocking stdio, MCP-style
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the orchestrator. Blocks on stdin.
        Reads JSON-RPC 2.0 requests, writes responses to stdout.
        Identical workflow to MCP server.run().
        """
        log.info("orchestrator %r starting on stdio", self._name)
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            log.info("orchestrator %r stopped", self._name)

    async def _serve(self) -> None:
        """Async stdio loop."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_event_loop()

        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # Use a direct stdout writer — asyncio.StreamWriter + connect_write_pipe
        # requires a FlowControlMixin protocol which broke in Python 3.13.
        # For stdio, synchronous writes to sys.stdout.buffer are correct and fast.
        writer = _StdioWriter(self._store)

        log.info("stdio connected — waiting for requests")

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    await _write(writer, _error_response(None, -32700, f"Parse error: {exc}"))
                    continue

                # Record the inbound request before dispatching
                if self._store:
                    self._store.log_message("in", request)

                asyncio.create_task(self._handle_request(request, writer))

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("serve loop error: %s", exc)

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    async def _handle_request(
        self,
        request: dict[str, Any],
        writer: _StdioWriter,
    ) -> None:
        method     = request.get("method", "")
        request_id = request.get("id")
        params     = request.get("params", {})

        log.debug("request method=%r id=%r", method, request_id)

        dispatch = {
            "csp.goal.submit":     self._handle_goal_submit,
            "csp.capability.list": self._handle_capability_list,
            "csp.elicit.respond":  self._handle_elicit_respond,
            "csp.ping":            self._handle_ping,
            "initialize":            self._handle_initialize,
        }

        handler = dispatch.get(method)
        if handler is None:
            await _write(writer, _error_response(request_id, -32601, f"Method not found: {method}"))
            return

        try:
            await handler(request_id, params, writer)
        except Exception as exc:
            log.error("handler error method=%r: %s", method, exc, exc_info=True)
            await _write(writer, _error_response(request_id, -32603, f"Internal error: {exc}"))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_initialize(
        self,
        request_id: Any,
        params: dict[str, Any],
        writer: _StdioWriter,
    ) -> None:
        """MCP-compatible initialize handshake."""
        await _write(writer, {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "name":    self._name,
                "version": self._version,
                "capabilities": {
                    "goal_submission": True,
                    "capability_synthesis": True,
                    "elicitation": True,
                    "streaming": True,
                },
            },
        })

    async def _handle_ping(
        self,
        request_id: Any,
        params: dict[str, Any],
        writer: _StdioWriter,
    ) -> None:
        await _write(writer, _ok_response(request_id, {"pong": True}))

    async def _handle_capability_list(
        self,
        request_id: Any,
        params: dict[str, Any],
        writer: _StdioWriter,
    ) -> None:
        all_caps = await self._registry.list_all()
        await _write(writer, _ok_response(request_id, {
            "capabilities": [
                {
                    "name":    cap.name,
                    "kind":    cap.kind.name.lower(),
                    "version": cap.version,
                    "description": getattr(cap, "description", ""),
                }
                for cap in all_caps
            ]
        }))

    async def _handle_goal_submit(
        self,
        request_id: Any,
        params: dict[str, Any],
        writer: _StdioWriter,
    ) -> None:
        """
        Core handler — plan + execute a goal, stream events back.

        Flow:
          1. Acknowledge receipt
          2. Stream PLANNING event
          3. Call planner → ExecutionPlan
          4. Stream each executor event (CAPABILITY, LOG, CAPABILITY_END)
          5. LLM summarizes result
          6. Stream final RESULT
        """
        goal       = params.get("goal", "")
        session_id = params.get("session_id", str(request_id))

        if not goal.strip():
            await _write(writer, _error_response(request_id, -32602, "goal is required"))
            return

        # Acknowledge
        await _write(writer, _ok_response(request_id, {
            "session_id": session_id,
            "status": "planning",
        }))

        # Create elicitation manager for this session
        elicit_mgr = ElicitationManager(timeout=self._elicitation_timeout)
        self._elicitation_managers[session_id] = elicit_mgr

        try:
            # Planning phase
            await _write_notification(writer, "csp.stream.event", {
                "kind":    "PLANNING",
                "message": "Planning your goal...",
            })

            plan = await self._planner.plan(goal)

            # Persist the plan so the developer can inspect what was built
            if self._store:
                self._store.save_plan(goal, plan)

            await _write_notification(writer, "csp.stream.event", {
                "kind":    "PLANNING",
                "message": f"Plan ready: {len(plan.steps)} steps",
                "metadata": {"steps": [s.capability for s in plan.steps]},
            })

            # Execution phase
            executor = Executor(
                registry=self._registry,
                synthesizer=self._synthesizer,
                elicitation_manager=elicit_mgr,
                goal=goal,
                sandbox=self._sandbox,
            )

            result_payload: dict[str, Any] = {}

            async for event in executor.execute(plan):
                method = event.get("method", "")

                if method == "csp.result":
                    # Hold result — summarize first
                    result_payload = event.get("params", {})
                elif method == "csp.elicit.request":
                    # Forward elicitation to client as-is
                    await _write(writer, event)
                else:
                    # Stream progress event
                    await _write(writer, event)

            # LLM summary
            summary = await self._summarize(goal, plan, result_payload)
            result_payload["summary"] = summary

            # Terminal result notification
            await _write_notification(writer, "csp.result", result_payload)

        except Exception as exc:
            log.error("goal execution failed: %s", exc, exc_info=True)
            await _write_notification(writer, "csp.result", {
                "status":  "ERROR",
                "summary": f"Execution failed: {exc}",
                "error":   str(exc),
                "output":  {},
            })

        finally:
            await elicit_mgr.cancel_all()
            self._elicitation_managers.pop(session_id, None)

    async def _handle_elicit_respond(
        self,
        request_id: Any,
        params: dict[str, Any],
        writer: _StdioWriter,
    ) -> None:
        session_id = params.get("session_id", "")
        elicit_id  = params.get("request_id", "")
        value      = params.get("value", "")

        mgr = self._elicitation_managers.get(session_id)
        if mgr is None:
            await _write(writer, _error_response(request_id, -32602, f"Unknown session: {session_id}"))
            return

        resolved = await mgr.respond(elicit_id, value)
        await _write(writer, _ok_response(request_id, {"resolved": resolved}))

    # ------------------------------------------------------------------
    # LLM summary
    # ------------------------------------------------------------------

    async def _summarize(
        self,
        goal: str,
        plan: Any,
        result: dict[str, Any],
    ) -> str:
        """Ask the LLM to produce a natural language summary of execution."""
        steps_run = [s.capability for s in plan.steps]
        status    = result.get("status", "OK")
        output    = result.get("output", {})

        prompt = (
            f"The user asked: {goal!r}\n\n"
            f"Steps executed: {', '.join(steps_run)}\n"
            f"Status: {status}\n"
            f"Output keys: {list(output.keys())}\n\n"
            "Write a concise 1-2 sentence summary of what was accomplished. "
            "Be direct and specific. No preamble."
        )
        try:
            response = await self._llm.complete_once(prompt, max_tokens=200, temperature=0.3)
            return response.content.strip()
        except Exception as exc:
            log.warning("summary LLM call failed: %s", exc)
            return f"Executed {len(steps_run)} steps with status {status}."


# ---------------------------------------------------------------------------
# Stdio writer — replaces asyncio.StreamWriter for stdout
# ---------------------------------------------------------------------------

class _StdioWriter:
    """
    Writes JSON-RPC 2.0 lines directly to sys.stdout.buffer.

    asyncio.StreamWriter + connect_write_pipe requires FlowControlMixin
    which broke in Python 3.13. For a stdio server, synchronous writes
    to the raw stdout buffer are correct: the OS buffers them and they
    never block for the tiny JSON-RPC messages we send.

    If a PlannerStore is attached, every outbound message is also logged
    to planner/jsonrpc.ndjson.
    """

    def __init__(self, store: Optional[PlannerStore] = None) -> None:
        self._store = store

    def log(self, obj: dict[str, Any]) -> None:
        if self._store:
            self._store.log_message("out", obj)

    def write(self, data: bytes) -> None:
        sys.stdout.buffer.write(data)

    async def drain(self) -> None:
        sys.stdout.buffer.flush()


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------

async def _write(writer: _StdioWriter, obj: dict[str, Any]) -> None:
    """Write a single JSON-RPC 2.0 object as one NDJSON line."""
    writer.log(obj)
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    writer.write(line.encode())
    await writer.drain()


async def _write_notification(
    writer: _StdioWriter,
    method: str,
    params: dict[str, Any],
) -> None:
    await _write(writer, {
        "jsonrpc": "2.0",
        "method":  method,
        "params":  params,
    })


def _ok_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }