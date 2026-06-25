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
from .borrow import BorrowScope
from .capability import AnyCapability, RegisteredCapability, SynthesizedCapability, capability_from_function
from .credentials import CredentialRequired, CredentialStore
from .elicitation import ElicitationManager
from .executor import Executor
from .planner import Planner
from .planner_store import PlannerStore
from .registry import CapabilityRegistry
from .selection import SelectionStrategy, TagLexicalStrategy, EmbeddingStrategy
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
        credentials_dir: Optional[str] = "credentials",
        synthesis_guidance: str = "",
        sandbox_env: Optional[dict[str, str]] = None,
        selection: Optional["SelectionStrategy"] = None,
        shortlist_k: int = 12,
        shortlist_threshold: int = 25,
        max_repair_attempts: int = 3,
        repair_collision_limit: int = 2,
        max_regrowths: int = 1,
    ) -> None:
        self._name                = name
        self._version             = version
        self._llm                 = llm
        self._synthesis_timeout   = synthesis_timeout
        self._elicitation_timeout = elicitation_timeout
        # Pac-Man self-repair budget — threaded into every Executor. See
        # csp.orchestrator.repair for the algorithm and termination bound.
        self._max_repair_attempts    = max_repair_attempts
        self._repair_collision_limit = repair_collision_limit
        self._max_regrowths          = max_regrowths

        # Planner store — auto-creates planner/ in the developer's project.
        # Pass planner_dir=None to disable persistence entirely.
        self._store: Optional[PlannerStore] = (
            PlannerStore(planner_dir) if planner_dir else None
        )

        # Selection strategy decides which capabilities the planner sees per
        # goal. Defaults (selection=None) to the dependency-free lexical
        # strategy inside CapabilityRegistry; pass EmbeddingStrategy(...) to opt
        # into semantic retrieval. shortlist_threshold gates when shortlisting
        # kicks in — below it, all capabilities are shown.
        self._registry    = CapabilityRegistry(
            strategy=selection,
            shortlist_k=shortlist_k,
            shortlist_threshold=shortlist_threshold,
        )
        self._synthesizer = Synthesizer(llm, guidance=synthesis_guidance)
        self._planner     = Planner(llm, self._registry)
        self._sandbox     = PythonSandbox(env=sandbox_env)
        self._cred_store  = CredentialStore(credentials_dir) if credentials_dir else None

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

            # ── Pre-synthesis pass ───────────────────────────────────────
            # Synthesize any missing capabilities NOW (before execution) so
            # we can inspect their credential requirements and ask the user
            # before touching the sandbox.
            for step in plan.steps:
                # ── evolve__ built-in: patch an existing synthesized cap ──
                if step.capability.startswith("evolve__"):
                    target = step.capability[len("evolve__"):]
                    instruction = step.args.get("instruction", goal)
                    existing = self._registry._synthesized.get(target)
                    if existing:
                        yield {"type": "event", "kind": "LOG",
                               "message": f"Evolving capability: {target}"}
                        evolved = await self._synthesizer.evolve(existing, instruction)
                        async with self._registry._lock:
                            self._registry._synthesized[target] = evolved
                        if self._store:
                            self._store.save_capability(evolved)
                        yield {"type": "event", "kind": "LOG",
                               "message": f"Capability evolved: {target}"}
                        # Replace the evolve__ step with the real target so it
                        # gets re-executed with the updated code.
                        step.capability = target
                        step.needs_synthesis = False
                        step.description = f"Re-run evolved capability: {target}"
                    else:
                        # Target not found — skip this step entirely
                        step.capability = "__skip__"
                    continue

                if not self._registry.exists(step.capability):
                    yield {"type": "event", "kind": "LOG",
                           "message": f"Synthesizing capability: {step.capability}"}
                    cap = await self._synthesizer.synthesize(
                        step.capability, goal,
                        context=f"existing capabilities: {list(self._registry._registered.keys()) + list(self._registry._synthesized.keys())}"
                    )
                    self._registry._synthesized[cap.name] = cap
                    if self._store:
                        self._store.save_capability(cap)
                    step.needs_synthesis = False
                    yield {"type": "event", "kind": "LOG",
                           "message": f"Capability synthesized: {cap.name}"}

            # ── Credential gate ──────────────────────────────────────────
            # Collect every credential required by synthesized capabilities
            # in this plan that isn't already in the store.
            if self._cred_store:
                missing: dict[str, dict] = {}
                for step in plan.steps:
                    cap = (self._registry._registered.get(step.capability)
                           or self._registry._synthesized.get(step.capability))
                    if isinstance(cap, SynthesizedCapability):
                        for cred in cap.credentials:
                            if not self._cred_store.has(cred["env_key"]):
                                missing[cred["env_key"]] = cred

                if missing:
                    for cred in missing.values():
                        yield {"type": "credential_required", **cred}
                    yield {
                        "type": "result",
                        "status": "PENDING_CREDENTIALS",
                        "summary": (
                            f"Need {len(missing)} credential(s) to continue. "
                            "Fill in the form and re-send your request."
                        ),
                        "output": {},
                        "pending_goal": goal,
                    }
                    return

            executor = Executor(
                registry=self._registry,
                synthesizer=self._synthesizer,
                elicitation_manager=elicit_mgr,
                goal=goal,
                sandbox=self._sandbox,
                max_repair_attempts=self._max_repair_attempts,
                repair_collision_limit=self._repair_collision_limit,
                max_regrowths=self._max_regrowths,
            )

            # Inject all stored credentials into the sandbox env so synthesized
            # code can reach them via os.environ["KEY"].
            if self._cred_store:
                self._sandbox._env.update(self._cred_store.as_env())

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

    async def evolve(self, name: str, instruction: str) -> "SynthesizedCapability":
        """
        Modify an existing synthesized capability based on a natural-language
        instruction.  The existing code is passed to the LLM which patches only
        what the instruction requires and returns the complete updated code.

        The evolved capability replaces the old one in the registry and on disk.

            evolved = await app.evolve(
                "fetch_weather_data",
                "also return humidity and UV index"
            )

        Raises KeyError if the capability doesn't exist or is registered (not
        synthesized) — you can only evolve what CSP wrote itself.
        """
        cap = self._registry._synthesized.get(name)
        if cap is None:
            raise KeyError(
                f"No synthesized capability {name!r} to evolve. "
                "Only CSP-synthesized capabilities can be evolved."
            )

        evolved = await self._synthesizer.evolve(cap, instruction)

        async with self._registry._lock:
            self._registry._synthesized[name] = evolved
        if self._store:
            self._store.save_capability(evolved)

        log.info("capability %r evolved: %s", name, instruction[:60])
        return evolved

    def provide_credential(self, env_key: str, value: str) -> None:
        """
        Store an API credential so synthesized capabilities can use it.

        Called by the app server when the user submits the credential form.
        The key is immediately persisted to disk and injected into the sandbox
        env on the next goal submission.

            app.provide_credential("OPENWEATHER_API_KEY", "abc123")
        """
        if self._cred_store is None:
            raise RuntimeError("credentials_dir is disabled — pass credentials_dir= to Orchestrator")
        self._cred_store.set(env_key, value)

    async def _invoke_resolved(self, cap: AnyCapability, args: dict[str, Any]) -> Any:
        """Run an already-resolved capability (registered fn or sandboxed code)."""
        if isinstance(cap, RegisteredCapability):
            return await cap.invoke(**args)
        if not cap.code:
            raise RuntimeError(f"synthesized capability {cap.name!r} has no executable code")
        sb = await self._sandbox.run(cap.code, args, entrypoint=cap.entrypoint)
        if not sb.ok:
            raise RuntimeError(f"capability {cap.name!r} failed: {sb.error}")
        return sb.result

    def borrow(self, name: str) -> BorrowScope:
        """
        Borrow an EXISTING capability (Rust-like). Returns an async context
        manager yielding a read-only, invokable handle. Borrowing never
        synthesizes — it raises KeyError if the capability doesn't exist. While
        a borrow is live, the capability cannot be forgotten/replaced.

            async with app.borrow("detect_anomalies") as cap:
                result = await cap.invoke(rows=rows)

        Many services can hold shared borrows of the same capability at once.
        """
        return BorrowScope(self._registry, name, self._invoke_resolved)

    async def call_capability(self, name: str, **args: Any) -> Any:
        """
        Invoke a single capability directly by name — no planner, no LLM.

        Counterpart to submit()/run_goal(): when you already know which
        capability you want, call it straight. Mirrors MCP's `tools/call`.
        Internally borrows the capability for the duration of the call, so it
        can't be forgotten mid-invocation. Raises KeyError if it doesn't exist.
        """
        async with self.borrow(name) as cap:
            return await cap.invoke(**args)

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
                max_repair_attempts=self._max_repair_attempts,
                repair_collision_limit=self._repair_collision_limit,
                max_regrowths=self._max_regrowths,
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