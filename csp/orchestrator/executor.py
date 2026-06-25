"""
csp.orchestrator.executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Executor — walks an ExecutionPlan, runs each step, streams events back.

For each step:
  - If registered: calls the developer's Python function directly.
  - If needs_synthesis: calls the synthesizer → stores the spec in the
    registry → runs the generated Python in the sandbox.
  - If synthesized (already in registry): runs its generated Python in
    the sandbox.

Events are yielded as JSON-RPC 2.0 notification dicts.
The server writes these to stdout (stdio transport) or the WebSocket.

Elicitations: if a registered capability raises ElicitRequired, the
executor yields an elicitation event and awaits the response before
continuing.

Synthesized execution: the capability's human-readable steps are streamed
as LOG events, then its generated `run(args)` code is executed in a
sandboxed subprocess and the real return value becomes the step output.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional

from .capability import (
    AnyCapability,
    RegisteredCapability,
    SynthesizedCapability,
)
from .elicitation import ElicitationManager, ElicitationRequest
from .planner import ExecutionPlan, PlanStep
from .registry import CapabilityRegistry
from .repair import (
    Hole,
    classify_error,
    regrowth_context,
    repair_instruction,
)
from .sandbox import PythonSandbox
from .synthesizer import Synthesizer

log = logging.getLogger("csp.executor")

# Delay between streamed progress steps (seconds)
_STEP_DELAY = 0.15


class ElicitRequired(Exception):
    """
    Raise this inside a registered capability to pause execution
    and ask the user a question.

        raise ElicitRequired(
            kind="approval",
            question="Deploy to production?",
        )
    """
    def __init__(
        self,
        kind: str,
        question: str,
        *,
        options: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> None:
        self.kind     = kind
        self.question = question
        self.options  = options or []
        self.context  = context
        super().__init__(question)


class Executor:
    """
    Executes an ExecutionPlan step by step, yielding JSON-RPC 2.0
    notification dicts for streaming to the client.

    Parameters
    ----------
    registry:
        Capability registry for lookup.
    synthesizer:
        Synthesizer for on-demand capability generation.
    elicitation_manager:
        Manages human-in-the-loop pauses for this session.
    goal:
        Original user goal — passed to synthesizer for context.
    """

    __slots__ = (
        "_registry", "_synthesizer", "_elicitation", "_goal", "_sandbox",
        "_max_repair", "_collision_limit", "_max_regrowths",
    )

    def __init__(
        self,
        registry: CapabilityRegistry,
        synthesizer: Synthesizer,
        elicitation_manager: ElicitationManager,
        goal: str,
        sandbox: Optional[PythonSandbox] = None,
        *,
        max_repair_attempts: int = 3,
        repair_collision_limit: int = 2,
        max_regrowths: int = 1,
    ) -> None:
        self._registry     = registry
        self._synthesizer  = synthesizer
        self._elicitation  = elicitation_manager
        self._goal         = goal
        self._sandbox      = sandbox or PythonSandbox()
        # Pac-Man self-repair budget (see csp.orchestrator.repair):
        #   _max_repair      — M, total heal attempts before giving up
        #   _collision_limit — C, same-hole hits that trigger a warp (regrow)
        #   _max_regrowths   — R, from-scratch regenerations allowed
        # Termination: O(H·C·(R+1)) LLM calls, bounded by M.
        self._max_repair      = max_repair_attempts
        self._collision_limit = repair_collision_limit
        self._max_regrowths   = max_regrowths

    async def execute(
        self,
        plan: ExecutionPlan,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Walk the plan and yield JSON-RPC 2.0 notification dicts.

        Yields:
          csp.stream.event   — progress events
          csp.elicit.request — human input needed
          csp.result         — terminal result
        """
        t_start     = time.monotonic()
        all_outputs: dict[str, Any] = {}
        errors:      list[str]      = []

        for step in plan.steps:
            async for event in self._execute_step(step, all_outputs):
                # Record any step that ended in failure (after self-repair has
                # had its chance) so the terminal result reflects it — without
                # this, a failed step still reported status OK.
                p = event.get("params", {})
                if p.get("kind") == "CAPABILITY_END" and not p.get("metadata", {}).get("success", True):
                    errors.append(p["metadata"].get("error") or f"{step.capability} failed")
                yield event

        # Final result
        duration = time.monotonic() - t_start
        status   = "ERROR" if errors else "OK"

        yield _result_notification(
            status=status,
            output=all_outputs,
            duration=duration,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(
        self,
        step: PlanStep,
        all_outputs: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:

        # Announce step start
        yield _event("CAPABILITY", f"Starting: {step.description or step.capability}", step.capability)

        # Synthesize if needed
        if step.needs_synthesis:
            yield _event("LOG", f"Synthesizing capability: {step.capability}", step.capability)
            try:
                cap = await self._synthesizer.synthesize(
                    capability_name=step.capability,
                    goal=self._goal,
                    context=_args_context(step.args),
                )
                await self._registry.store_synthesized(cap)
                yield _event("LOG", f"Capability synthesized: {step.capability}", step.capability)
            except Exception as exc:
                log.error("synthesis failed for %r: %s", step.capability, exc)
                yield _event("LOG", f"Synthesis failed: {exc}", step.capability)
                yield _capability_end(step.capability, success=False, error=str(exc))
                return

        # Resolve from registry
        cap = await self._registry.resolve(step.capability)
        if cap is None:
            msg = f"Capability not found after synthesis: {step.capability}"
            log.error(msg)
            yield _event("LOG", msg, step.capability)
            yield _capability_end(step.capability, success=False, error=msg)
            return

        # Execute
        t_cap = time.monotonic()
        try:
            if isinstance(cap, RegisteredCapability):
                output = await self._run_registered(cap, step, all_outputs)
            else:
                # Stream the human-readable progress steps, then run the
                # generated code for real in the sandbox.
                async for event in self._stream_steps(cap, step):
                    yield event

                if not cap.code:
                    msg = f"Synthesized capability {cap.name!r} has no executable code"
                    yield _event("LOG", msg, cap.name)
                    yield _capability_end(step.capability, success=False, error=msg)
                    return

                yield _event("LOG", "Running generated code in sandbox...", cap.name)
                # Run the code; if it fails at runtime, the Pac-Man regeneration
                # loop heals it in place (classify → DNA-repair / warp+regrow)
                # before we ever report failure. holder carries the outcome out
                # of the async generator (PEP 525: async gens can't return one).
                holder: dict[str, Any] = {}
                async for ev in self._heal(cap, step, holder):
                    yield ev
                sb  = holder["sb"]
                cap = holder["cap"]          # possibly the healed/regrown version
                if not sb.ok:
                    yield _event("LOG", f"Code error (unhealed): {sb.error}", cap.name)
                    yield _capability_end(step.capability, success=False, error=sb.error)
                    return
                output = sb.result if isinstance(sb.result, dict) else {"result": sb.result}
                yield _event(
                    "LOG",
                    f"Code executed in {sb.duration:.2f}s",
                    cap.name,
                    metadata={"sandbox_duration": sb.duration},
                )

            all_outputs[step.capability] = output
            duration = time.monotonic() - t_cap
            yield _capability_end(step.capability, success=True, output=output, duration=duration)

        except ElicitRequired as exc:
            # Pause — ask the user
            async for event in self._handle_elicit(exc, step, all_outputs):
                yield event

        except Exception as exc:
            log.error("capability %r failed: %s", step.capability, exc)
            yield _event("LOG", f"Error: {exc}", step.capability)
            yield _capability_end(step.capability, success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Registered capability execution
    # ------------------------------------------------------------------

    async def _run_registered(
        self,
        cap: RegisteredCapability,
        step: PlanStep,
        all_outputs: dict[str, Any],
    ) -> Any:
        """Invoke the Python function directly."""
        log.debug("invoking registered %r args=%s", cap.name, list(step.args.keys()))
        return await cap.invoke(**step.args)

    # ------------------------------------------------------------------
    # Synthesized capability — stream progress, then run real code
    # ------------------------------------------------------------------

    async def _stream_steps(
        self,
        cap: SynthesizedCapability,
        step: PlanStep,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream a synthesized capability's human-readable progress steps as LOG
        events. The actual computation is the generated code, run separately in
        the sandbox by the caller.
        """
        log.debug("streaming steps for synthesized %r steps=%d", cap.name, len(cap.steps))
        for exec_step in cap.steps:
            yield _event("LOG", exec_step, cap.name)
            await asyncio.sleep(_STEP_DELAY)

    # ------------------------------------------------------------------
    # Pac-Man self-repair — heal a synthesized capability that runs but fails
    # ------------------------------------------------------------------

    async def _heal(
        self,
        cap: SynthesizedCapability,
        step: PlanStep,
        holder: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a synthesized capability's code; if it fails at runtime, heal it.

        The Pac-Man regeneration loop (see csp.orchestrator.repair):
          1. classify the traceback into a pigeonhole (error class)
          2. record it on the scent trail; count collisions per hole
          3. same hole hit C times  → WARP: apoptosis + regrow (synthesize fresh)
             otherwise               → DNA repair: evolve() the code in place
          4. re-run; on green, persist the healed code and stop
          5. bounded by M attempts / R regrowths → provably terminates

        Only GREEN code is written back to the registry, so a capability that
        has never executed successfully is never cached for reuse.

        Streams a LOG per move; writes the final {sb, cap} into `holder`.
        """
        sb = await self._sandbox.run(cap.code, step.args, entrypoint=cap.entrypoint)
        if sb.ok:
            holder["sb"], holder["cap"] = sb, cap
            return

        trail: list[tuple[Hole, Optional[str]]] = []
        collisions: dict[Hole, int] = {}
        regrowths = 0

        for attempt in range(1, self._max_repair + 1):
            hole = classify_error(sb.error, sb.traceback)

            # A missing credential is not a code bug — let the credential gate
            # handle it. Bail out of repair with the original failure intact.
            if hole is Hole.CREDENTIAL:
                yield _event("LOG", f"⛔ repair skipped: {cap.name} needs a credential, not a fix", cap.name)
                break

            trail.append((hole, sb.error))
            collisions[hole] = collisions.get(hole, 0) + 1

            warp = collisions[hole] >= self._collision_limit and regrowths < self._max_regrowths
            if warp:
                regrowths += 1
                collisions.clear()           # fresh maze after a warp
                yield _event(
                    "LOG",
                    f"🔵 warp {regrowths}/{self._max_regrowths}: regrowing {cap.name} "
                    f"from scratch (hole={hole.value} cornered)",
                    cap.name,
                )
                candidate = await self._synthesizer.synthesize(
                    capability_name=cap.name,
                    goal=self._goal,
                    context=regrowth_context(step.args, trail),
                )
            else:
                yield _event(
                    "LOG",
                    f"🟡 repair {attempt}/{self._max_repair}: hole={hole.value}",
                    cap.name,
                )
                candidate = await self._synthesizer.evolve(
                    cap,
                    repair_instruction(hole, sb.error, sb.traceback, step.args, trail),
                )

            sb  = await self._sandbox.run(candidate.code, step.args, entrypoint=candidate.entrypoint)
            cap = candidate                  # carry forward; next patch builds on this

            if sb.ok:
                await self._registry.store_synthesized(cap)   # persist only green code
                yield _event(
                    "LOG",
                    f"🟢 healed {cap.name} after {attempt} attempt(s)",
                    cap.name,
                    metadata={"repair_attempts": attempt, "regrowths": regrowths},
                )
                holder["sb"], holder["cap"] = sb, cap
                return

        # Budget exhausted (or credential bail-out): hand back the last failure.
        holder["sb"], holder["cap"] = sb, cap

    # ------------------------------------------------------------------
    # Elicitation handling
    # ------------------------------------------------------------------

    async def _handle_elicit(
        self,
        exc: ElicitRequired,
        step: PlanStep,
        all_outputs: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Pause execution, ask the user, resume."""
        elicit_req, fut = await self._elicitation.request(
            kind=exc.kind,
            question=exc.question,
            options=exc.options or None,
            context=exc.context,
            capability=step.capability,
        )

        # Send elicitation to client
        yield elicit_req.to_jsonrpc()

        # Wait for response
        try:
            answer = await asyncio.wait_for(
                asyncio.shield(fut),
                timeout=120.0,
            )
            log.debug("elicitation answered: %r", answer[:40])
            # Re-run the step with the answer injected into args
            step.args["_elicit_response"] = answer
            async for event in self._execute_step(step, all_outputs):
                yield event

        except asyncio.TimeoutError:
            yield _event("LOG", f"Elicitation timed out for {step.capability}", step.capability)
            yield _capability_end(step.capability, success=False, error="elicitation timeout")


def _args_context(args: dict[str, Any]) -> str:
    """
    Describe the actual args dict the generated code will receive, so the
    synthesizer reads the right keys (e.g. args['rows']) instead of guessing.
    """
    if not args:
        return "The args dict passed to run(args) will be empty."

    lines = ["The run(args) function will receive args with these keys:"]
    for key, val in args.items():
        if isinstance(val, list):
            sample = val[0] if val else None
            if isinstance(sample, dict):
                cols = list(sample.keys())
                lines.append(
                    f"- args[{key!r}]: list of {len(val)} dict rows; "
                    f"each row has keys {cols}. Example row: {sample}"
                )
            else:
                lines.append(f"- args[{key!r}]: list of {len(val)} items, e.g. {sample!r}")
        elif isinstance(val, dict):
            lines.append(f"- args[{key!r}]: dict with keys {list(val.keys())}")
        else:
            lines.append(f"- args[{key!r}]: {type(val).__name__} = {val!r}")
    lines.append("Read inputs from these exact keys. Return a JSON-serializable dict.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 notification builders
# ---------------------------------------------------------------------------

def _event(
    kind: str,
    message: str,
    capability: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "csp.stream.event",
        "params": {
            "kind":       kind,
            "message":    message,
            "capability": capability,
            "metadata":   metadata or {},
        },
    }


def _capability_end(
    capability: str,
    *,
    success: bool,
    output: Optional[Any] = None,
    error: Optional[str] = None,
    duration: Optional[float] = None,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "csp.stream.event",
        "params": {
            "kind":       "CAPABILITY_END",
            "message":    f"{'Completed' if success else 'Failed'}: {capability}",
            "capability": capability,
            "metadata": {
                "success":  success,
                "output":   output,
                "error":    error,
                "duration": duration,
            },
        },
    }


def _result_notification(
    status: str,
    output: dict[str, Any],
    duration: float,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "csp.result",
        "params": {
            "status":   status,
            "output":   output,
            "duration": duration,
            "errors":   errors,
        },
    }