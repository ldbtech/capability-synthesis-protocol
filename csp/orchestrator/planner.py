"""
csp.orchestrator.planner
~~~~~~~~~~~~~~~~~~~~~~~~~~
Planner — converts a user goal into an ordered execution graph.

The execution graph is a list of PlanStep objects. Each step names a
capability to invoke, with its input arguments. The planner reasons
over:

  - The user's goal
  - Available registered capabilities (from registry)
  - Previously synthesized capabilities (from registry)
  - Any resource context the developer registered

For capabilities that don't exist yet, the planner marks them as
needs_synthesis=True. The executor will call the synthesizer before
running those steps.

Output is always an ExecutionPlan — a validated, ordered list of steps.
The LLM is prompted to return ONLY JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm.base import BaseLLM, LLMMessage
from .registry import CapabilityRegistry

log = logging.getLogger("csp.planner")

# ---------------------------------------------------------------------------
# Plan types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PlanStep:
    """A single step in an execution plan."""
    capability:      str                   # capability name to invoke
    args:            dict[str, Any]        = field(default_factory=dict)
    needs_synthesis: bool                  = False   # True if not in registry
    description:     str                   = ""      # human-readable label

    def __repr__(self) -> str:
        synth = " [needs_synthesis]" if self.needs_synthesis else ""
        return f"<PlanStep {self.capability!r}{synth} args={list(self.args.keys())}>"


@dataclass(slots=True)
class ExecutionPlan:
    """Ordered list of steps produced by the planner."""
    goal:  str
    steps: list[PlanStep]          = field(default_factory=list)
    notes: str                     = ""   # planner reasoning notes

    def __repr__(self) -> str:
        return f"<ExecutionPlan steps={len(self.steps)} goal={self.goal!r:.60}>"


# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are the CSP execution planner.
Your job is to break a user goal into an ordered list of capability invocations.

You will be given:
- The user's goal
- A list of available capabilities (registered or previously synthesized)

CAPABILITIES ARE GENERAL, REUSABLE VERBS — NOT ONE-OFF TASKS.
A capability like `plot_chart`, `aggregate_table`, or `call_rest_api` handles an
ENTIRE CLASS of work. The specifics of THIS request — which columns, which chart
kind, which URL, which aggregation — are passed as ARGS, never baked into a new
capability. Two different plots are two invocations of ONE `plot_chart`. Your job
is to choose the right verb and supply the right args.

Rules:
1. REUSE FIRST. Before creating anything, scan the available capabilities. If an
   existing capability's general function covers this goal — EVEN IF it was first
   created for a different specific request — REUSE it by passing new args. Match
   on the capability's FUNCTION/CATEGORY (plotting, aggregation, lookup, API call,
   transformation, ranking), not on whether its past use matches this exact task.
   When you reuse, set needs_synthesis: false and fill `args` to match that
   capability's listed params (shown as "| params: ..."), inferring values from
   the current goal.
2. NAME NEW CAPABILITIES AS GENERAL VERBS. When nothing fits and you must create
   one, give it a general, reusable, snake_case name describing the CLASS of
   operation (plot_chart, aggregate_table, filter_rows, call_rest_api,
   rank_top_n, transform_columns) — NEVER a task-specific name like
   plot_revenue_by_quarter or average_salary_by_department. Put every specific in
   `args`. This is what lets the NEXT similar goal reuse it instead of
   regenerating. A reusable general capability beats a precise narrow one.

   A VARIATION IS AN ARG, NOT A NEW CAPABILITY. These are all the SAME verb:
     • bar / line / scatter / histogram / pie / box  → ONE `plot_chart`, arg kind=...
       (do NOT make plot_histogram, plot_scatter, plot_bar — they are kind=...)
     • mean / median / sum / count / min / max        → ONE `aggregate_table`, arg agg=...
     • asc / desc, top-N                              → ONE `rank_top_n`, args order=, n=
     • GET / POST, different endpoints                → ONE `call_rest_api`, args method=, url=
   If two goals differ ONLY by such a knob, they MUST resolve to the same
   capability with different args. Keep distinct verbs separate though:
   aggregation (compute numbers) and plotting (draw a figure) are different verbs.

   INVERSE — A DIFFERENT FUNCTION IS A DIFFERENT VERB, even in the same domain.
   Reuse matches on FUNCTION, not on TOPIC/DOMAIN. Do NOT force a goal onto an
   existing capability just because they share a subject. Example (football):
   `fetch_standings`, `predict_match`, `fetch_squad`, `leaderboard` are FOUR
   different verbs — "win probability for A vs B" is a PREDICTION, NOT a variation
   of a standings table. If the only existing capability does a different function
   than the goal needs, SYNTHESIZE A NEW general verb (needs_synthesis: true) —
   never twist an unrelated capability to fit.
3. Mark a step needs_synthesis: true ONLY when no existing capability's category
   covers the goal. Prefer ONE general capability invoked with rich args over
   several narrow ones.
4. Keep the plan minimal — only steps genuinely needed to achieve the goal.
5. Each step must have a capability name in snake_case.
6. Args should be inferred from the goal context. When reusing a capability, pass
   the full set of knobs its params list declares (e.g. kind, x, y, agg). Do NOT
   invent large data arguments — bulk data (e.g. dataset rows) is injected
   automatically.
7. CRITICAL — DATA APP RULE: For goals like "build a weather app", "show a dashboard",
   "display stock data", "create a chart", or any goal involving fetching + displaying
   data, plan at MOST 2 steps: optionally reset_canvas + ONE synthesized capability
   that does everything (fetch + compute + render). NEVER split into city_selector,
   fetch_data, display_info, convert_units — those separate caps cannot share state
   between sandbox executions. The single synthesized cap should accept parameters
   like city, unit, symbol as args.

You MUST respond with ONLY valid JSON. No prose. No markdown. No backticks.

Response format:
{
  "steps": [
    {
      "capability": "<snake_case_name>",
      "args": { "<param>": "<value>" },
      "needs_synthesis": false,
      "description": "<one line: what this step does>"
    }
  ],
  "notes": "<optional: brief planner reasoning>"
}
"""


class Planner:
    """
    Converts a user goal into an ExecutionPlan.

    Parameters
    ----------
    llm:
        LLM provider.
    registry:
        Capability registry — planner reads it to know what exists.
    max_retries:
        Retry count on JSON parse failure.
    """

    __slots__ = ("_llm", "_registry", "_max_retries")

    def __init__(
        self,
        llm: BaseLLM,
        registry: CapabilityRegistry,
        *,
        max_retries: int = 2,
    ) -> None:
        self._llm         = llm
        self._registry    = registry
        self._max_retries = max_retries

    async def plan(self, goal: str) -> ExecutionPlan:
        """
        Generate an ExecutionPlan for the given goal.

        Marks steps as needs_synthesis if the capability isn't in registry.
        """
        capabilities_summary = await self._registry.summary_for_planner()
        prompt = _build_prompt(goal, capabilities_summary)

        raw_plan = await self._generate_plan(prompt, goal)
        plan     = _parse_plan(goal, raw_plan, self._registry)

        log.info(
            "plan generated steps=%d synthesize=%d goal=%r",
            len(plan.steps),
            sum(1 for s in plan.steps if s.needs_synthesis),
            goal[:60],
        )
        return plan

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _generate_plan(
        self,
        prompt: str,
        goal: str,
    ) -> dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                log.warning("planner retry %d/%d: %s", attempt, self._max_retries, last_error)

            try:
                response = await self._llm.complete(
                    [LLMMessage(role="user", content=prompt)],
                    system=_PLANNER_SYSTEM,
                    temperature=0.0,    # deterministic planning
                    max_tokens=2048,
                )
                raw = _strip_markdown(response.content)
                parsed = json.loads(raw)
                if "steps" not in parsed:
                    raise ValueError("plan missing 'steps' field")
                return parsed

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc

        log.error("planner failed after %d attempts, using fallback", self._max_retries + 1)
        return _fallback_plan(goal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(goal: str, capabilities_summary: str) -> str:
    return (
        f"User goal: {goal}\n\n"
        f"Available capabilities:\n{capabilities_summary}\n\n"
        "Generate the execution plan."
    )


def _strip_markdown(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return cleaned


def _parse_plan(
    goal: str,
    raw: dict[str, Any],
    registry: CapabilityRegistry,
) -> ExecutionPlan:
    """
    Convert raw LLM JSON into an ExecutionPlan.
    Overrides needs_synthesis based on actual registry state.
    """
    steps = []
    for raw_step in raw.get("steps", []):
        cap_name = raw_step.get("capability", "unknown")
        steps.append(
            PlanStep(
                capability=cap_name,
                args=raw_step.get("args", {}),
                # Trust registry as source of truth, not LLM
                needs_synthesis=not registry.exists(cap_name),
                description=raw_step.get("description", ""),
            )
        )
    return ExecutionPlan(
        goal=goal,
        steps=steps,
        notes=raw.get("notes", ""),
    )


def _fallback_plan(goal: str) -> dict[str, Any]:
    """Single-step fallback plan when LLM fails."""
    return {
        "steps": [
            {
                "capability": "execute_goal",
                "args": {"goal": goal},
                "needs_synthesis": True,
                "description": f"Execute: {goal}",
            }
        ],
        "notes": "fallback plan — planner LLM failed",
    }