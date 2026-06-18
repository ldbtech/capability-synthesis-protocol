"""
Ad-hoc end-to-end test: does CSP now REUSE a general capability instead of
synthesizing a new one for every new plot/aggregation request?

Drives the real orchestrator (real LLM) over employees.csv with several
different goals and prints, per goal, which capability the planner chose and
whether it was synthesized fresh or reused.

Run:
    cd helloworld/backend && ../../.venv/bin/python test_reuse.py
"""
from __future__ import annotations

import asyncio
import csv
import os

# Load .env exactly like app.py does
_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV):
    for line in open(_ENV):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from csp_app import app  # noqa: E402

CSV = os.path.join(os.path.dirname(__file__), "..", "sample_data", "employees.csv")


def load_rows():
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    return rows, list(rows[0].keys())


GOALS = [
    "plot the average salary by department as a bar chart",
    "plot the distribution of age as a histogram",
    "plot salary versus years_experience as a scatter plot",
    "what is the median salary by city",          # aggregation, different verb
    "what is the average age by department",       # should REUSE the aggregation cap
]


async def run_goal(goal: str, rows, columns):
    """Submit one goal; return (chosen_capability, was_synthesized)."""
    ambient = {"rows": rows, "columns": columns}
    chosen = None
    synthesized_names: list[str] = []
    plan_caps: list[str] = []
    async for ev in app.submit(goal, ambient=ambient):
        if ev.get("type") == "planning" and "steps" in ev:
            plan_caps = ev["steps"]
        msg = ev.get("message", "")
        if ev.get("kind") == "LOG" and msg.startswith("Synthesizing capability:"):
            synthesized_names.append(msg.split(":", 1)[1].strip())
    # the "real work" cap is the last non-housekeeping step
    work = [c for c in plan_caps if c not in ("reset_canvas", "chat")]
    chosen = work[-1] if work else (plan_caps[-1] if plan_caps else "?")
    return chosen, synthesized_names


async def main():
    rows, columns = load_rows()
    print(f"Loaded {len(rows)} rows, columns={columns}\n")

    synthesized_ever: set[str] = set()
    for i, goal in enumerate(GOALS, 1):
        chosen, synth = await run_goal(goal, rows, columns)
        synthesized_ever.update(synth)
        was_new = bool(synth)
        tag = f"🆕 SYNTHESIZED {synth}" if was_new else "♻️  REUSED (no synthesis)"
        print(f"[{i}] {goal}")
        print(f"     → capability: {chosen!r}   {tag}\n")

    print("=" * 64)
    print(f"Total distinct capabilities synthesized: {len(synthesized_ever)}")
    print(f"  {sorted(synthesized_ever)}")
    print("\nEXPECTED: a small number of GENERAL verbs (e.g. plot_chart,")
    print("aggregate_table) reused across goals — NOT one cap per goal.")


if __name__ == "__main__":
    asyncio.run(main())
