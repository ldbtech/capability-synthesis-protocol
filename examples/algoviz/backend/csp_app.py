"""
backend/csp_app.py
~~~~~~~~~~~~~~~~~~~
The CSP orchestrator for the Algorithm Visualizer.

There are NO pre-written algorithm capabilities. When the user asks to
visualize quicksort / BFS / binary search / anything, the planner finds that
no such capability exists, and CSP synthesizes one on the fly — real Python
that runs the algorithm step by step and renders each step as a PNG frame.

The synthesis_guidance below is the *contract* the generated code must follow
so the frontend can play the frames as an animation. This lives in the app —
CSP itself knows nothing about algorithms or matplotlib.
"""

from __future__ import annotations

import json
import logging

from csp import Orchestrator, AnthropicLLM

log = logging.getLogger("algoviz.csp_app")

llm = AnthropicLLM()   # ANTHROPIC_API_KEY / ANTHROPIC_MODEL from env

_VIZ_GUIDANCE = """\
This app turns an algorithm into an animation. A synthesized capability runs an
algorithm STEP BY STEP and renders each step as a frame.

Implement the ENTIRE visualization inside ONE capability — run the algorithm
AND render every frame in the same run(args). Never assume another capability
supplies data; if you need input that isn't in args, create it yourself.

Your generated run(args) MUST:
- Read the input from args (e.g. args['data'] for a list to sort/search, or
  args['n'] for a size). If absent, create a small sensible example
  (8-15 elements) so the visualization always works.
- Execute the algorithm, capturing the relevant state after EACH meaningful
  step (a comparison, swap, visit, partition, probe, etc.).
- Render each captured state as a matplotlib figure (the 'Agg' backend is
  already active — do NOT call plt.show()). Use figsize=(7,4), dpi=100.
  Save each figure to an in-memory PNG (io.BytesIO), base64-encode it,
  and append the string to a list.
- Close every figure with plt.close(fig) to free memory.
- Keep total frames <= 60 (sample/stride if needed).

Rendering conventions by algorithm type:
- Sorting (bubble, quick, merge, insertion, selection, heap): draw the array as
  a bar chart; color the bars being compared/swapped differently (e.g. the
  active indices in red/orange, sorted region in green). Title shows the step.
- Searching (binary, linear): bar chart or number line; highlight the current
  search range and the probe index.
- Graph / pathfinding (BFS, DFS, Dijkstra, A*): build a small fixed graph with
  networkx if available else a manual layout; color visited nodes, the current
  frontier, and the final path distinctly. Draw with matplotlib.

Return EXACTLY this dict (plain JSON types only):
{
  "frames": [<base64 png>, ...],   # the animation, in order
  "step_count": <int>,
  "algorithm": "<name>",
  "input": <the input you used>,
  "explanation_hint": "<one line on what the colors mean>"
}
"""

app = Orchestrator(
    "algoviz-server",
    llm=llm,
    planner_dir="planner",
    synthesis_guidance=_VIZ_GUIDANCE,
    sandbox_env={"MPLBACKEND": "Agg"},
    synthesis_timeout=60.0,
)


# ── Registered capabilities ───────────────────────────────────────────────────
# The workflow's reasoning steps are real CSP capabilities, just like the
# synthesized visualizer. The LangGraph nodes call these directly via
# app.call_capability(...). Only the visualizer itself is synthesized at runtime.

@app.capability("understand")
async def understand(request: str = "") -> dict:
    """Normalize a visualization request into a canonical algorithm name and a
    small concrete sample input to run it on."""
    prompt = (
        f"User wants to visualize an algorithm. Request: {request!r}.\n"
        "Reply with ONLY compact JSON: "
        '{"algorithm": "<canonical snake_case name, e.g. quick_sort>", '
        '"sample_input": <a small concrete input: a list of 8-12 ints to sort/'
        'search, or for graphs an adjacency dict>}'
    )
    resp = await llm.complete_once(prompt, max_tokens=400, temperature=0.0)
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"algorithm": request, "sample_input": None}
    return {
        "algorithm":    parsed.get("algorithm", request),
        "sample_input": parsed.get("sample_input"),
    }


@app.capability("narrate")
async def narrate(algorithm: str = "") -> dict:
    """Explain an algorithm in 2-3 plain-language sentences."""
    resp = await llm.complete_once(
        f"In 2-3 short sentences, explain how the {algorithm} algorithm works "
        "for a general audience. No preamble.",
        max_tokens=250,
        temperature=0.4,
    )
    return {"explanation": resp.content.strip()}
