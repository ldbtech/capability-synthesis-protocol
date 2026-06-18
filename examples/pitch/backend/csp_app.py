"""
pitch/backend/csp_app.py
~~~~~~~~~~~~~~~~~~~~~~~~~
The CSP orchestrator for Pitch — a World Cup copilot.

One ask-box, two modes, one engine:
  • LIVE      → "Group A table", "today's scores", "Argentina squad"
                CSP synthesizes FETCH verbs over a football API.
  • PREDICT   → "who wins Argentina vs France", "simulate the bracket"
                CSP synthesizes COMPUTE verbs that write a real model
                (Elo / Poisson) and run it in the sandbox.

Nothing about football is hand-written here. The ONLY registered capability is
`chat` (small talk / help). Every standings table, squad card, match prediction
and bracket is a GENERAL verb the planner synthesizes once and then reuses —
"Group A" then "Group B" reuses `fetch_standings`; "add goal difference"
evolves it.

Synthesized capabilities return a typed VIEW the frontend renders:
  { "view": "table|cards|bracket|chart|stat", "title": str, "data": {...},
    "summary": str }
"""

from __future__ import annotations

import logging

from csp import Orchestrator, AnthropicLLM

log = logging.getLogger("pitch.csp_app")

llm = AnthropicLLM()  # reads ANTHROPIC_API_KEY / ANTHROPIC_MODEL

# Domain conventions handed to the synthesizer. This lives in the app, not the
# CSP library — CSP stays domain-agnostic.
_SYNTHESIS_GUIDANCE = """\
You synthesize capabilities for PITCH, a World Cup (football/soccer) copilot.
Each capability is a GENERAL, REUSABLE verb and returns ONE typed VIEW the UI
renders. The same verb is invoked again later with different args.

════════════════════════════════════════════════
RETURN CONTRACT — every run(args) returns exactly this dict:
════════════════════════════════════════════════
  {
    "view":    "table" | "cards" | "bracket" | "chart" | "stat",
    "title":   "<short heading>",
    "data":    { ... shape depends on view, see below ... },
    "summary": "<one sentence describing the result, with real numbers>"
  }

VIEW shapes (data field):
  • table   → {"columns": ["Team","P","W","D","L","GD","Pts"],
               "rows": [["Argentina",3,3,0,0,7,9], ...]}
  • cards   → {"cards": [{"title":"Argentina","subtitle":"Group A · 1st",
               "image":"<logo url or ''>",
               "stats":[{"label":"FIFA Rank","value":"1"},
                        {"label":"Form","value":"WWDWL"}]}, ...]}
  • bracket → {"rounds": [{"name":"Round of 16",
               "matches":[{"home":"Argentina","away":"Australia",
                           "homeScore":2,"awayScore":1,"note":"prob 0.78"}]}, ...]}
  • chart   → {"unit":"goals","bars":[{"label":"Mbappé","value":6},
               {"label":"Messi","value":5}]}   # bars rendered as a bar chart
  • stat    → {"value":"2.7","label":"Expected goals","sub":"Argentina vs France"}

════════════════════════════════════════════════
RULE 1 — ONE GENERAL VERB, SPECIFICS IN ARGS
════════════════════════════════════════════════
  Read EVERY specific from args (group, team, date, home, away, metric) with
  sensible defaults. Never hardcode the triggering request. "Group A" and
  "Group B" must be the SAME capability with arg group="A" vs "B".
  params_schema is the full interface for the class, designed first.

════════════════════════════════════════════════
RULE 2 — LIVE DATA: fetch real values from football-data.org
════════════════════════════════════════════════
  For standings / fixtures / matches / scorers, use football-data.org v4 — it has
  real World Cup data. Read the key from os.environ["FOOTBALL_DATA_API_KEY"] and
  send it as the header  {"X-Auth-Token": key}.  Fetch inside run(args) with the
  `requests` library, timeout=20. Useful endpoints (competition code "WC"):
    GET https://api.football-data.org/v4/competitions/WC/standings
    GET https://api.football-data.org/v4/competitions/WC/matches?status=SCORED
    GET https://api.football-data.org/v4/competitions/WC/scorers
  The standings response groups teams under standings[].group ("GROUP_A" …) with
  table[] rows carrying team.name, playedGames, won, draw, lost, goalDifference,
  points. Map the requested group letter to "GROUP_<letter>".

  You MUST declare the key in a ##CREDENTIALS block so the UI can collect it:
      ##CREDENTIALS
      FOOTBALL_DATA_API_KEY: football-data.org · get at https://www.football-data.org/client/register

  If the key is missing the orchestrator gates execution and asks the user — do
  NOT hardcode a key. Handle HTTP errors / missing data gracefully: return a
  stat view whose summary explains what went wrong, never crash.

════════════════════════════════════════════════
RULE 3 — PREDICTIONS: write a REAL model, no fabrication
════════════════════════════════════════════════
  For "who wins", "simulate", "favorites": build an actual model in code —
  e.g. Elo from FIFA ranking / recent results, or a Poisson goals model — and
  compute probabilities. Show the method in the summary. Use random with a
  fixed seed for reproducible simulations. Never invent a scoreline without a
  model behind it.

Reusability beats specificity: a wrong-but-narrow capability is worse than a
general one. Always import uuid/requests/math as needed at the top of run().
"""

app = Orchestrator(
    "pitch-server",
    llm=llm,
    planner_dir="planner",
    synthesis_guidance=_SYNTHESIS_GUIDANCE,
)


@app.capability("chat")
async def chat(message: str = "") -> dict:
    """Friendly small talk / help about what Pitch can do. Use ONLY for
    greetings or meta questions ('what can you do?'), NEVER for anything that
    needs football data, standings, fixtures, squads, stats or predictions —
    those are synthesized capabilities."""
    resp = await llm.complete_once(
        message or "Hello",
        system=(
            "You are Pitch, a concise World Cup copilot. You can pull live "
            "standings/fixtures/squads and build match predictions and bracket "
            "simulations. Keep replies short and suggest a concrete example."
        ),
        max_tokens=300,
        temperature=0.5,
    )
    return {"view": "stat", "title": "Pitch",
            "data": {"value": "⚽", "label": resp.content.strip(), "sub": ""},
            "summary": resp.content.strip()}
