"""
montage-ai/backend/app.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Figma-style canvas powered by CSP.  The canvas state lives in memory as a list
of elements (rect / circle / text / line).  Every user message is sent as a
natural-language goal; CSP plans which capability to run and synthesizes new
ones on the fly when none fit.

Routes
  GET  /api/canvas              — current canvas state
  POST /api/chat                — SSE stream of CSP events + final result
  GET  /api/capabilities        — registered + synthesized caps (with code)
  POST /api/canvas/clear        — reset the canvas
  POST /api/credential          — store an API key {env_key, value}
  GET  /api/credentials         — list which env_keys are stored (no values)

Run:
  cd montage-ai/backend && ../../.venv/bin/python -m uvicorn app:app --reload --port 8002
"""

from __future__ import annotations

import json
import os
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── env (ANTHROPIC_API_KEY / ANTHROPIC_MODEL) ────────────────────────────────
_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV):
    for _line in open(_ENV):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from csp import AnthropicLLM, Orchestrator  # noqa: E402

# ── CSP orchestrator ─────────────────────────────────────────────────────────

SYNTHESIS_GUIDANCE = """
You are synthesizing design capabilities for Montage AI, a vector canvas tool.
Canvas is 1200×700 px with a #f8f9fc background.

Your def run(args) MUST return:
  {
    "elements": [...],   # complete list of ALL canvas elements to display
    "summary":  "..."    # one short sentence describing what was built
  }

════════════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════════════

RULE 1 — PRESERVE EXISTING ELEMENTS ON EDITS:
  args["elements"] contains the CURRENT canvas state (list of existing elements).
  Always start run() with:
      import json
      existing = args.get("elements", [])
      if isinstance(existing, str):
          try: existing = json.loads(existing)
          except: existing = []

  • If the goal is to ADD, EDIT, MOVE, STYLE, or MODIFY something →
    start with `elements = list(existing)` and only add/change what's needed.
  • If the goal is to build a completely NEW design from scratch →
    start with `elements = []`.
  • NEVER discard existing elements unless the goal explicitly says "replace",
    "start over", or "clear everything".

RULE 2 — SELF-CONTAINED FOR EXTERNAL DATA:
  If the goal involves external data (weather, prices, images, APIs):
  • Fetch the data INSIDE run(args) using the requests library. Always use timeout=20.
  • Embed the REAL fetched values directly into the element "text" fields.
  • Return complete canvas elements with live data already in them.
  • NEVER return placeholder text like "N/A" or "{{temperature}}".
  • Do NOT split into a fetch step + separate render steps.
    One capability does both: fetch + build elements + return.

RULE 3 — ONE CAPABILITY PER DATA APP:
  For goals like "build a weather app", "show a dashboard", "display stock data":
  • Synthesize ONE single capability that does EVERYTHING: fetch + compute + render.
  • NEVER split into city_selector + fetch + display + convert — they cannot share state.
  • The capability should accept city/symbol/topic as an arg (defaulting to a sensible value).
  • "Unit toggle", "city selector", "filters" are NOT separate capabilities — they are
    args the user can change by asking again (e.g. "now in Fahrenheit" → evolve the cap).
  • If the planner wants 3+ steps for a data app, collapse them into ONE synthesized cap.

Example for weather app (ONE capability does EVERYTHING):
  def run(args):
      import requests, uuid
      city = args.get("city", "London")
      unit = args.get("unit", "C")   # "C", "F", or "K"
      # 1. Geocode city
      geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1").json()
      lat = geo["results"][0]["latitude"]
      lon = geo["results"][0]["longitude"]
      # 2. Fetch weather
      url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,apparent_temperature&wind_speed_unit=kmh"
      w = requests.get(url).json()["current"]
      temp_c = w["temperature_2m"]
      feels_c = w["apparent_temperature"]
      humidity = w["relative_humidity_2m"]
      wind = w["wind_speed_10m"]
      # 3. Convert units
      if unit == "F":
          temp_disp = f"{temp_c * 9/5 + 32:.1f}°F"
          feels_disp = f"{feels_c * 9/5 + 32:.1f}°F"
      elif unit == "K":
          temp_disp = f"{temp_c + 273.15:.1f}K"
          feels_disp = f"{feels_c + 273.15:.1f}K"
      else:
          temp_disp = f"{temp_c:.1f}°C"
          feels_disp = f"{feels_c:.1f}°C"
      # 4. Heat index warning
      heat_index = -8.78469475556 + 1.61139411*temp_c + 2.33854883889*humidity - 0.14611605*temp_c*humidity ...
      # 5. Build ALL elements and return
      return {"elements": [...], "summary": f"Weather: {city} {temp_disp}"}

════════════════════════════════════════════════
Element schema
════════════════════════════════════════════════
  {
    "id":       str,      # str(uuid.uuid4())
    "type":     "rect" | "circle" | "text" | "line",
    "x":        float,    # pixels from left  (canvas is 1200 wide)
    "y":        float,    # pixels from top   (canvas is 700 tall)
    "w":        float,    # width
    "h":        float,    # height
    "fill":     str,      # hex color e.g. "#6c8cff"
    "stroke":   str,      # hex or "none"
    "opacity":  float,    # 0–1
    "rx":       int,      # border-radius for rect
    "text":     str,      # visible text content (for type="text")
    "fontSize": int,      # px
    "label":    str       # name shown in Layers panel
  }

Design conventions:
  • NEVER use white (#fff) fill on rect — the canvas is already near-white.
    Use #f0f4ff, #e8f5e9, or a coloured accent for card backgrounds.
  • Colour palette: accent #6c8cff, green #36d399, amber #ffb454,
    dark #1b1f2a, muted #8b94a7, card bg #f0f4ff.
  • Text colour on light bg: #1b1f2a. Text on dark bg: #e6e9ef.
  • Reasonable font sizes: heading 32–48px, body 18–24px, caption 14px.
  • Always import uuid at the top of your code block.
  • Centre designs: e.g. a 400px-wide card → x = (1200-400)/2 = 400.
"""

csp_app = Orchestrator(
    "montage-ai",
    llm=AnthropicLLM(),
    planner_dir="planner",
    synthesis_guidance=SYNTHESIS_GUIDANCE,
)

# ── seed capabilities ─────────────────────────────────────────────────────────
# Only clear_canvas is registered. Everything else — layouts, widgets, data
# visualisations — is synthesized on demand as a single self-contained
# capability that fetches data + builds all elements + returns them.
# Keeping add_rectangle / add_text as registered caps caused the planner to
# chain them after synthesized fetch steps, which wiped the real elements.

@csp_app.capability("reset_canvas")
async def reset_canvas(**_) -> dict:
    """Wipe ALL elements — only use when starting a completely new design from scratch."""
    _canvas["elements"].clear()
    return {"elements": [], "summary": "Canvas reset"}


# ── canvas state (in-memory) ──────────────────────────────────────────────────

_canvas: dict = {
    "elements": [],
    "canvas_width": 1200,
    "canvas_height": 700,
}

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Montage AI · CSP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str

class CredentialRequest(BaseModel):
    env_key: str
    value:   str


_TYPE_ALIASES = {
    "rectangle": "rect", "box": "rect", "square": "rect",
    "label": "text", "heading": "text", "paragraph": "text", "caption": "text",
    "ellipse": "circle", "oval": "circle",
    "path": "line", "divider": "line",
}

def _normalize_element(el: dict) -> dict:
    """Normalise a synthesized element so the SVG renderer can draw it."""
    el = dict(el)
    raw_type = str(el.get("type", "rect")).lower()
    el["type"] = _TYPE_ALIASES.get(raw_type, raw_type)
    if el["type"] not in ("rect", "circle", "text", "line"):
        el["type"] = "rect"
    el.setdefault("id", str(uuid.uuid4()))
    el.setdefault("fill", "#6c8cff")
    el.setdefault("stroke", "none")
    el.setdefault("opacity", 1)
    el.setdefault("rx", 0)
    el.setdefault("text", "")
    el.setdefault("fontSize", 18)
    el.setdefault("label", el.get("text") or el["type"])
    el.setdefault("w", 200)
    el.setdefault("h", 60)
    # Cast numeric fields to avoid SVG rendering errors
    for k in ("x", "y", "w", "h", "opacity", "fontSize"):
        try:
            el[k] = float(el[k]) if k != "fontSize" else int(float(el[k]))
        except (TypeError, ValueError):
            el[k] = 0
    return el


@app.get("/api/canvas")
async def get_canvas():
    return _canvas


@app.get("/api/canvas/debug")
async def debug_canvas():
    """Return element count + first element for debugging."""
    els = _canvas["elements"]
    return {
        "count": len(els),
        "types": list({e.get("type") for e in els}),
        "first": els[0] if els else None,
    }


@app.post("/api/canvas/clear")
async def clear():
    _canvas["elements"] = []
    return {"ok": True}


@app.get("/api/capabilities")
async def list_caps():
    caps = await csp_app.list_capabilities()
    return {"capabilities": caps}


@app.post("/api/credential")
async def store_credential(req: CredentialRequest):
    """Store an API key provided by the user via the credential form."""
    csp_app.provide_credential(req.env_key, req.value)
    return {"ok": True, "env_key": req.env_key}


@app.get("/api/credentials")
async def list_credentials():
    """Return which env keys are stored (no values — never expose secrets)."""
    if csp_app._cred_store is None:
        return {"stored": []}
    return {"stored": list(csp_app._cred_store._data.keys())}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def stream():
        ambient = dict(_canvas)
        async for ev in csp_app.submit(req.message, ambient=ambient):
            # Update _canvas BEFORE yielding the event so that when the client
            # receives the result and calls refreshCanvas(), the state is ready.
            if ev.get("type") == "result":
                output = ev.get("output") or {}
                print(f"[DEBUG] result output keys: {list(output.keys())}")
                best: list = []
                for k, v in output.items():
                    if isinstance(v, dict):
                        els_raw = v.get("elements", [])
                        print(f"[DEBUG]   {k}: elements={len(els_raw) if isinstance(els_raw, list) else type(els_raw)}")
                        if isinstance(els_raw, list):
                            els = [_normalize_element(e) for e in els_raw if isinstance(e, dict)]
                            if len(els) > len(best):
                                best = els
                print(f"[DEBUG] best elements count: {len(best)}")
                if best:
                    _canvas["elements"] = best
                    print(f"[DEBUG] canvas updated, first el type: {best[0].get('type')}, x={best[0].get('x')}, y={best[0].get('y')}")
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
