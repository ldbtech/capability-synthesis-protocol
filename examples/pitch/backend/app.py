"""
pitch/backend/app.py
~~~~~~~~~~~~~~~~~~~~~
FastAPI server exposing the Pitch CSP orchestrator to the React frontend.

Routes
  GET  /api/board         — current rendered views (typed: table/cards/...)
  POST /api/chat          — SSE stream of CSP events; views applied to the board
  GET  /api/capabilities  — registered + synthesized caps (with code)
  POST /api/board/clear   — empty the board
  POST /api/credential    — store an API key {env_key, value}
  GET  /api/credentials   — which env keys are stored (no values)

Run:
  cd pitch/backend && ../../.venv/bin/python -m uvicorn app:app --reload --port 8003
"""

from __future__ import annotations

import json
import os

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

from csp_app import app as csp  # noqa: E402

# ── board state (in-memory) ──────────────────────────────────────────────────
# Each "view" is a typed dict {view, title, data, summary} produced by a
# synthesized capability. A single goal may produce several (e.g. a comparison
# returning a table + a chart).
_board: dict = {"views": []}

_VIEW_KINDS = {"table", "cards", "bracket", "chart", "stat"}


def _extract_views(output: dict) -> list[dict]:
    """Pull every typed view out of a result's per-capability outputs."""
    views = []
    for _name, val in (output or {}).items():
        if isinstance(val, dict) and val.get("view") in _VIEW_KINDS:
            views.append({
                "view":    val["view"],
                "title":   val.get("title", ""),
                "data":    val.get("data", {}),
                "summary": val.get("summary", ""),
            })
    return views


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Pitch · World Cup copilot · CSP")
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


@app.get("/api/board")
async def get_board():
    return _board


@app.post("/api/board/clear")
async def clear_board():
    _board["views"] = []
    return {"ok": True}


@app.get("/api/capabilities")
async def list_caps():
    return {"capabilities": await csp.list_capabilities()}


@app.post("/api/credential")
async def store_credential(req: CredentialRequest):
    csp.provide_credential(req.env_key, req.value)
    return {"ok": True, "env_key": req.env_key}


@app.get("/api/credentials")
async def list_credentials():
    if getattr(csp, "_cred_store", None) is None:
        return {"stored": []}
    return {"stored": list(csp._cred_store._data.keys())}


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def stream():
        async for ev in csp.submit(req.message):
            # Apply views to the board BEFORE yielding the result event, so the
            # client's refresh after the stream sees up-to-date state.
            if ev.get("type") == "result":
                views = _extract_views(ev.get("output") or {})
                if views:
                    _board["views"] = views
            yield f"data: {json.dumps(ev, default=str)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
