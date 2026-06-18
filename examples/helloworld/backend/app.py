"""
backend/app.py
~~~~~~~~~~~~~~
FastAPI server that exposes the CSP orchestrator to the React frontend.

Routes:
  POST /api/upload     — upload a CSV, build the RAG index
  GET  /api/dataset    — current dataset summary
  GET  /api/capabilities — registered + synthesized capabilities (with code)
  POST /api/chat       — submit a message, stream CSP events back as SSE

Run:
  cd backend && uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import os

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load .env (ANTHROPIC_API_KEY / ANTHROPIC_MODEL) before importing csp_app
_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV):
    for line in open(_ENV):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from csp_app import app as csp, store   # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app.server")

api = FastAPI(title="CSV-RAG + CSP")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@api.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    df  = pd.read_csv(io.BytesIO(raw))
    store.index_csv(df, file.filename)
    return store.summary()


@api.get("/api/dataset")
async def dataset():
    return store.summary()


@api.get("/api/capabilities")
async def capabilities():
    return {"capabilities": await csp.list_capabilities()}


@api.post("/api/describe")
async def describe():
    """
    Borrow the existing `describe_dataset` capability and invoke it directly —
    no planner, no LLM. Demonstrates CSP's Rust-like borrowing: we reuse a
    capability that already exists instead of routing a goal through the planner.
    """
    async with csp.borrow("describe_dataset") as cap:
        return {"borrowed": cap.name, "result": await cap.invoke()}


@api.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream CSP execution events to the browser as Server-Sent Events."""

    # Ambient data: hand the CSV rows to synthesized capabilities so generated
    # code can compute over the real data. Capped to keep payloads sane.
    ambient = {}
    if store.ready:
        ambient = {"rows": store.rows[:5000], "columns": store.columns}

    async def event_stream():
        async for ev in csp.submit(req.message, ambient=ambient):
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@api.get("/api/health")
async def health():
    return {"ok": True}
