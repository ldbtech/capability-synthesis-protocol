"""
backend/app.py
~~~~~~~~~~~~~~
FastAPI server for the Algorithm Visualizer (port 8001).

Routes:
  POST /api/visualize  — run the LangGraph workflow for a request, streaming
                         every node + CSP event (including the freshly
                         synthesized code) and the final animation as SSE.
  GET  /api/capabilities — capabilities CSP has invented so far (with code).
  GET  /api/health

Run:
  cd backend && ../../.venv/bin/python -m uvicorn app:api --reload --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load .env before importing the graph/CSP app
_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV):
    for line in open(_ENV):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import graph as gmod          # noqa: E402
from csp_app import app as csp  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("algoviz.server")

api = FastAPI(title="Algorithm Visualizer · CSP + LangGraph")
api.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class VizRequest(BaseModel):
    request: str


@api.post("/api/visualize")
async def visualize(req: VizRequest):
    queue: asyncio.Queue = asyncio.Queue()

    async def sink(event: dict) -> None:
        await queue.put(event)

    async def run():
        token = gmod.emitter.set(sink)
        try:
            await gmod.run_visualization(req.request)
        except Exception as exc:                       # surface errors to the UI
            log.exception("visualization failed")
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            gmod.emitter.reset(token)
            await queue.put(None)                      # sentinel

    task = asyncio.create_task(run())

    async def stream():
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream")


@api.get("/api/capabilities")
async def capabilities():
    return {"capabilities": await csp.list_capabilities()}


@api.get("/api/health")
async def health():
    return {"ok": True}
