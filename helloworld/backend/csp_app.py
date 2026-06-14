"""
backend/csp_app.py
~~~~~~~~~~~~~~~~~~~
The CSP orchestrator for the CSV-RAG app.

Registers the capabilities a developer would hand-write:
  - chat              : normal conversation
  - answer_from_data  : RAG — retrieve relevant CSV rows, answer with citations
  - describe_dataset  : schema / shape summary of the loaded CSV

Anything the planner asks for that ISN'T one of these (e.g. "median salary by
department", "correlation between age and income") is NOT registered — CSP
synthesizes real Python for it and runs it in the sandbox over the CSV rows
passed in as ambient data. That's the end-to-end magic.
"""

from __future__ import annotations

import logging

from csp import Orchestrator, AnthropicLLM
from csp.llm import LLMMessage

from rag_store import RagStore

log = logging.getLogger("app.csp_app")

# Shared singletons used by the registered capabilities below.
llm   = AnthropicLLM()                 # reads ANTHROPIC_API_KEY / ANTHROPIC_MODEL
store = RagStore()                     # populated when a CSV is uploaded

# Domain conventions for THIS app, handed to the synthesizer so generated
# capabilities match how our data flows and how our UI renders results.
# This lives in the app, not the CSP library — CSP stays domain-agnostic.
_SYNTHESIS_GUIDANCE = """\
This app analyzes tabular CSV data. Synthesized capabilities receive the full
dataset in args['rows'] (a list of dict rows) and args['columns'] (list of
column names). Clean/parse values defensively (strings may contain commas,
currency symbols, units like 'kms').

If the goal asks to plot, chart, graph, visualize, draw, or show a
distribution/scatter/bar/line/histogram/box/heatmap: render an actual figure
with matplotlib (the non-GUI 'Agg' backend is already active), save it to an
in-memory PNG (io.BytesIO), base64-encode it, and return that string under the
key 'image_base64'. Do NOT return raw coordinate lists for plot requests — the
UI renders the PNG. Call plt.close(fig) when done.
"""

app = Orchestrator(
    "csv-rag-server",
    llm=llm,
    planner_dir="planner",
    synthesis_guidance=_SYNTHESIS_GUIDANCE,
    # Headless matplotlib for any generated plotting code. This is an app
    # concern (our generated code uses matplotlib) — not the CSP library's.
    sandbox_env={"MPLBACKEND": "Agg"},
)


@app.capability("chat")
async def chat(message: str = "") -> dict:
    """Have a normal, friendly conversation. Use for greetings and general questions
    that are NOT about the uploaded dataset."""
    resp = await llm.complete_once(
        message or "Hello",
        system="You are a concise, friendly assistant inside a CSV analysis app.",
        max_tokens=400,
        temperature=0.5,
    )
    return {"answer": resp.content.strip()}


@app.capability("answer_from_data")
async def answer_from_data(question: str = "") -> dict:
    """Answer a LOOKUP question about the uploaded CSV by retrieving the most
    relevant individual rows (semantic RAG) and grounding the answer in them.
    Use ONLY for 'which/who/what/find' questions about specific records
    (e.g. 'who works in Sales', 'find the engineer in Seattle').
    DO NOT use for math, averages, totals, counts, sorting, grouping, or any
    calculation across many rows — those need a computed capability instead."""
    if not store.ready:
        return {"answer": "No dataset is loaded yet. Please upload a CSV first.", "sources": []}

    hits = store.search(question, k=5)
    context = "\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    prompt = (
        f"Dataset rows most relevant to the question:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY the rows above. Cite row numbers like [1], [2]. "
        "If the rows don't contain the answer, say so."
    )
    resp = await llm.complete_once(
        prompt,
        system="You answer questions strictly from the provided dataset rows.",
        max_tokens=500,
        temperature=0.2,
    )
    return {
        "answer":  resp.content.strip(),
        "sources": [{"score": round(h["score"], 3), "row": h["row"]} for h in hits],
    }


@app.capability("describe_dataset")
async def describe_dataset() -> dict:
    """Summarize the loaded dataset: its columns, number of rows, and a sample.
    Use when the user asks what's in the data or what columns exist."""
    if not store.ready:
        return {"summary": "No dataset loaded.", "columns": [], "row_count": 0}
    s = store.summary()
    s["sample"] = store.rows[:3]
    return s
