# CSV-RAG — a CSP demo app

A small web app that shows CSP end-to-end:

- Upload a **CSV**.
- Ask **lookup** questions → answered with **RAG** (BGE embeddings + retrieval).
- Ask **computational** questions (averages, counts, correlations, …) that no
  registered capability covers → CSP **synthesizes real Python**, runs it in a
  sandbox over your actual rows, and returns the result.
- Every synthesized capability's **generated code is shown in the UI** and saved
  to `planner/capabilities/<name>.py`, so you can verify exactly what ran.

```
helloworld/
├── backend/            FastAPI + CSP orchestrator
│   ├── app.py          HTTP routes + SSE streaming
│   ├── csp_app.py      Orchestrator + registered capabilities (chat, RAG, describe)
│   ├── rag_store.py    in-memory vector store over the CSV
│   └── embeddings.py   local BGE embeddings (sentence-transformers)
├── frontend/           React + Vite UI
├── sample_data/        employees.csv to try
└── .env                ANTHROPIC_API_KEY (+ optional ANTHROPIC_MODEL)
```

## Prerequisites

From the repo root, install the library and Python deps into the venv:

```bash
cd ..                       # csp/ repo root
pip install -e .
pip install sentence-transformers fastapi "uvicorn[standard]" pandas python-multipart matplotlib
```

Set your key in `helloworld/.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

## Run (two terminals)

**1 — backend** (port 8000):

```bash
cd helloworld/backend
../../.venv/bin/python -m uvicorn app:api --reload --port 8000
```

> Use the venv's Python explicitly (`../../.venv/bin/python -m uvicorn`). A bare
> `uvicorn` may resolve to a different global Python with incompatible
> numpy/pandas, even when the venv looks active.

The first request downloads the BGE model (~one time).

**2 — frontend** (port 5173, proxies `/api` to the backend):

```bash
cd helloworld/frontend
npm install
npm run dev
```

Open http://localhost:5173, upload `sample_data/employees.csv`, and try:

| Question | What CSP does |
|---|---|
| "Who works in Data Science?" | RAG → `answer_from_data` |
| "What columns are in the data?" | `describe_dataset` |
| "Average salary per department" | synthesizes + runs `average_salary_by_department` |
| "Correlation between age and salary" | synthesizes + runs `correlation_between_age_and_salary` |

After a synthesized question, open the capability in the sidebar to read the
generated Python, or look in `backend/planner/capabilities/`.

**Borrowing:** the **🔗 Describe (borrows capability)** button calls
`POST /api/describe`, which does `async with csp.borrow("describe_dataset")` and
invokes it directly — reusing the existing capability with no planner/LLM, the
Rust-like borrow path.
