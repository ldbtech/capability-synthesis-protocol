#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run one or more CSP demo apps (backend + frontend) with prefixed logs.
#
#   scripts/run.sh                  → all three (csv-rag, algoviz, montage)
#   scripts/run.sh csv-rag          → just CSV-RAG
#   scripts/run.sh algoviz montage  → AlgoViz + Montage
#
# Ctrl-C stops everything (kill 0 targets this script's process group only).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

# ROOT = the examples/ dir (where the demo apps live).
# REPO = the project root, one level up — it holds the shared .venv.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PY="$REPO/.venv/bin/python"

apps=("$@")
[ ${#apps[@]} -eq 0 ] && apps=(csv-rag algoviz montage pitch)

# Free any ports we're about to use so a stale server doesn't block a fresh one.
stale=$(lsof -nP -iTCP:8000 -iTCP:8001 -iTCP:8002 -iTCP:8003 -iTCP:5173 -iTCP:5174 -iTCP:5175 -iTCP:5176 \
             -sTCP:LISTEN -t 2>/dev/null || true)
if [ -n "$stale" ]; then kill $stale 2>/dev/null || true; sleep 1; fi

# Ctrl-C / TERM → tear down every child (kill 0 = this process group).
trap 'echo; echo "→ stopping CSP demos..."; kill 0 2>/dev/null' INT TERM

start_csv_rag() {
  ( cd "$ROOT/helloworld/backend"  && "$PY" -m uvicorn app:api --port 8000 ) 2>&1 | sed 's/^/[csv-rag api] /' &
  ( cd "$ROOT/helloworld/frontend" && npm run dev )                                   2>&1 | sed 's/^/[csv-rag web] /' &
}
start_algoviz() {
  ( cd "$ROOT/algoviz/backend"  && "$PY" -m uvicorn app:api --port 8001 ) 2>&1 | sed 's/^/[algoviz api] /' &
  ( cd "$ROOT/algoviz/frontend" && npm run dev )                                   2>&1 | sed 's/^/[algoviz web] /' &
}
start_montage() {
  ( cd "$ROOT/montage-ai/backend"  && "$PY" -m uvicorn app:app --port 8002 ) 2>&1 | sed 's/^/[montage api] /' &
  ( cd "$ROOT/montage-ai/frontend" && npm run dev )                                                               2>&1 | sed 's/^/[montage web] /' &
}
start_pitch() {
  ( cd "$ROOT/pitch/backend"  && "$PY" -m uvicorn app:app --port 8003 ) 2>&1 | sed 's/^/[pitch api] /' &
  ( cd "$ROOT/pitch/frontend" && npm run dev )                                                               2>&1 | sed 's/^/[pitch web] /' &
}

echo "Starting CSP demos: ${apps[*]}"
for a in "${apps[@]}"; do
  case "$a" in
    csv-rag) start_csv_rag; echo "    CSV-RAG     http://localhost:5173  (api :8000)" ;;
    algoviz) start_algoviz; echo "    AlgoViz     http://localhost:5174  (api :8001)" ;;
    montage) start_montage; echo "    Montage AI  http://localhost:5175  (api :8002)" ;;
    pitch)   start_pitch;   echo "    Pitch       http://localhost:5176  (api :8003)" ;;
    *) echo "    ?? unknown app '$a' (use: csv-rag | algoviz | montage | pitch)" ;;
  esac
done
echo "    (Ctrl-C stops everything)"
echo

wait
