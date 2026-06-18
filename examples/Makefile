# ────────────────────────────────────────────────────────────────────────────
# CSP demos — run all three example apps with one command.
#
#   make dev       run all 3 backends + 3 frontends (Ctrl-C stops everything)
#   make stop      free all dev ports
#   make install   npm install for all three frontends
#   make test      run the CSP library test suite
#
#   make csv-rag   just CSV-RAG   → http://localhost:5173  (api :8000)
#   make algoviz   just AlgoViz   → http://localhost:5174  (api :8001)
#   make montage   just Montage AI→ http://localhost:5175  (api :8002)
# ────────────────────────────────────────────────────────────────────────────

# ROOT = the examples/ dir (this Makefile's location, so `make -C examples` works too).
# REPO = the project root, one level up — it holds the shared .venv and tests/.
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO := $(abspath $(ROOT)/..)
PY   := $(REPO)/.venv/bin/python

.ONESHELL:
.SILENT:
.DEFAULT_GOAL := help
.PHONY: help dev stop install test csv-rag algoviz montage pitch
SHELL := /bin/bash

# Interactive multi-process targets live in scripts/run.sh — a plain shell
# script handles trap/wait/process-group cleanup far more reliably than make's
# .ONESHELL recipes do.

help:
	echo "CSP demo apps:"
	echo "  make dev       — run all three (csv-rag, algoviz, montage)"
	echo "  make stop      — free ports 8000-8002 / 5173-5175"
	echo "  make install   — npm install for all three frontends"
	echo "  make test      — run the CSP test suite"
	echo ""
	echo "  make csv-rag   — CSV-RAG    → http://localhost:5173  (api :8000)"
	echo "  make algoviz   — AlgoViz    → http://localhost:5174  (api :8001)"
	echo "  make montage   — Montage AI → http://localhost:5175  (api :8002)"
	echo "  make pitch     — Pitch (WC) → http://localhost:5176  (api :8003)"

dev:
	bash $(ROOT)/scripts/run.sh

pitch:
	bash $(ROOT)/scripts/run.sh pitch

csv-rag:
	bash $(ROOT)/scripts/run.sh csv-rag

algoviz:
	bash $(ROOT)/scripts/run.sh algoviz

montage:
	bash $(ROOT)/scripts/run.sh montage

stop:
	pids=$$(lsof -nP -iTCP:8000 -iTCP:8001 -iTCP:8002 -iTCP:5173 -iTCP:5174 -iTCP:5175 -sTCP:LISTEN -t 2>/dev/null || true)
	if [ -n "$$pids" ]; then kill $$pids 2>/dev/null || true; sleep 1; echo "stopped dev servers: $$pids"; else echo "no dev servers running"; fi
	true

install:
	( cd $(ROOT)/helloworld/frontend && npm install )
	( cd $(ROOT)/algoviz/frontend    && npm install )
	( cd $(ROOT)/montage-ai/frontend && npm install )
	( cd $(ROOT)/pitch/frontend      && npm install )

test:
	cd $(REPO) && $(PY) -m pytest tests/ -q
