"""
csp.orchestrator.planner_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PlannerStore — makes the otherwise-invisible JSON-RPC visible and durable.

When an Orchestrator starts, it creates a `planner/` folder inside the
developer's project (the server's current working directory). Everything
that flows over the wire and everything the synthesizer generates is
written there, so the developer can open the files and see exactly what
CSP planned, registered, and executed.

Layout created in the developer's project:

    planner/
    ├── jsonrpc.ndjson          # every wire message, in + out, timestamped
    ├── capabilities/           # synthesized capability specs (registered)
    │   └── <name>.json         # reloaded on next startup → reuse, no re-synth
    └── plans/                  # one file per submitted goal
        └── <ts>-<goal>.json    # the ExecutionPlan that was built

The `capabilities/` folder is the "registered" half — synthesized specs are
persisted here and loaded back into the registry on the next run, so a
capability is only ever synthesized once. The `jsonrpc.ndjson` log is the
"executed from" half — a complete, replayable record of the wire traffic.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from .capability import SynthesizedCapability

log = logging.getLogger("csp.planner_store")

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")


class PlannerStore:
    """
    Persists JSON-RPC traffic, synthesized capability specs, and plans
    into a `planner/` folder in the developer's project.

    Parameters
    ----------
    root:
        Folder name (or path) to create. Relative paths are resolved
        against the current working directory — i.e. the developer's
        project that spawned the server.
    """

    __slots__ = ("root", "caps_dir", "plans_dir", "log_path")

    def __init__(self, root: str = "planner") -> None:
        self.root      = Path(root).resolve()
        self.caps_dir  = self.root / "capabilities"
        self.plans_dir = self.root / "plans"
        self.log_path  = self.root / "jsonrpc.ndjson"

        self.caps_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self._write_gitignore()

        log.info("planner store ready at %s", self.root)

    def _write_gitignore(self) -> None:
        """
        Drop a .gitignore so the runtime wire log and per-goal plans are
        ignored, while synthesized capability specs (reusable artifacts)
        stay committable.
        """
        gitignore = self.root / ".gitignore"
        if gitignore.exists():
            return
        gitignore.write_text(
            "# CSP planner — runtime artifacts (auto-generated)\n"
            "jsonrpc.ndjson\n"
            "plans/\n"
            "\n"
            "# Synthesized capabilities are reusable — keep them tracked.\n"
            "!capabilities/\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # JSON-RPC wire log
    # ------------------------------------------------------------------

    def log_message(self, direction: str, obj: dict[str, Any]) -> None:
        """
        Append one wire message to jsonrpc.ndjson.

        direction: "in"  — received from client (stdin)
                   "out" — sent to client (stdout)
        """
        entry = {
            "ts":     round(time.time(), 4),
            "dir":    direction,
            "method": obj.get("method"),
            "id":     obj.get("id"),
            "msg":    obj,
        }
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:                       # never break the server over logging
            log.warning("planner log write failed: %s", exc)

    # ------------------------------------------------------------------
    # Synthesized capability specs (the "registered" half)
    # ------------------------------------------------------------------

    def save_capability(self, cap: SynthesizedCapability) -> None:
        """
        Persist a synthesized capability: the full spec as <name>.json AND,
        when it carries real Python, the runnable source as <name>.py so the
        developer can read exactly what gets executed.
        """
        safe = _safe(cap.name)
        path = self.caps_dir / f"{safe}.json"
        record = {
            "name":           cap.name,
            "description":    cap.description,
            "version":        cap.version,
            "synthesized_at": cap.synthesized_at,
            "id":             cap.id,
            "spec":           cap.spec,
        }
        try:
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

            # Also drop the runnable source next to it, if present
            code = cap.spec.get("params", {}).get("execution", {}).get("code", "")
            if code:
                py_path = self.caps_dir / f"{safe}.py"
                header = (
                    f'"""\nSynthesized capability: {cap.name}\n'
                    f"{cap.description}\n\n"
                    f"Auto-generated by CSP. This is the exact code the sandbox runs.\n"
                    f'"""\n\n'
                )
                py_path.write_text(header + code + "\n", encoding="utf-8")

            log.info("persisted synthesized capability %r → %s", cap.name, path.name)
        except Exception as exc:
            log.warning("failed to persist capability %r: %s", cap.name, exc)

    def load_capabilities(self) -> list[SynthesizedCapability]:
        """
        Reload previously synthesized capabilities from disk.

        Called at startup so a capability is synthesized only once, ever —
        subsequent runs reuse the persisted spec.
        """
        caps: list[SynthesizedCapability] = []
        for path in sorted(self.caps_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                caps.append(SynthesizedCapability(
                    name=record["name"],
                    spec=record["spec"],
                    description=record.get("description", ""),
                    version=record.get("version", "1.0.0"),
                ))
            except Exception as exc:
                log.warning("skipping unreadable capability file %s: %s", path.name, exc)
        if caps:
            log.info("reloaded %d persisted capabilities from %s", len(caps), self.caps_dir)
        return caps

    # ------------------------------------------------------------------
    # Execution plans
    # ------------------------------------------------------------------

    def save_plan(self, goal: str, plan: Any) -> None:
        """Persist an ExecutionPlan as one JSON file per goal."""
        ts   = time.strftime("%Y%m%d-%H%M%S")
        path = self.plans_dir / f"{ts}-{_safe(goal)[:48]}.json"
        record = {
            "ts":    round(time.time(), 4),
            "goal":  goal,
            "steps": [
                {
                    "capability":     s.capability,
                    "description":    getattr(s, "description", ""),
                    "needs_synthesis": getattr(s, "needs_synthesis", False),
                    "args":           getattr(s, "args", {}),
                }
                for s in getattr(plan, "steps", [])
            ],
        }
        try:
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("persisted plan for goal %r → %s", goal[:40], path.name)
        except Exception as exc:
            log.warning("failed to persist plan: %s", exc)


def _safe(name: str) -> str:
    """Make a string safe to use as a filename."""
    return _SAFE_NAME.sub("_", name.strip()).strip("_") or "unnamed"
