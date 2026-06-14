"""
csp.orchestrator.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~
CapabilityRegistry — unified store for both registered and synthesized
capabilities.

Resolution order (always):
  1. Registered capabilities (developer-defined, exact name match)
  2. Synthesized capabilities (previously generated, exact name match)
  3. Not found → synthesizer must create it

Design:
- Two separate dicts for O(1) lookup by name
- Synthesized capabilities also indexed by semantic tags for fuzzy
  matching (planner may use slightly different names)
- Thread-safe via asyncio.Lock — registry is mutated at runtime when
  new capabilities are synthesized
- list_all() returns a snapshot for the planner to reason about what
  already exists before deciding to synthesize
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from .capability import (
    AnyCapability,
    CapabilityKind,
    RegisteredCapability,
    SynthesizedCapability,
)

log = logging.getLogger("csp.registry")


class CapabilityRegistry:
    """
    Unified capability store.

    Usage:
        registry = CapabilityRegistry()
        registry.register(my_capability)
        cap = await registry.resolve("predict_churn")
    """

    __slots__ = ("_registered", "_synthesized", "_lock", "persist_hook")

    def __init__(self) -> None:
        self._registered:  dict[str, RegisteredCapability]  = {}
        self._synthesized: dict[str, SynthesizedCapability] = {}
        self._lock = asyncio.Lock()
        # Optional callback invoked when a capability is synthesized at runtime.
        # The orchestrator sets this to persist specs into planner/capabilities/.
        self.persist_hook: Optional[Callable[[SynthesizedCapability], None]] = None

    # ------------------------------------------------------------------
    # Registration (called at startup by @app.capability decorator)
    # ------------------------------------------------------------------

    def register(self, capability: RegisteredCapability) -> None:
        """Register a developer-defined capability. Called at startup."""
        self._registered[capability.name] = capability
        log.debug("registered capability %r", capability.name)

    # ------------------------------------------------------------------
    # Synthesis storage (called by synthesizer at runtime)
    # ------------------------------------------------------------------

    async def store_synthesized(self, capability: SynthesizedCapability) -> None:
        """Store a newly synthesized capability. Thread-safe."""
        async with self._lock:
            self._synthesized[capability.name] = capability
        log.debug("stored synthesized capability %r id=%s", capability.name, capability.id[:8])

        # Persist the spec (e.g. to planner/capabilities/) if a hook is set
        if self.persist_hook is not None:
            try:
                self.persist_hook(capability)
            except Exception as exc:
                log.warning("persist_hook failed for %r: %s", capability.name, exc)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve(self, name: str) -> Optional[AnyCapability]:
        """
        Look up a capability by name.

        Returns registered first, then synthesized, then None.
        """
        # Registered takes priority — exact match
        if name in self._registered:
            return self._registered[name]

        # Synthesized — exact match
        async with self._lock:
            if name in self._synthesized:
                return self._synthesized[name]

        # Fuzzy: try normalized name (lowercase, underscores)
        normalized = _normalize(name)
        for cap_name, cap in self._registered.items():
            if _normalize(cap_name) == normalized:
                return cap

        async with self._lock:
            for cap_name, cap in self._synthesized.items():
                if _normalize(cap_name) == normalized:
                    return cap

        return None

    def exists(self, name: str) -> bool:
        """Synchronous existence check — safe to call from planner."""
        return (
            name in self._registered
            or name in self._synthesized
            or _normalize(name) in {_normalize(k) for k in self._registered}
            or _normalize(name) in {_normalize(k) for k in self._synthesized}
        )

    # ------------------------------------------------------------------
    # Introspection — used by planner to build context
    # ------------------------------------------------------------------

    def list_registered(self) -> list[RegisteredCapability]:
        return list(self._registered.values())

    async def list_synthesized(self) -> list[SynthesizedCapability]:
        async with self._lock:
            return list(self._synthesized.values())

    async def list_all(self) -> list[AnyCapability]:
        """Full snapshot for planner context."""
        async with self._lock:
            synth = list(self._synthesized.values())
        return list(self._registered.values()) + synth

    async def summary_for_planner(self) -> str:
        """
        Human-readable capability list for injecting into planner prompt.
        Registered capabilities include their param schemas.
        Synthesized capabilities include their description.
        """
        lines = []
        all_caps = await self.list_all()

        if not all_caps:
            return "No capabilities registered yet."

        for cap in all_caps:
            kind_label = "registered" if cap.kind == CapabilityKind.REGISTERED else "synthesized"
            if isinstance(cap, RegisteredCapability):
                params = ", ".join(
                    f"{p.name}: {p.type}" for p in cap.params
                )
                lines.append(
                    f"- {cap.name} ({kind_label}): {cap.description or 'no description'}"
                    + (f" | params: {params}" if params else "")
                )
            else:
                lines.append(
                    f"- {cap.name} ({kind_label}): {cap.description or 'no description'}"
                )

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._registered) + len(self._synthesized)

    def __repr__(self) -> str:
        return (
            f"<CapabilityRegistry registered={len(self._registered)} "
            f"synthesized={len(self._synthesized)}>"
        )


def _normalize(name: str) -> str:
    """Normalize capability name for fuzzy matching."""
    return name.lower().replace("-", "_").replace(" ", "_").strip()