"""
csp.orchestrator.borrow
~~~~~~~~~~~~~~~~~~~~~~~~~
Rust-like *borrowing* of capabilities.

Synthesis CREATES a capability (ownership). Borrowing instead takes a shared,
read-only handle to one that ALREADY EXISTS — it never creates. Like `&T` in
Rust:

  - You can only borrow a capability that exists (borrowing a missing one is an
    error, never a silent synthesis).
  - A borrow is immutable: you can invoke it and inspect it, but not mutate or
    replace the underlying capability.
  - Many shared borrows can be live at once.
  - While ANY borrow is live, the capability cannot be forgotten/replaced —
    the registry refuses, the way Rust won't let you free a value that's still
    borrowed.
  - Borrows are scoped: use `async with app.borrow(name) as cap:` and the
    borrow is released automatically at the end of the block (RAII).

This makes reuse explicit and safe: a service declares "I'm borrowing
`detect_anomalies`" rather than risking a duplicate synthesis.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable


class BorrowError(RuntimeError):
    """Raised when a borrow rule is violated (e.g. freeing a borrowed capability)."""


class BorrowedCapability:
    """
    A live, read-only handle to a borrowed capability.

    Invoke it like a function; inspect its metadata; you cannot mutate the
    underlying capability through it.
    """

    __slots__ = ("_cap", "_invoke", "_live")

    def __init__(
        self,
        capability: Any,
        invoke: Callable[[Any, dict], Awaitable[Any]],
    ) -> None:
        self._cap    = capability
        self._invoke = invoke
        self._live   = True

    # ── read-only views ───────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self._cap.name

    @property
    def kind(self) -> str:
        return self._cap.kind.name.lower()

    @property
    def description(self) -> str:
        return getattr(self._cap, "description", "")

    @property
    def version(self) -> str:
        return getattr(self._cap, "version", "")

    @property
    def code(self) -> str:
        """Generated source, for synthesized capabilities (empty otherwise)."""
        return getattr(self._cap, "code", "")

    @property
    def live(self) -> bool:
        return self._live

    # ── invocation ────────────────────────────────────────────────────
    async def invoke(self, **args: Any) -> Any:
        if not self._live:
            raise BorrowError(f"borrow of {self.name!r} has been released")
        return await self._invoke(self._cap, args)

    async def __call__(self, **args: Any) -> Any:
        return await self.invoke(**args)

    def _release(self) -> None:
        self._live = False

    def __repr__(self) -> str:
        state = "live" if self._live else "released"
        return f"<BorrowedCapability {self.name!r} {self.kind} ({state})>"


class BorrowScope:
    """
    Async context manager returned by ``Orchestrator.borrow(name)``.

    Acquires the borrow on enter (raising if the capability doesn't exist),
    releases it on exit. Also usable manually via ``acquire()`` / ``release()``.
    """

    __slots__ = ("_registry", "_name", "_invoke", "_handle")

    def __init__(self, registry: Any, name: str, invoke) -> None:
        self._registry = registry
        self._name     = name
        self._invoke   = invoke
        self._handle: BorrowedCapability | None = None

    async def acquire(self) -> BorrowedCapability:
        cap = await self._registry.acquire_borrow(self._name)
        self._handle = BorrowedCapability(cap, self._invoke)
        return self._handle

    async def release(self) -> None:
        if self._handle is not None:
            self._handle._release()
            self._handle = None
            await self._registry.release_borrow(self._name)

    async def __aenter__(self) -> BorrowedCapability:
        return await self.acquire()

    async def __aexit__(self, *exc: Any) -> bool:
        await self.release()
        return False
