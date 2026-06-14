"""
csp.orchestrator.elicitation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ElicitationManager — manages pending human-in-the-loop requests.

When the executor or a capability needs human input, it calls
manager.request(). This suspends execution (awaits a Future) until
the client sends back a response via manager.respond().

The JSON-RPC 2.0 elicitation request is sent to the client over the
active stream. The client surfaces it as an ElicitRequest, the user
responds, and the client POSTs back to /elicit/respond which calls
manager.respond() here — resuming execution.

Timeout: if no response arrives within timeout seconds, the Future
is cancelled and execution continues with a default value (or error).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("csp.elicitation")

_DEFAULT_TIMEOUT = 120.0   # seconds


@dataclass(slots=True)
class ElicitationRequest:
    """
    An outbound elicitation — sent to the client over the stream.
    Mirrors the client-side ElicitRequest type exactly.
    """
    id:         str
    kind:       str           # "approval" | "input" | "choice"
    question:   str
    options:    list[str]     = field(default_factory=list)
    context:    Optional[str] = None
    capability: Optional[str] = None

    def to_jsonrpc(self) -> dict[str, Any]:
        """Serialize to JSON-RPC 2.0 notification for wire."""
        return {
            "jsonrpc": "2.0",
            "method": "csp.elicit.request",
            "params": {
                "id":         self.id,
                "kind":       self.kind,
                "question":   self.question,
                "options":    self.options,
                "context":    self.context,
                "capability": self.capability,
            },
        }


class ElicitationManager:
    """
    Manages pending elicitation futures for a session.

    One manager instance per active session.

    Usage inside executor:
        answer = await manager.request(
            kind="approval",
            question="Deploy to production?",
            capability="deploy_model",
        )
        if answer.lower() == "yes":
            ...
    """

    __slots__ = ("_pending", "_timeout", "_lock")

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._timeout = timeout
        self._lock    = asyncio.Lock()

    async def request(
        self,
        kind: str,
        question: str,
        *,
        options: Optional[list[str]] = None,
        context: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> tuple[ElicitationRequest, asyncio.Future[str]]:
        """
        Create a pending elicitation.

        Returns (ElicitationRequest, Future).
        Caller sends the request over the stream, then awaits the future.

        Example:
            elicit_req, fut = await manager.request("approval", "Deploy?")
            yield elicit_req          # send to client via stream
            answer = await fut        # wait for client response
        """
        request_id = str(uuid.uuid4())
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()

        async with self._lock:
            self._pending[request_id] = fut

        elicit_req = ElicitationRequest(
            id=request_id,
            kind=kind,
            question=question,
            options=options or [],
            context=context,
            capability=capability,
        )

        log.debug("elicitation created id=%s kind=%s", request_id[:8], kind)
        return elicit_req, fut

    async def respond(self, request_id: str, value: str) -> bool:
        """
        Resolve a pending elicitation with the user's answer.

        Returns True if the request was found and resolved, False if unknown.
        Called by the server when POST /elicit/respond arrives.
        """
        async with self._lock:
            fut = self._pending.pop(request_id, None)

        if fut is None:
            log.warning("respond() called with unknown request_id=%s", request_id[:8])
            return False

        if not fut.done():
            fut.set_result(value)
            log.debug("elicitation resolved id=%s value=%r", request_id[:8], value[:40])
            return True

        return False

    async def cancel_all(self) -> None:
        """Cancel all pending elicitations — called on session close."""
        async with self._lock:
            pending = dict(self._pending)
            self._pending.clear()

        for request_id, fut in pending.items():
            if not fut.done():
                fut.cancel()
                log.debug("elicitation cancelled id=%s", request_id[:8])

    @property
    def pending_count(self) -> int:
        return len(self._pending)