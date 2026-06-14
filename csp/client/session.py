"""
csp.client.session
~~~~~~~~~~~~~~~~~~~~
ClientSession — the stateful execution context.
 
Responsibilities:
- Owns the transport for its lifetime
- Maps raw _Envelope objects to public types (StreamEvent, ElicitRequest, Result)
- Manages the elicitation loop: pauses the stream, waits for ElicitResponse,
  sends it back, resumes
- Tracks conversation history for multi-turn sessions
- Provides both streaming (stream()) and blocking (run()) interfaces
 
Performance choices:
- _pending_elicits uses a dict[str, asyncio.Future] so each elicitation
  resolves O(1) without scanning.
- Event mapping is a dispatch table (dict lookup) not an if/elif chain.
- Conversation history is a bounded deque so long sessions don't leak memory.
"""

from __future__ import annotations

import asyncio
import logging 
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Optional,
    Union
)

from .transport import StreamableHTTP

from .types import (
    _Envelope,
    CapabilityResult,
    ElicitKind,
    ElicitRequest,
    ElicitResponse,
    EventKind,
    Result,
    ResultStatus,
    StreamEvent,
)

log = logging.getLogger("csp.session")

# Max Conversation turns kept in memory
_HISTORY_MAXLEN = 128

# Type alias for the elicit handler the developer can register
ElicitHandler = Callable[
    [ElicitRequest],
    Coroutine[Any, Any, ElicitResponse]
]

# Union of everything the public stream() yields
StreamItem = Union[StreamEvent, ElicitRequest, Result]

class ClientSession:
    """
    Stateful session between client and CSP orchestrator.
 
    Typical usage (managed — recommended):
 
        async with ClientSession(transport) as session:
            async for item in session.stream("predict churn"):
                if isinstance(item, StreamEvent):
                    print(item.message)
                elif isinstance(item, ElicitRequest):
                    resp = ElicitResponse(item.id, "yes")
                    await session.respond(resp)
                elif isinstance(item, Result):
                    print(item.summary)
 
    Or blocking:
 
        async with ClientSession(transport) as session:
            result = await session.run("predict churn")
            print(result.summary)
 
    Parameters
    ----------
    transport:
        A connected (or not yet connected) StreamableHTTP instance.
    session_id:
        Optional fixed session id. If None, a UUID4 is generated.
    elicit_handler:
        Optional async callable invoked automatically for each ElicitRequest
        when using run().  If not provided and an elicitation arrives during
        run(), a TimeoutError is raised after elicit_timeout seconds.
    elicit_timeout:
        Seconds to wait for an elicitation response during run().
    extra_headers:
        Extra HTTP headers forwarded on every request (e.g. auth tokens).
    history_maxlen:
        Max conversation turns to retain.
    """

    __slots__ = (
        "_transport",
        "_session_id",
        "_elicit_handler",
        "_elicit_timeout",
        "_extra_headers",
        "_history",
        "_pending_elicits",   # dict[request_id, Future[ElicitResponse]]
        "_closed",
        "_lock",
    )

    def __init__(
        self,
        transport: StreamableHTTP,
        *,
        session_id: Optional[str] = None,
        elicit_handler: Optional[ElicitHandler] = None,
        elicit_timeout: float = 120.0,
        extra_headers: Optional[dict[str, str]] = None,
        history_maxlen: int = _HISTORY_MAXLEN,
    ) -> None:
        self._transport             = transport,
        self._session_id = session_id or str(uuid.uuid4())
        self._elicit_handler  = elicit_handler
        self._elicit_timeout  = elicit_timeout
        self._extra_headers   = extra_headers or {}
        self._history: deque[dict[str, Any]] = deque(maxlen=history_maxlen)
        self._pending_elicits: dict[str, asyncio.Future[ElicitResponse]] = {}
        self._closed          = False
        self._lock            = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def session_id(self) -> str:
        return self._session_id
    
    @property
    def history(self) -> list[dict[str, Any]]:
        return list[self._history]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def open(self) -> None:
        await self._transport.connect()
        log.debug("session opened id=%s", self._session_id)
 
    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            # resolve any pending elicits with cancellation
            for fut in self._pending_elicits.values():
                if not fut.done():
                    fut.cancel()
            self._pending_elicits.clear()
            await self._transport.close()
            log.debug("session closed id=%s", self._session_id)
 
    async def __aenter__(self) -> "ClientSession":
        await self.open()
        return self
 
    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Elicitation response (called by developer or elicit_handler)
    # ------------------------------------------------------------------
    async def respond(self, response: ElicitResponse) -> None:
        """
        Provide an answer to a pending ElicitRequest.
 
        This both:
        1. Resolves the local Future so stream() can continue yielding.
        2. Sends the answer to the orchestrator over HTTP.
        """
        async with self._lock:
            fut = self._pending_elicits.pop(response.request_id, None)
 
        if fut is None:
            log.warning("respond() called with unknown request_id=%s", response.request_id)
            return
 
        if not fut.done():
            fut.set_result(response)
 
        await self._transport.send_elicit_response(
            session_id=self._session_id,
            request_id=response.request_id,
            value=response.value,
        )
        log.debug("elicit response sent request_id=%s", response.request_id)

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------
 
    async def stream(self, goal: str) -> AsyncIterator[StreamItem]:
        """
        Submit a goal and yield StreamEvent / ElicitRequest / Result.
 
        ElicitRequests are yielded to the caller. The caller must call
        session.respond() before the next item is yielded (the stream
        pauses internally until responded to).
 
        The final item yielded is always a Result.
        """
        self._assert_open()
        self._history.append({"role": "user", "content": goal})
        t_start = time.monotonic()
 
        capabilities: list[CapabilityResult] = []
        elicitations: list[tuple[ElicitRequest, ElicitResponse]] = []
 
        async for envelope in self._transport.stream(
            goal=goal,
            session_id=self._session_id,
            extra_headers=self._extra_headers,
        ):
            item = self._map_envelope(envelope, capabilities)
 
            if item is None:
                continue
 
            if isinstance(item, ElicitRequest):
                # Register a future so respond() can resolve it
                fut: asyncio.Future[ElicitResponse] = asyncio.get_event_loop().create_future()
                async with self._lock:
                    self._pending_elicits[item.id] = fut
 
                yield item
 
                # Pause stream until responded — wait for the future
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(fut),
                        timeout=self._elicit_timeout,
                    )
                    elicitations.append((item, response))
                except asyncio.TimeoutError:
                    log.error("elicitation timed out request_id=%s", item.id)
                    raise TimeoutError(
                        f"No response to elicitation '{item.question}' "
                        f"within {self._elicit_timeout}s"
                    )
 
            elif isinstance(item, Result):
                # Rebuild result with accumulated local data
                final = Result(
                    status=item.status,
                    summary=item.summary,
                    capabilities=tuple(capabilities),
                    elicitations=tuple(elicitations),
                    output=item.output,
                    error=item.error,
                    duration=time.monotonic() - t_start,
                )
                self._history.append({"role": "assistant", "content": final.summary})
                yield final
                return
 
            else:
                yield item
    
    # ------------------------------------------------------------------
    # Blocking interface
    # ------------------------------------------------------------------
 
    async def run(self, goal: str) -> Result:
        """
        Submit a goal and block until a Result is returned.
 
        If elicitations occur and elicit_handler is set, it is called
        automatically.  Otherwise raises TimeoutError.
        """
        result: Optional[Result] = None
 
        async for item in self.stream(goal):
            if isinstance(item, StreamEvent):
                log.debug("[%s] %s", item.kind.name, item.message)
 
            elif isinstance(item, ElicitRequest):
                if self._elicit_handler is None:
                    raise RuntimeError(
                        "Received an ElicitRequest but no elicit_handler was provided. "
                        "Use session.stream() to handle elicitations manually, or pass "
                        "elicit_handler= to ClientSession."
                    )
                response = await self._elicit_handler(item)
                await self.respond(response)
 
            elif isinstance(item, Result):
                result = item
 
        assert result is not None, "stream ended without a Result"
        return result

    # ------------------------------------------------------------------
    # Internal: envelope → public type mapping
    # ------------------------------------------------------------------
 
    # Dispatch table: envelope type string → mapper method name
    _DISPATCH: dict[str, str] = {
        "event":  "_map_event",
        "elicit": "_map_elicit",
        "result": "_map_result",
        "error":  "_map_error",
    }
 
    def _map_envelope(
        self,
        env: _Envelope,
        capabilities: list[CapabilityResult],
    ) -> Optional[StreamItem]:
        method_name = self._DISPATCH.get(env.type)
        if method_name is None:
            log.warning("unknown envelope type=%r", env.type)
            return None
        return getattr(self, method_name)(env.payload, capabilities)
 
    def _map_event(
        self,
        payload: dict[str, Any],
        capabilities: list[CapabilityResult],
    ) -> Optional[StreamEvent]:
        raw_kind   = payload.get("kind", "LOG")
        message    = payload.get("message", "")
        capability = payload.get("capability")
        metadata   = payload.get("metadata", {})
 
        try:
            kind = EventKind[raw_kind.upper()]
        except KeyError:
            kind = EventKind.LOG
 
        # Track completed capabilities locally
        if kind == EventKind.CAPABILITY_END:
            capabilities.append(
                CapabilityResult(
                    name=capability or "unknown",
                    success=metadata.get("success", True),
                    output=metadata.get("output", {}),
                    error=metadata.get("error"),
                    duration=metadata.get("duration"),
                )
            )
 
        return StreamEvent(
            kind=kind,
            message=message,
            capability=capability,
            metadata=metadata,
        )
 
    def _map_elicit(
        self,
        payload: dict[str, Any],
        _capabilities: list[CapabilityResult],
    ) -> ElicitRequest:
        raw_kind = payload.get("kind", "INPUT")
        try:
            kind = ElicitKind[raw_kind.upper()]
        except KeyError:
            kind = ElicitKind.INPUT
 
        return ElicitRequest(
            id=payload["id"],
            kind=kind,
            question=payload["question"],
            options=tuple(payload.get("options", [])),
            context=payload.get("context"),
            capability=payload.get("capability"),
        )
 
    def _map_result(
        self,
        payload: dict[str, Any],
        _capabilities: list[CapabilityResult],
    ) -> Result:
        raw_status = payload.get("status", "OK")
        try:
            status = ResultStatus[raw_status.upper()]
        except KeyError:
            status = ResultStatus.OK
 
        return Result(
            status=status,
            summary=payload.get("summary", ""),
            output=payload.get("output", {}),
            error=payload.get("error"),
        )
 
    def _map_error(
        self,
        payload: dict[str, Any],
        _capabilities: list[CapabilityResult],
    ) -> Result:
        return Result(
            status=ResultStatus.ERROR,
            summary=payload.get("message", "An error occurred."),
            error=payload.get("message"),
        )
 
    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
 
    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("ClientSession is closed")
 
 