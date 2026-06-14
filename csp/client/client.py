"""
csp.client.client
~~~~~~~~~~~~~~~~~~~
BrainClient — the single object developers import and use.

This is the only public surface of the client package. It wraps
transport + session lifecycle so the developer never has to manage
either directly.

Design:
- Context manager for fire-and-forget convenience
- Persistent mode (keep_alive=True) reuses the same session across
  multiple run() calls — good for REPL / interactive use
- Ephemeral mode (keep_alive=False, default) opens a fresh session
  per run() / stream() call — good for stateless request handlers
- on_elicit() decorator lets developers register a handler once
- on_event() decorator lets developers register a handler for streaming
  events (e.g. to pipe to a UI in real time)

Example — minimal:

    from csp.client import BrainClient
    from csp.client.transport import StreamableHTTP

    client = BrainClient(StreamableHTTP("http://localhost:8000"))
    result = await client.run("predict customer churn")
    print(result.summary)

Example — with elicitation handler:

    client = BrainClient(StreamableHTTP("http://localhost:8000"))

    @client.on_elicit
    async def handle(request):
        print(f"[{request.kind.name}] {request.question}")
        answer = input("> ")
        return ElicitResponse(request.id, answer)

    result = await client.run("deploy model to production")
    print(result.summary)

Example — streaming with event handler:

    @client.on_event
    async def show(event):
        print(f"  → {event.message}")

    async for item in client.stream("run etl pipeline"):
        if isinstance(item, Result):
            print(item.summary)

Example — multi-turn persistent session:

    async with client.session() as session:
        r1 = await session.run("load dataset")
        r2 = await session.run("train model on it")
        r3 = await session.run("deploy to staging")
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Coroutine, Optional, Union

from .session import ClientSession, ElicitHandler, StreamItem
from .transport import StreamableHTTP
from .types import ElicitRequest, ElicitResponse, Result, StreamEvent

log = logging.getLogger("csp.client")


class BrainClient:
    """
    Top-level developer interface for CSP.

    Parameters
    ----------
    transport:
        A StreamableHTTP instance pointing at the orchestrator.
    keep_alive:
        If True, reuse the same ClientSession across calls. The session
        is lazily created on first use and closed when the client is
        closed or used as an async context manager.
        If False (default), each run()/stream() gets a fresh session.
    session_id:
        Fixed session id passed to every ClientSession. Only meaningful
        when keep_alive=True.
    extra_headers:
        Headers forwarded on every request (e.g. Authorization bearer).
    elicit_timeout:
        Seconds to wait for an elicitation response in run() mode.
    """

    def __init__(
        self,
        transport: StreamableHTTP,
        *,
        keep_alive: bool = False,
        session_id: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        elicit_timeout: float = 120.0,
    ) -> None:
        self._transport      = transport
        self._keep_alive     = keep_alive
        self._session_id     = session_id
        self._extra_headers  = extra_headers or {}
        self._elicit_timeout = elicit_timeout

        self._elicit_handler: Optional[ElicitHandler] = None
        self._event_handler:  Optional[Callable[[StreamEvent], Coroutine[Any, Any, None]]] = None

        self._persistent_session: Optional[ClientSession] = None
        self._closed = False

    # ------------------------------------------------------------------
    # Handler registration (decorator API)
    # ------------------------------------------------------------------

    def on_elicit(
        self,
        fn: ElicitHandler,
    ) -> ElicitHandler:
        """
        Register an async elicitation handler.

            @client.on_elicit
            async def handler(request: ElicitRequest) -> ElicitResponse:
                answer = input(request.question + " > ")
                return ElicitResponse(request.id, answer)
        """
        self._elicit_handler = fn
        return fn

    def on_event(
        self,
        fn: Callable[[StreamEvent], Coroutine[Any, Any, None]],
    ) -> Callable[[StreamEvent], Coroutine[Any, Any, None]]:
        """
        Register an async event handler called for every StreamEvent.

            @client.on_event
            async def handler(event: StreamEvent) -> None:
                print(event.message)
        """
        self._event_handler = fn
        return fn

    # ------------------------------------------------------------------
    # Primary interfaces
    # ------------------------------------------------------------------

    async def run(self, goal: str) -> Result:
        """
        Submit a goal and block until a Result is returned.

        Elicitations are handled automatically via the registered
        on_elicit handler. If no handler is registered and an
        elicitation arrives, a RuntimeError is raised.
        """
        self._assert_open()

        if self._keep_alive:
            session = await self._get_or_create_persistent_session()
            return await self._run_on_session(session, goal)
        else:
            async with self._ephemeral_session() as session:
                return await self._run_on_session(session, goal)

    async def stream(self, goal: str) -> AsyncIterator[StreamItem]:
        """
        Submit a goal and yield StreamEvent / ElicitRequest / Result.

        The caller is responsible for calling client.respond() (or
        session.respond() if using session() context manager) when an
        ElicitRequest is yielded.
        """
        self._assert_open()

        if self._keep_alive:
            session = await self._get_or_create_persistent_session()
            async for item in self._stream_on_session(session, goal):
                yield item
        else:
            async with self._ephemeral_session() as session:
                async for item in self._stream_on_session(session, goal):
                    yield item

    async def respond(self, response: ElicitResponse) -> None:
        """
        Send an elicitation response.

        Only meaningful when using stream() in keep_alive mode where
        the persistent session is held internally. For explicit session
        management, call session.respond() directly.
        """
        if self._persistent_session is None:
            raise RuntimeError(
                "respond() on BrainClient is only valid in keep_alive mode "
                "after stream() has been called. For ephemeral sessions, "
                "use the session() context manager and call session.respond()."
            )
        await self._persistent_session.respond(response)

    # ------------------------------------------------------------------
    # Explicit session management (advanced use)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[ClientSession]:
        """
        Explicit session context manager for multi-turn conversations.

            async with client.session() as s:
                r1 = await s.run("load data")
                r2 = await s.run("train model")
        """
        self._assert_open()
        sess = self._make_session(session_id=session_id)
        async with sess:
            yield sess

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._persistent_session:
                await self._persistent_session.close()
                self._persistent_session = None
            await self._transport.close()
            log.debug("BrainClient closed")

    async def __aenter__(self) -> "BrainClient":
        await self._transport.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_session(self, session_id: Optional[str] = None) -> ClientSession:
        return ClientSession(
            transport=self._transport,
            session_id=session_id or self._session_id,
            elicit_handler=self._elicit_handler,
            elicit_timeout=self._elicit_timeout,
            extra_headers=self._extra_headers,
        )

    @asynccontextmanager
    async def _ephemeral_session(self) -> AsyncIterator[ClientSession]:
        sess = self._make_session()
        async with sess:
            yield sess

    async def _get_or_create_persistent_session(self) -> ClientSession:
        if self._persistent_session is None or self._persistent_session._closed:
            self._persistent_session = self._make_session()
            await self._persistent_session.open()
        return self._persistent_session

    async def _run_on_session(self, session: ClientSession, goal: str) -> Result:
        result: Optional[Result] = None
        async for item in self._stream_on_session(session, goal):
            if isinstance(item, Result):
                result = item
        assert result is not None
        return result

    async def _stream_on_session(
        self,
        session: ClientSession,
        goal: str,
    ) -> AsyncIterator[StreamItem]:
        async for item in session.stream(goal):
            if isinstance(item, StreamEvent):
                if self._event_handler is not None:
                    try:
                        await self._event_handler(item)
                    except Exception as exc:
                        log.warning("event_handler raised: %s", exc)
                yield item

            elif isinstance(item, ElicitRequest):
                # If a handler is registered, resolve it automatically
                # and do NOT yield it to the caller (transparent handling)
                if self._elicit_handler is not None:
                    try:
                        response = await self._elicit_handler(item)
                        await session.respond(response)
                    except Exception as exc:
                        log.error("elicit_handler raised: %s", exc)
                        raise
                else:
                    # No handler — yield to caller who must call respond()
                    yield item

            elif isinstance(item, Result):
                yield item

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("BrainClient is closed")