"""
csp.client.transport
~~~~~~~~~~~~~~~~~~~~~~
StreamableHTTP — the single transport the client uses.
 
Behaviour:
- All requests start as HTTP POST (JSON body, streaming response via
  chunked transfer / SSE).
- If the server signals upgrade=websocket in the response headers, the
  transport transparently re-connects over WebSocket for the remainder
  of that session.  This handles long-running capability executions
  (Kafka, Spark, etc.) without the developer ever noticing.
 
Wire format (internal, never exposed to developer):
  Each chunk / WS message is a newline-delimited JSON object:
      {"type": "event",  "payload": {...}}
      {"type": "elicit", "payload": {...}}
      {"type": "result", "payload": {...}}
      {"type": "error",  "payload": {"message": "..."}}
 
Performance choices:
- Single aiohttp.ClientSession reused across all requests (connection pool).
- orjson for zero-copy JSON decode (falls back to stdlib json if absent).
- Backpressure via asyncio.Queue with bounded maxsize so a slow consumer
  cannot cause unbounded memory growth.
- Exponential backoff reconnect for WebSocket drops.
"""

from __future__ import annotations

import asyncio
import logging 
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Text
from urllib.parse import urlparse, urlunparse
from webbrowser import get

import aiohttp
from .types import _Envelope

try:
    import orjson as _json

    def _loads(data: str | bytes) -> dict:
        return _json.loads(data)
except ImportError:
    import json as _json_std

    def _loads(data: str | bytes) -> dict:
        return _json_std.loads(data)

log = logging.getLogger("csp.transport")

# MAX Messages buffered between producer (network) and consumer (session)
_QUEUE_MAXSIZE = 256

# Websocket reconnect backoff: start, max, multiplier
_WS_BACKOFF_START = 0.5
_WS_BACKOFF_MAX = 30.0
_WS_BACKOFF_MUL = 2.0

def _http_to_ws(url: str) -> str:
    """ Convert http(s):// base url to ws(s):// for Websocket upgrade. """
    parsed = urlparse(url)

    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme = scheme))

class StreamableHTTP:
    """
    Transport layer for CSP client.
 
    Usage (managed by ClientSession — developer never instantiates directly):
 
        transport = StreamableHTTP("https://orchestrator.example.com")
        async with transport:
            async for envelope in transport.stream(goal, session_id, headers):
                ...
 
    Parameters
    ----------
    base_url:
        Root URL of the CSP orchestrator.
    timeout:
        Total timeout in seconds for a single HTTP response to begin.
        Does not apply to streaming — streams run until closed.
    connect_timeout:
        TCP connect timeout in seconds.
    max_ws_retries:
        How many times to retry a dropped WebSocket before giving up.
    """
    __slots__ = (
        "_base_url",
        "_ws_base_url",
        "connect_timeout",
        "_max_ws_retries",
        "_http_session",
        "_closed",
    )

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_ws_retries: int = 5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._ws_base_url = _http_to_ws(self._base_url)
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._max_ws_retries = max_ws_retries
        self._http_session = Optional[aiohttp.ClientSession] = None
        self._closed = False

    async def connect(self): 
        """Open the underlying HTTP connection pool"""
        if self._http_session is not None:
            return 
        
        connector = aiohttp.TCPConnector(
            limit=64,
            limit_per_host=16, # meaning 4 hosts with each 16 . 
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=None, # Stream have no total timeout.
            connect=self._connect_timeout,
            sock_connect=self._connect_timeout,
            sock_read=self._timeout,
        )

        self._http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Accept": "application/x-ndjson"},
        )
        log.debug("transport connected to %s", self._base_url)

    async def close(self):
        """Close the connection pool gracefully"""
        if self._http_session and not self._closed:
            self._closed = True
            await self._http_session.close()

            # aihttp needs a moment to release SSL Connections
            await asyncio.sleep(0.1)
            log.debug("transport closed")

    async def __aenter__(self) -> "StreamableHTTP":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public streaming interface
    # ------------------------------------------------------------------
    async def stream(
        self,
        goal: str,
        session_id: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[_Envelope]:
        """
        Submit a goal and yield _Envelope objects until the stream ends.
 
        Transparently upgrades to WebSocket if the server requests it.
        """
        assert self._http_session, "call connect() first"

        queue: asyncio.Queue[_Envelope | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)

        body = {"goal": goal, "session_id": session_id}

        headers = {"X-CSP-Session": session_id, **(extra_headers or {})}

        async with self._http_session.post(
            f"{self._base_url}/run",
            json=body,
            headers=headers,
        ) as resp:
            resp.raise_for_status()

            upgrade = resp.headers.get("X-CSP-Upgrade", "").lower()

            if upgrade == "websocket":
                # Server wants a long running WS stream - kick off in the background.
                ws_task = asyncio.create_task(
                    self._ws_stream(session_id, queue, extra_headers)
                )

                try:
                    async for envolope in self._drain_queue(queue):
                        yield envolope
                finally:
                    ws_task.cancel()
                    try:
                        await ws_task
                    except asyncio.CancelledError:
                        pass 
            else:
                # Standard NDJSON streaming over HTTP chunked transfer.
                async for envolope in self._http_ndjson_stream(resp):
                    yield envolope
    
    # ------------------------------------------------------------------
    # Elicitation response (out-of-band POST back to orchestrator)
    # ------------------------------------------------------------------
    async def send_elicit_response(
        self,
        session_id: str,
        request_id: str,
        value: str,
    ) -> None:
        """Send an elicitation answer back to the orchestrator."""
        assert self._http_session
        body = {
            "session_id": session_id,
            "request_id": request_id,
            "value":      value,
        }
        async with self._http_session.post(
            f"{self._base_url}/elicit/respond",
            json=body,
            headers={"X-CSP-Session": session_id},
        ) as resp:
            resp.raise_for_status()
            log.debug("elicit response sent request_id=%s", request_id)

    # ------------------------------------------------------------------
    # Internal: HTTP NDJSON stream reader
    # ------------------------------------------------------------------
    async def _http_ndjson_stream(
        self,
        resp: aiohttp.ClientResponse,
    ) -> AsyncIterator[_Envelope]:
        """
        READ newline_delimeter JSON Chunks from an HTTP response
        Each non-empty line is decoded and yielded as an _Envolope
        """
        buf = b""

        async for chunk, end_of_chunk in resp.content.iter_chunks():
            buf += chunk
            if not end_of_chunk:
                continue

            # Process all complete lines in buf
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()

                if not line:
                    continue

                try:
                    raw = _loads(line)
                    yield _Envelope(type=raw["type"], payload=raw.get("payload", {}))
                except Exception as exc:
                    log.warning("failed to decode envolope: %s | raw=%r", exc, line[:120])

            # flush remainder
            if buf.strip():
                try:
                    raw = _loads(buf)
                    yield _Envelope(type=raw["type"], payload=raw.get("payload", {}))
                except Exception as exc:
                    log.warning("failed to decode final envolope: %s", exc)
    
    # ------------------------------------------------------------------
    # Internal: WebSocket stream reader with reconnect
    # ------------------------------------------------------------------
    async def _ws_stream(
        self,
        session_id: str,
        queue: asyncio.Queue[_Envelope | None],
        extra_headers: Optional[dict[str, str]],
    )-> None:
        """
        Connect to the websocket endpoint and push envolopes into queue. 
        Reconnects with exeptional backoff on unexpected disconnects.
        Push None sentinel when the stream is truly done.
        """
        url = f"{self._ws_base_url}/ws/{session_id}"
        headers = {"X-CSP-Session": session_id, **(extra_headers or {})}
        backoff = _WS_BACKOFF_START
        retries = 0

        while retries < self._max_ws_retries:
            try:
                async with self._http_session.ws_connect(
                    url,
                    headers=headers,
                    heartbeat=20.0,
                    max_msg_size=0,     # No limit - large IR payloads
                ) as ws:
                    log.debug("ws connected session=%s", session_id)
                    backoff = _WS_BACKOFF_START # RESET ON SUCCESSFUL CONNECT.
                    retries = 0

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                raw = _loads(msg.data)
                                env = _Envelope(
                                    type=raw["type"],
                                    payload=raw.get("payload", {}),
                                )

                                await queue.put(env)
                                if env.type in ("result", "error"):
                                    await queue.put(None) # sentinel
                                    return
                            except Exception as exc:
                                log.warning("ws decode error: %s", exc)

                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            try:
                                raw = _loads(msg.data)
                                env = _Envelope(type=raw["type"], payload=raw.get("payload", {}))
                                await queue.put(env)
                            except Exception as exc:
                                log.warning("ws binary decode error: %s", exc)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            log.warning("ws error frame: %s", ws.exception())
                            break
 
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            log.debug("ws close frame received")
                            await queue.put(None)
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                retries += 1
                if retries > self._max_ws_retries:
                    log.error("ws max retries exceeded session=%s", session_id)
                    await queue.put(
                        _Envelope(
                            type="error",
                            payload={"message": f"WebSocket disconnected: {exc}"},
                        )
                    )
                    await queue.put(None)
                    return
                log.warning(
                    "ws disconnected (attempt %d/%d), retrying in %.1fs: %s",
                    retries,
                    self._max_ws_retries,
                    backoff,
                    exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * _WS_BACKOFF_MUL, _WS_BACKOFF_MAX)
 
        await queue.put(None)
 
    # ------------------------------------------------------------------
    # Internal: drain queue into async generator
    # ------------------------------------------------------------------
 
    @staticmethod
    async def _drain_queue(
        queue: asyncio.Queue[_Envelope | None],
    ) -> AsyncIterator[_Envelope]:
        """Yield from queue until None sentinel is received."""
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item