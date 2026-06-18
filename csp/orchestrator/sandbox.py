"""
csp.orchestrator.sandbox
~~~~~~~~~~~~~~~~~~~~~~~~~~
PythonSandbox — runs LLM-generated capability code in an isolated subprocess.

This is what makes a *synthesized* capability actually execute: the synthesizer
writes a Python function, and the sandbox runs it in a fresh `python`
subprocess with:

  - a timeout (killed if it runs too long)
  - arguments passed in as JSON on stdin
  - the return value read back as JSON on stdout
  - the developer's own interpreter, so it can use whatever is installed
    in their project venv (pandas, numpy, ...) — the capability is part of
    THEIR app, not the CSP library

The generated code must define a single entrypoint function:

    def run(args: dict) -> dict:
        ...
        return {...}

Security note
------------
A subprocess gives process isolation and a hard timeout, but it is NOT a
hard security boundary — generated code runs with the same OS permissions
as the server. For untrusted multi-tenant use, swap `_HARNESS` execution
for a container (Docker/gVisor) by overriding `PythonSandbox.run`. The
interface is deliberately small so that drop-in is easy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger("csp.sandbox")

_DEFAULT_TIMEOUT = 30.0


# The harness runs inside the subprocess. It reads {code, args} as JSON from
# stdin, exec's the code to define `run`, calls run(args), and prints the
# result as a single JSON line to stdout. All capability stdout/prints are
# redirected to stderr so they can't corrupt the JSON result channel.
_HARNESS = r"""
import sys, json, io, contextlib, traceback, os

payload = json.loads(sys.stdin.read())
code = payload["code"]
args = payload.get("args", {})
entrypoint = payload.get("entrypoint", "run")

ns = {}
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(code, ns)
        if entrypoint not in ns:
            raise NameError(f"generated code must define {entrypoint}(args)")
        result = ns[entrypoint](args)
    out = {"ok": True, "result": result, "stdout": buf.getvalue()}
except Exception as exc:
    out = {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "stdout": buf.getvalue(),
    }

# Result channel: exactly one JSON line on real stdout.
sys.stderr.write(buf.getvalue())
print(json.dumps(out, default=str))
"""


@dataclass(slots=True)
class SandboxResult:
    """Outcome of running generated code in the sandbox."""
    ok:        bool
    result:    Any                = None
    error:     Optional[str]      = None
    traceback: Optional[str]      = None
    stdout:    str                = ""
    duration:  float              = 0.0


class PythonSandbox:
    """
    Runs generated Python in an isolated subprocess.

    Parameters
    ----------
    timeout:
        Max seconds the generated code may run before it's killed.
    python:
        Interpreter to use. Defaults to the same one running the server,
        so generated code sees the developer's installed packages.
    """

    __slots__ = ("_timeout", "_python", "_env")

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        python: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self._timeout = timeout
        self._python  = python or sys.executable
        # Extra environment variables merged into the subprocess. The library
        # sets none by default — apps inject what their generated code needs
        # (e.g. {"MPLBACKEND": "Agg"} for headless matplotlib rendering).
        self._env     = env or {}

    async def run(
        self,
        code: str,
        args: dict[str, Any],
        *,
        entrypoint: str = "run",
        extra_env: Optional[dict[str, str]] = None,
    ) -> SandboxResult:
        """Execute `code`, calling entrypoint(args), return a SandboxResult.

        extra_env: per-call env vars (e.g. API credentials) merged on top of
                   self._env so concurrent calls don't interfere.
        """
        loop = asyncio.get_event_loop()
        t0   = loop.time()

        payload = json.dumps({
            "code":       code,
            "args":       args,
            "entrypoint": entrypoint,
        })

        # Inherit the parent environment (PATH, venv, etc.) and overlay the
        # app-provided extras. matplotlib & friends read these from os.environ.
        import os
        sub_env = {**os.environ, **self._env, **(extra_env or {})}

        try:
            proc = await asyncio.create_subprocess_exec(
                self._python, "-I", "-c", _HARNESS,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=sub_env,
            )
        except Exception as exc:
            return SandboxResult(ok=False, error=f"failed to spawn sandbox: {exc}")

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(payload.encode()),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                ok=False,
                error=f"capability timed out after {self._timeout}s",
                duration=loop.time() - t0,
            )

        duration = loop.time() - t0
        captured = stderr.decode(errors="replace")

        # The harness prints exactly one JSON line on stdout.
        line = stdout.decode(errors="replace").strip().splitlines()
        if not line:
            return SandboxResult(
                ok=False,
                error=f"sandbox produced no output (exit {proc.returncode})",
                traceback=captured,
                duration=duration,
            )

        try:
            data = json.loads(line[-1])
        except json.JSONDecodeError as exc:
            return SandboxResult(
                ok=False,
                error=f"could not parse sandbox output: {exc}",
                traceback=captured,
                duration=duration,
            )

        return SandboxResult(
            ok=data.get("ok", False),
            result=data.get("result"),
            error=data.get("error"),
            traceback=data.get("traceback") or captured or None,
            stdout=data.get("stdout", ""),
            duration=duration,
        )
