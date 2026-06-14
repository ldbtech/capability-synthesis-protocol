"""
tests/test_csp.py
~~~~~~~~~~~~~~~~~~
Core CSP tests that run WITHOUT any network/LLM calls. A FakeLLM stands in for
the provider, so the planner/synthesizer plumbing is exercised deterministically.

Run:
    pip install -e ".[dev]"
    pytest -q
"""

from __future__ import annotations

import pytest

from csp import Orchestrator
from csp.llm import BaseLLM, LLMResponse
from csp.orchestrator.capability import capability_from_function
from csp.orchestrator.sandbox import PythonSandbox
from csp.orchestrator.synthesizer import _assemble_spec, _extract_block
from csp.orchestrator.planner_store import PlannerStore


# ── A scripted LLM so tests never hit the network ─────────────────────────────
class FakeLLM(BaseLLM):
    def __init__(self, content: str = "") -> None:
        self._content = content

    async def complete(self, messages, *, max_tokens=4096, temperature=0.0, system=None):
        return LLMResponse(content=self._content, input_tokens=1, output_tokens=1)


# ── Sandbox: real code execution ──────────────────────────────────────────────
async def test_sandbox_runs_real_code():
    sb = PythonSandbox(timeout=10)
    code = "def run(args):\n    return {'sum': sum(args['xs'])}"
    res = await sb.run(code, {"xs": [1, 2, 3, 4]})
    assert res.ok
    assert res.result == {"sum": 10}


async def test_sandbox_captures_errors():
    sb = PythonSandbox(timeout=10)
    res = await sb.run("def run(args):\n    return 1/0", {})
    assert not res.ok
    assert "ZeroDivisionError" in res.error


async def test_sandbox_enforces_timeout():
    sb = PythonSandbox(timeout=1)
    res = await sb.run("def run(args):\n    import time; time.sleep(5)\n    return {}", {})
    assert not res.ok
    assert "timed out" in res.error


async def test_sandbox_extra_env_applied():
    # env is merged into the subprocess; verify it reaches os.environ.
    sb = PythonSandbox(timeout=10, env={"CSP_TEST_FLAG": "42"})
    code = "def run(args):\n    import os\n    return {'flag': os.environ.get('CSP_TEST_FLAG')}"
    res = await sb.run(code, {})
    assert res.ok and res.result == {"flag": "42"}


# ── Registered capability: kwargs filtering ───────────────────────────────────
async def test_registered_capability_filters_unknown_kwargs():
    async def greet(name: str) -> dict:
        return {"msg": f"hi {name}"}

    cap = capability_from_function("greet", greet)
    # Extra contextual kwarg must be dropped, not crash.
    out = await cap.invoke(name="Ada", _elicit_response="yes")
    assert out == {"msg": "hi Ada"}


async def test_registered_capability_passes_varkwargs():
    async def sink(name: str, **kw) -> dict:
        return {"name": name, "extra": sorted(kw)}

    cap = capability_from_function("sink", sink)
    out = await cap.invoke(name="x", a=1, b=2)
    assert out == {"name": "x", "extra": ["a", "b"]}


# ── Synthesizer: two-block parsing assembles a valid spec ─────────────────────
def test_assemble_spec_from_two_blocks():
    raw = (
        "```python\n"
        "def run(args):\n    return {'doubled': args['n'] * 2}\n"
        "```\n"
        "```json\n"
        '{"description": "double it", "params_schema": {}, '
        '"result_schema": {}, "steps": ["double"]}\n'
        "```"
    )
    spec = _assemble_spec(raw, "double_it")
    params = spec["params"]
    assert params["capability_id"] == "double_it"
    assert params["execution"]["target"] == "python"
    assert "def run" in params["execution"]["code"]


def test_extract_block_missing_returns_empty():
    assert _extract_block("no fences here", "python") == ""


# ── Orchestrator: direct capability call + forget ─────────────────────────────
async def test_call_capability_direct(tmp_path):
    app = Orchestrator("t", llm=FakeLLM(), planner_dir=None)

    @app.capability("add")
    async def add(a: int, b: int) -> dict:
        return {"sum": a + b}

    out = await app.call_capability("add", a=2, b=3)
    assert out == {"sum": 5}


async def test_call_capability_unknown_raises():
    app = Orchestrator("t", llm=FakeLLM(), planner_dir=None)
    with pytest.raises(KeyError):
        await app.call_capability("nope")


async def test_forget_removes_synthesized():
    from csp.orchestrator.capability import SynthesizedCapability

    app = Orchestrator("t", llm=FakeLLM(), planner_dir=None)
    spec = {"jsonrpc": "2.0", "method": "csp.capability.invoke",
            "params": {"capability_id": "x", "execution":
                       {"target": "python", "entrypoint": "run", "code": "def run(args): return {}"}}}
    app._registry._synthesized["x"] = SynthesizedCapability(name="x", spec=spec)
    assert app._registry.exists("x")
    removed = await app.forget("x")
    assert removed and not app._registry.exists("x")


# ── PlannerStore: persistence round-trips synthesized capabilities ────────────
def test_planner_store_roundtrip(tmp_path):
    from csp.orchestrator.capability import SynthesizedCapability

    store = PlannerStore(str(tmp_path / "planner"))
    spec = {"jsonrpc": "2.0", "method": "csp.capability.invoke",
            "params": {"capability_id": "calc", "description": "c",
                       "execution": {"target": "python", "entrypoint": "run",
                                     "code": "def run(args):\n    return {'ok': True}"}}}
    cap = SynthesizedCapability(name="calc", spec=spec, description="c")
    store.save_capability(cap)

    loaded = store.load_capabilities()
    assert len(loaded) == 1 and loaded[0].name == "calc"
    assert (store.caps_dir / "calc.py").exists()   # readable source persisted

    store.delete_capability("calc")
    assert store.load_capabilities() == []
