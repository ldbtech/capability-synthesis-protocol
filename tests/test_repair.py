"""
tests/test_repair.py
~~~~~~~~~~~~~~~~~~~~~
The Pac-Man self-repair loop — error pigeonholing + DNA-repair/warp regrowth.

Two layers, both network-free:
  1. The pure brain (csp.orchestrator.repair): error → hole routing, instruction
     and regrowth-context building.
  2. The Executor._heal loop end-to-end with a REAL sandbox and a scripted LLM
     that hands back fixed code — so we prove broken synthesized code actually
     heals, warps, exhausts, and bails on credentials.
"""

from __future__ import annotations

from csp.llm import BaseLLM, LLMResponse
from csp.orchestrator.capability import SynthesizedCapability
from csp.orchestrator.elicitation import ElicitationManager
from csp.orchestrator.executor import Executor
from csp.orchestrator.planner import PlanStep
from csp.orchestrator.registry import CapabilityRegistry
from csp.orchestrator.sandbox import PythonSandbox
from csp.orchestrator.repair import (
    Hole,
    classify_error,
    describe_args,
    regrowth_context,
    repair_instruction,
)


# ── A scripted LLM that returns a queue of two-block responses ────────────────
class ScriptedLLM(BaseLLM):
    """Pops one canned response per complete() call so synthesize()/evolve()
    are deterministic. Tracks how many times it was called."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, *, max_tokens=4096, temperature=0.0, system=None):
        self.calls += 1
        content = self._responses.pop(0) if self._responses else _two_block(
            "def run(args):\n    raise KeyError('still broken')"
        )
        return LLMResponse(content=content, input_tokens=1, output_tokens=1)


def _two_block(code: str, description: str = "fixed") -> str:
    return (
        "```python\n" + code + "\n```\n"
        "```json\n"
        f'{{"description": "{description}", "params_schema": {{}}, '
        '"result_schema": {}, "tags": [], "steps": ["run"]}\n'
        "```"
    )


def _broken_cap(name: str, code: str) -> SynthesizedCapability:
    spec = {
        "jsonrpc": "2.0", "method": "csp.capability.invoke",
        "params": {"capability_id": name, "description": "broken",
                   "execution": {"target": "python", "entrypoint": "run", "code": code}},
    }
    return SynthesizedCapability(name=name, spec=spec, description="broken")


def _executor(llm, *, max_repair=3, collision=2, regrowths=1) -> Executor:
    from csp.orchestrator.synthesizer import Synthesizer
    return Executor(
        registry=CapabilityRegistry(),
        synthesizer=Synthesizer(llm),
        elicitation_manager=ElicitationManager(timeout=5),
        goal="test goal",
        sandbox=PythonSandbox(timeout=10),
        max_repair_attempts=max_repair,
        repair_collision_limit=collision,
        max_regrowths=regrowths,
    )


async def _drive_heal(ex: Executor, cap, step) -> tuple[dict, list[dict]]:
    """Run _heal to completion, returning (holder, streamed_events)."""
    holder: dict = {}
    events: list[dict] = []
    async for ev in ex._heal(cap, step, holder):
        events.append(ev)
    return holder, events


# ════════════════════════════════════════════════════════════════════════════
# Layer 1 — the pure brain
# ════════════════════════════════════════════════════════════════════════════

def test_classify_routes_each_hole():
    assert classify_error("KeyError: 'rows'", "  rows = args['rows']") is Hole.ARG_SHAPE
    assert classify_error("IndexError: list index out of range", "x[3]") is Hole.ARG_SHAPE
    assert classify_error("ModuleNotFoundError: No module named 'foo'", "") is Hole.IMPORT
    assert classify_error("ImportError: cannot import bar", "") is Hole.IMPORT
    assert classify_error("TypeError: Object of type int64 is not JSON serializable", "") is Hole.TYPE
    assert classify_error("NameError: name 'z' is not defined", "") is Hole.LOGIC
    assert classify_error("AttributeError: 'NoneType' has no attribute 'x'", "") is Hole.LOGIC
    assert classify_error("ValueError: bad literal", "") is Hole.VALUE
    assert classify_error("ZeroDivisionError: division by zero", "") is Hole.VALUE
    assert classify_error("capability timed out after 30s", "") is Hole.TIMEOUT
    assert classify_error("RuntimeError: something weird", "") is Hole.UNKNOWN


def test_classify_credential_beats_arg_shape():
    # An UPPER_SNAKE KeyError is a missing credential, not arg shape — even when
    # the traceback has no os.environ line (sandbox exec drops source lines).
    assert classify_error("KeyError: 'OPENWEATHER_API_KEY'", "") is Hole.CREDENTIAL
    # The os.environ traceback signal also still works on its own.
    tb = 'File "<cap>", line 3\n    key = os.environ["WEATHER_KEY"]'
    assert classify_error("KeyError: 'WEATHER_KEY'", tb) is Hole.CREDENTIAL
    # A lowercase data key stays arg-shape, not credential.
    assert classify_error("KeyError: 'rows'", "") is Hole.ARG_SHAPE


def test_describe_args_lists_keys_and_row_shape():
    desc = describe_args({"rows": [{"a": 1, "b": 2}], "k": 3})
    assert "args['rows']" in desc and "['a', 'b']" in desc
    assert "args['k']" in desc and "int" in desc
    assert describe_args({}) == "args will be an EMPTY dict."


def test_repair_instruction_carries_tactic_traceback_and_trail():
    trail = [(Hole.ARG_SHAPE, "KeyError: 'rows'")]
    instr = repair_instruction(
        Hole.ARG_SHAPE, "KeyError: 'rows'", "tb-here", {"rows": []}, trail
    )
    assert "args.get(" in instr               # the ARG_SHAPE tactic
    assert "tb-here" in instr                  # the live traceback
    assert "do not repeat" in instr.lower()    # the scent trail
    assert "GENERAL" in instr                  # generality contract preserved


def test_regrowth_context_includes_history_and_different_approach():
    ctx = regrowth_context({"rows": []}, [(Hole.IMPORT, "ModuleNotFoundError: numpy")])
    assert "DIFFERENT" in ctx
    assert "ModuleNotFoundError" in ctx


# ════════════════════════════════════════════════════════════════════════════
# Layer 2 — Executor._heal end-to-end (real sandbox)
# ════════════════════════════════════════════════════════════════════════════

async def test_green_on_first_run_skips_repair():
    llm = ScriptedLLM([])
    ex = _executor(llm)
    cap = _broken_cap("ok", "def run(args):\n    return {'v': 1}")
    holder, events = await _drive_heal(ex, cap, PlanStep(capability="ok", args={}))
    assert holder["sb"].ok and holder["sb"].result == {"v": 1}
    assert llm.calls == 0                      # no LLM touched when code works
    assert not any("repair" in e["params"]["message"] for e in events)


async def test_dna_repair_heals_arg_shape_error():
    fixed = _two_block("def run(args):\n    return {'v': args.get('missing', 0)}")
    llm = ScriptedLLM([fixed])
    ex = _executor(llm, collision=2)
    cap = _broken_cap("getit", "def run(args):\n    return {'v': args['missing']}")
    holder, events = await _drive_heal(ex, cap, PlanStep(capability="getit", args={}))

    assert holder["sb"].ok and holder["sb"].result == {"v": 0}
    assert llm.calls == 1                       # one evolve() call
    msgs = " ".join(e["params"]["message"] for e in events)
    assert "🟡 repair 1/3" in msgs and "ARG_SHAPE" in msgs and "healed" in msgs
    # Only the GREEN version is persisted to the registry.
    synth = await ex._registry.list_synthesized()
    assert len(synth) == 1 and synth[0].code.endswith("args.get('missing', 0)}")


async def test_warp_regrows_after_collision():
    # collision_limit=1 → the first failure of a hole immediately warps.
    fixed = _two_block("def run(args):\n    return {'ok': True}")
    llm = ScriptedLLM([fixed])
    ex = _executor(llm, collision=1, regrowths=1)
    cap = _broken_cap("hard", "def run(args):\n    return {'v': args['nope']}")
    holder, events = await _drive_heal(ex, cap, PlanStep(capability="hard", args={}))

    assert holder["sb"].ok and holder["sb"].result == {"ok": True}
    msgs = " ".join(e["params"]["message"] for e in events)
    assert "🔵 warp" in msgs and "cornered" in msgs


async def test_exhaustion_returns_failure_and_persists_nothing():
    # LLM keeps returning broken code → loop exhausts the budget.
    broken = _two_block("def run(args):\n    return {'v': args['nope']}")
    llm = ScriptedLLM([broken, broken, broken, broken])
    ex = _executor(llm, max_repair=3, collision=5, regrowths=0)
    cap = _broken_cap("doomed", "def run(args):\n    return {'v': args['nope']}")
    holder, events = await _drive_heal(ex, cap, PlanStep(capability="doomed", args={}))

    assert not holder["sb"].ok
    assert llm.calls == 3                        # exactly the budget, then stop
    # Never cached: a capability that never ran green stays out of the registry.
    assert await ex._registry.list_synthesized() == []


async def test_credential_error_bails_without_calling_llm():
    llm = ScriptedLLM([])
    ex = _executor(llm)
    cap = _broken_cap(
        "needs_key",
        "def run(args):\n    import os\n    return {'k': os.environ['MISSING_API_KEY']}",
    )
    holder, events = await _drive_heal(ex, cap, PlanStep(capability="needs_key", args={}))

    assert not holder["sb"].ok
    assert llm.calls == 0                         # credential != code bug, no repair
    assert any("credential" in e["params"]["message"].lower() for e in events)
