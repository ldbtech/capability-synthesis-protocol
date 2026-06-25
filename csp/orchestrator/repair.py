"""
csp.orchestrator.repair
~~~~~~~~~~~~~~~~~~~~~~~~~
The Pac-Man regeneration brain — the *pure* logic of self-repair.

When a synthesized capability compiles but blows up at runtime, we don't give
up and we don't blindly re-prompt "try again". We heal it the way a cell heals,
and we navigate the error like Pac-Man navigates a maze:

  • Pac-Man in a maze  — a repair agent traversing the error-states of ONE
    capability; each move "eats" the current error (a pellet).

  • Pigeonhole holes   — every traceback, however unique on the surface, hashes
    into one of a BOUNDED set of error CLASSES (the holes). Each hole has its
    own tailored fix tactic, so we route the repair instead of guessing.

  • Pigeonhole collision — the SAME hole hit twice means Pac-Man is cornered:
    local patching is not working on this class of bug.

  • The warp tunnel    — on a collision Pac-Man warps: a *discontinuous jump*
    out of incremental patching into a from-scratch regrowth.

  • Cell regeneration   — two healing modes:
        DNA repair          = evolve() the code in place (keep what works)
        apoptosis + regrow  = forget the body, synthesize() a fresh one
    DNA repair while we're making progress; regrow when we collide.

  • Scent trail         — the accumulated failure memory, fed into every prompt
    so Pac-Man never re-eats a dead end.

Why pigeonhole is not just flavour: it BOUNDS the work and proves termination.
With H holes, at most C collisions per hole before a warp, and R max regrowths,
the loop halts in O(H·C·(R+1)) LLM calls. No infinite repair.

This module is deliberately I/O-free (no LLM, no sandbox, no registry) so the
routing is unit-testable in isolation. The loop that *drives* it lives in the
Executor (the only place that holds the sandbox + synthesizer + registry).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

# Pull the offending key out of "KeyError: 'NAME'" so we can tell a missing
# credential (UPPER_SNAKE env var) from a missing data arg (lowercase).
_KEYERR_RE = re.compile(r"KeyError:\s*['\"]([^'\"]+)['\"]")


class Hole(str, Enum):
    """The pigeonholes — the bounded set of runtime-error classes a healed
    capability can fall into. Every traceback maps to exactly one."""

    ARG_SHAPE  = "ARG_SHAPE"   # read a key/index that isn't in args
    IMPORT     = "IMPORT"      # imported a module that isn't installed
    TYPE       = "TYPE"        # TypeError / value not JSON-serializable (numpy, pandas)
    LOGIC      = "LOGIC"       # NameError / AttributeError / undefined name
    VALUE      = "VALUE"       # ValueError / arithmetic / domain error
    TIMEOUT    = "TIMEOUT"     # ran too long, killed by the sandbox
    CREDENTIAL = "CREDENTIAL"  # missing API key — NOT a code bug, don't repair here
    UNKNOWN    = "UNKNOWN"     # anything else — patch from the raw traceback


# Per-hole fix tactic. This is the difference between routed repair and blind
# "try again": each class of bug gets the specific instruction that fixes it.
_TACTIC: dict[Hole, str] = {
    Hole.ARG_SHAPE: (
        "The code raised a KeyError/IndexError — it read a key or index that is "
        "NOT present in args. Rewrite run(args) to read ONLY the keys listed "
        "above, using args.get(key, default) for anything optional. Never index "
        "a list/dict without checking it exists first."
    ),
    Hole.IMPORT: (
        "The code imported a module that isn't installed in this environment. "
        "Rewrite it using only the Python standard library (math, statistics, "
        "json, re, csv, datetime, collections, …); use pandas/numpy ONLY if "
        "truly necessary. Remove the failing import entirely."
    ),
    Hole.TYPE: (
        "A value was the wrong type or not JSON-serializable — typically a "
        "numpy/pandas scalar or object leaking into the result. Convert EVERY "
        "returned value to a plain Python int/float/str/bool/list/dict before "
        "returning. Fix any other TypeError shown in the traceback."
    ),
    Hole.LOGIC: (
        "There is a logic bug — an undefined name, a wrong attribute, or a "
        "variable used before assignment. Read the traceback and fix it: define "
        "every name before use and call only attributes that exist."
    ),
    Hole.VALUE: (
        "The code raised a ValueError or arithmetic error at runtime. Add the "
        "guards needed for the given args (empty inputs, zero division, bad "
        "casts) so it returns a valid dict instead of raising."
    ),
    Hole.TIMEOUT: (
        "The code ran too long and was killed. Make it finish quickly: remove "
        "unbounded loops, avoid network retries/sleeps, and prefer a single "
        "vectorized pass over the data."
    ),
    Hole.UNKNOWN: (
        "The code failed at runtime. Diagnose the cause from the traceback "
        "below and fix it so run(args) returns a valid dict."
    ),
}

# Shared generality reminder appended to every repair instruction — repairing a
# bug must never collapse the capability into a one-off answer.
_GENERALITY = (
    "Keep the capability GENERAL: read every specific from args with sensible "
    "defaults, do not hardcode the values of this particular request. Return a "
    "JSON-serializable dict. Return the COMPLETE updated run(args)."
)


def classify_error(error: str | None, traceback: str | None) -> Hole:
    """Hash a sandbox failure into exactly one pigeonhole.

    Order matters: more specific classes are checked before the KeyError-ish
    ones, because a missing-credential KeyError and an arg-shape KeyError look
    alike on the surface.
    """
    err = (error or "")
    tb  = (traceback or "")
    blob = f"{err}\n{tb}"
    low  = blob.lower()

    if "timed out" in low or "timeout" in low:
        return Hole.TIMEOUT

    # A KeyError on an env-style key is a missing credential, not a code bug —
    # route it out so the credential flow handles it. We can't rely on the
    # traceback mentioning os.environ (the sandbox exec's code from a string, so
    # source lines aren't captured), so we read the key out of the error: env
    # vars are UPPER_SNAKE by universal convention; data args here are lowercase.
    if "KeyError" in err:
        m = _KEYERR_RE.search(err)
        key = m.group(1) if m else ""
        if "environ" in tb or (key.isidentifier() and key.isupper()):
            return Hole.CREDENTIAL

    if "ModuleNotFoundError" in blob or "ImportError" in blob:
        return Hole.IMPORT

    if (
        "not JSON serializable" in blob
        or "Object of type" in blob          # json's "Object of type X is not JSON serializable"
        or "TypeError" in err
    ):
        return Hole.TYPE

    if "KeyError" in err or "IndexError" in err:
        return Hole.ARG_SHAPE

    if "NameError" in err or "AttributeError" in err or "UnboundLocalError" in err:
        return Hole.LOGIC

    if "ValueError" in err or "ZeroDivisionError" in err or "ArithmeticError" in err:
        return Hole.VALUE

    return Hole.UNKNOWN


def describe_args(args: dict[str, Any]) -> str:
    """One-line-per-key description of the args the code will actually receive,
    so the repair reads the right keys instead of guessing (mirrors the shape
    hint the synthesizer got at birth)."""
    if not args:
        return "args will be an EMPTY dict."
    lines = ["run(args) receives args with these keys:"]
    for key, val in args.items():
        if isinstance(val, list):
            sample = val[0] if val else None
            if isinstance(sample, dict):
                lines.append(
                    f"- args[{key!r}]: list of {len(val)} dict rows; "
                    f"each row has keys {list(sample.keys())}"
                )
            else:
                lines.append(f"- args[{key!r}]: list of {len(val)} items, e.g. {sample!r}")
        elif isinstance(val, dict):
            lines.append(f"- args[{key!r}]: dict with keys {list(val.keys())}")
        else:
            lines.append(f"- args[{key!r}]: {type(val).__name__} = {val!r}")
    return "\n".join(lines)


def _trail_summary(trail: list[tuple[Hole, str | None]]) -> str:
    """Render the scent trail — what's already been tried and failed — so the
    LLM doesn't walk back into a dead end."""
    if not trail:
        return ""
    lines = ["Already tried and STILL failed (do not repeat these mistakes):"]
    for i, (hole, err) in enumerate(trail, 1):
        lines.append(f"  {i}. [{hole.value}] {err}")
    return "\n".join(lines)


def repair_instruction(
    hole: Hole,
    error: str | None,
    traceback: str | None,
    args: dict[str, Any],
    trail: list[tuple[Hole, str | None]],
) -> str:
    """Build the DNA-repair instruction handed to Synthesizer.evolve() — the
    hole's tactic + the live traceback + the arg shape + the scent trail."""
    parts = [
        "The current code FAILED at runtime in the sandbox.",
        _TACTIC.get(hole, _TACTIC[Hole.UNKNOWN]),
        "",
        describe_args(args),
        "",
        f"Runtime error: {error}",
    ]
    if traceback:
        parts += ["Traceback:", traceback.strip()]
    summary = _trail_summary(trail)
    if summary:
        parts += ["", summary]
    parts += ["", _GENERALITY]
    return "\n".join(parts)


def regrowth_context(
    args: dict[str, Any],
    trail: list[tuple[Hole, str | None]],
) -> str:
    """Build the context for a WARP — a from-scratch synthesize(). Incremental
    patching has stalled, so we hand the fresh synthesis the full failure
    history (the power-pellet memory) plus the real arg shape, and tell it to
    take a DIFFERENT approach."""
    parts = [
        "This capability is being REGENERATED from scratch because incremental "
        "patching kept failing on the same class of error. Take a DIFFERENT, "
        "simpler approach than before.",
        "",
        describe_args(args),
    ]
    summary = _trail_summary(trail)
    if summary:
        parts += ["", summary]
    return "\n".join(parts)
