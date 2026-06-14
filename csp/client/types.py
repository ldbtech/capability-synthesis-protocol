"""
csp.client.types
~~~~~~~~~~~~~~~~~~~~
All client-facing types. Zero JSON RPC leakage - these are only objects a developer ever touches. 

Design goals: 
    - Immutable data classes via __slots__ for memory effeciency
    - Enum-based descriminators so isinstance() checks are O(1)
    - __repr__ trimmed for large payloads (avoids log spam)
"""

from __future__ import annotations

from multiprocessing import Value
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------
# Enumeration
# ---------------------------------------------------------
class EventKind(Enum):
    """ 
    Discriminator for SteamEvent - Lets callers match on kind without isinstance() chains acrosss subclass.
    """
    PLANNING            = auto() # Planning is building the execution graph.
    CAPABILITY          = auto() # A capability is about to run / is running 
    CAPABILITY_END      = auto() # A capability finished (success or failure)
    LOG                 = auto() # ree-form log line from executor
    ELICIT              = auto() # system needs input — see ElicitRequest
    DONE                = auto() # terminal event; Result follows immediately

class ElicitKind(Enum):
    """ What kind of human input is needed """
    APPROVAL    = auto() # yes / no
    INPUT       = auto() # free-form text
    CHOICE      = auto() # one of N options

class ResultStatus(Enum):
    OK          = auto()
    PARTIAL     = auto() # Some capabilities failed, rest completed.
    ERROR       = auto()

# ---------------------------------------------------------
# Stream Events
# ---------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StreamEvent:
    """
    A single event emitted during execution

    kind            : EventKind discriminator
    message         : human-readable description (LLM-generated or system)
    capability      : name of the capability this event belongs to, if any.
    metadata        : arbitraty extra data (e.g progress % capability version)
    ts              : Unix timestamp (float) - set automatically
    """
    kind:       EventKind
    message:    str
    capability: Optional[str]       = None
    metadata:   dict[str, Any]      = field(default_factory=dict)
    ts:         float               = field(default_factory=time.monotonic)

    def __repr__(self) -> str:
        meta = f"meta={self.metadata}" if self.metadata else ""
        cap = f" cap={self.capability!r}" if self.capability else ""
        return f"<StreamEvent {self.kind.name}{cap} {self.message!r:.60}{meta}"

# ---------------------------------------------------------------------------
# Elicitation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ElicitRequest:
    """
    Emitted when the orchestrator needs a human decision before proceeding.
 
    id          : unique request id — must be echoed back in ElicitResponse
    kind        : APPROVAL | INPUT | CHOICE
    question    : human-readable question
    options     : populated for CHOICE kind only
    context     : optional extra context the developer may surface to the user
    capability  : which capability triggered this (for display purposes)
    """
    id:         str
    kind:       ElicitKind
    question:   str
    options:    tuple[str, ...]            = field(default_factory=tuple)
    context:    Optional[str]             = None
    capability: Optional[str]             = None
 
    def __repr__(self) -> str:
        opts = f" options={self.options}" if self.options else ""
        return f"<ElicitRequest {self.kind.name} id={self.id!r} {self.question!r:.60}{opts}>"
     

@dataclass(frozen=True, slots=True)
class ElicitResponse:
    """
    Developer constructs this ad passes it back to session.respond().

    request_id          : must watch ElicitResponse.id
    value               : the answer - "yes/no" for approval, free text for input
                          one for ElicitRequest.options for choice
    """
    request_id: str
    value:      str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("ElicitResponse.value cannot be blank")
    
    def __repr__(self) -> str:
        return f"<ElicitResponse id={self.request_id!r} value={self.value!r:.40}"


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------
 
@dataclass(frozen=True, slots=True)
class CapabilityResult:
    """Outcome of a single capability execution."""
    name:     str
    success:  bool
    output:   dict[str, Any]              = field(default_factory=dict)
    error:    Optional[str]               = None
    duration: Optional[float]             = None   # seconds
 
    def __repr__(self) -> str:
        status = "ok" if self.success else f"err={self.error!r:.40}"
        dur    = f" {self.duration:.2f}s" if self.duration is not None else ""
        return f"<CapabilityResult {self.name!r} {status}{dur}>"
   

@dataclass(frozen=True, slots=True)
class Result:
    """
    Terminal object — the last thing a session.run() or stream() yields.
 
    status          : OK | PARTIAL | ERROR
    summary         : LLM-generated natural language summary of what happened
    capabilities    : ordered list of every capability that ran
    elicitations    : every (request, response) pair that occurred
    output          : merged final output across all capabilities
    error           : top-level error message if status is ERROR
    duration        : wall-clock seconds for the full execution
    """
    status:        ResultStatus
    summary:       str
    capabilities:  tuple[CapabilityResult, ...]  = field(default_factory=tuple)
    elicitations:  tuple[tuple[ElicitRequest, ElicitResponse], ...] = field(default_factory=tuple)
    output:        dict[str, Any]                = field(default_factory=dict)
    error:         Optional[str]                = None
    duration:      Optional[float]              = None
 
    def ok(self) -> bool:
        return self.status == ResultStatus.OK
 
    def __repr__(self) -> str:
        caps = len(self.capabilities)
        dur  = f" {self.duration:.2f}s" if self.duration is not None else ""
        return (
            f"<Result {self.status.name} caps={caps}"
            f" summary={self.summary!r:.80}{dur}>"
        )

# ---------------------------------------------------------------------------
# Internal Wire envolope (not public - used by transport + session only)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Envelope:
    """
    Raw decoded message off the wire before it is mapped to a public type.
    Transport layer produces these; session layer consumes them.
    Not part of the public API.
    """
    type:    str            # "event" | "elicit" | "result" | "error"
    payload: dict[str, Any]