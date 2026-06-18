"""
csp — Capability Synthesis Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public API. Import everything you need from here:

    from csp import Orchestrator, ElicitRequired, AnthropicLLM

Server pattern (what a developer writes):

    from csp import Orchestrator, ElicitRequired, AnthropicLLM

    app = Orchestrator(
        "my-server",
        llm=AnthropicLLM(api_key="sk-ant-..."),   # or set ANTHROPIC_API_KEY env var
    )

    @app.capability("greet")
    async def greet(name: str) -> dict:
        return {"message": f"Hello, {name}!"}

    if __name__ == "__main__":
        app.run()
"""

from .orchestrator.server import Orchestrator
from .orchestrator.executor import ElicitRequired
from .orchestrator.borrow import BorrowError, BorrowedCapability
from .orchestrator.credentials import CredentialRequired, CredentialStore
from .llm.anthropic import AnthropicLLM
from .llm.base import BaseLLM, LLMMessage, LLMResponse

__all__ = [
    "Orchestrator",
    "ElicitRequired",
    "BorrowError",
    "BorrowedCapability",
    "CredentialRequired",
    "CredentialStore",
    "AnthropicLLM",
    "BaseLLM",
    "LLMMessage",
    "LLMResponse",
]
