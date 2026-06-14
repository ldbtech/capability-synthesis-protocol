"""
examples/basic_usage.py
~~~~~~~~~~~~~~~~~~~~~~~
All client usage patterns in one file.
Run with: python examples/basic_usage.py
"""

import asyncio
from csp.client import (
    BrainClient,
    ClientSession,
    StreamableHTTP,
    ElicitKind,
    ElicitRequest,
    ElicitResponse,
    EventKind,
    Result,
    StreamEvent,
)


# ---------------------------------------------------------------------------
# Pattern 1 — Minimal blocking call
# ---------------------------------------------------------------------------

async def example_minimal():
    transport = StreamableHTTP("http://localhost:8000")
    client    = BrainClient(transport)

    async with client:
        result = await client.run("predict customer churn")
        print(result.summary)


# ---------------------------------------------------------------------------
# Pattern 2 — Streaming with live event output
# ---------------------------------------------------------------------------

async def example_streaming():
    transport = StreamableHTTP("http://localhost:8000")
    client    = BrainClient(transport)

    @client.on_event
    async def show_event(event: StreamEvent) -> None:
        icon = {
            EventKind.PLANNING:       "🧠",
            EventKind.CAPABILITY:     "⚙️ ",
            EventKind.CAPABILITY_END: "✅",
            EventKind.LOG:            "   ",
            EventKind.DONE:           "🏁",
        }.get(event.kind, "  ")
        print(f"{icon} {event.message}")

    async with client:
        async for item in client.stream("run etl pipeline and train model"):
            if isinstance(item, Result):
                print(f"\nSummary: {item.summary}")
                print(f"Status:  {item.status.name}")
                print(f"Time:    {item.duration:.2f}s")


# ---------------------------------------------------------------------------
# Pattern 3 — Elicitation with decorator handler
# ---------------------------------------------------------------------------

async def example_elicitation():
    transport = StreamableHTTP("http://localhost:8000")
    client    = BrainClient(transport)

    @client.on_elicit
    async def handle_elicit(request: ElicitRequest) -> ElicitResponse:
        print(f"\n[{request.kind.name}] {request.question}")

        if request.kind == ElicitKind.APPROVAL:
            answer = input("  (yes/no) > ").strip().lower() or "no"

        elif request.kind == ElicitKind.CHOICE:
            for i, opt in enumerate(request.options, 1):
                print(f"  {i}. {opt}")
            idx = int(input("  Choice > ").strip()) - 1
            answer = request.options[idx]

        else:  # INPUT
            answer = input("  > ").strip()

        return ElicitResponse(request_id=request.id, value=answer)

    async with client:
        result = await client.run("deploy latest model to production")
        print(result.summary)


# ---------------------------------------------------------------------------
# Pattern 4 — Manual elicitation in stream loop
# ---------------------------------------------------------------------------

async def example_manual_elicitation():
    transport = StreamableHTTP("http://localhost:8000")
    client    = BrainClient(transport)

    async with client.session() as session:
        async for item in session.stream("setup kafka cluster"):
            if isinstance(item, StreamEvent):
                print(f"  {item.message}")

            elif isinstance(item, ElicitRequest):
                # Full control — developer handles however they want
                print(f"\nSystem needs input: {item.question}")
                value = input("> ")
                await session.respond(ElicitResponse(item.id, value))

            elif isinstance(item, Result):
                print(f"\nDone: {item.summary}")
                for cap in item.capabilities:
                    status = "✓" if cap.success else "✗"
                    print(f"  {status} {cap.name} ({cap.duration:.1f}s)")


# ---------------------------------------------------------------------------
# Pattern 5 — Multi-turn persistent session
# ---------------------------------------------------------------------------

async def example_multi_turn():
    transport = StreamableHTTP("http://localhost:8000")
    client    = BrainClient(transport, keep_alive=True)

    @client.on_elicit
    async def auto_approve(request: ElicitRequest) -> ElicitResponse:
        return ElicitResponse(request.id, "yes")

    async with client:
        r1 = await client.run("load the sales dataset")
        print(f"Step 1: {r1.summary}\n")

        r2 = await client.run("clean and normalise it")
        print(f"Step 2: {r2.summary}\n")

        r3 = await client.run("train a churn prediction model")
        print(f"Step 3: {r3.summary}\n")

        r4 = await client.run("deploy model to staging")
        print(f"Step 4: {r4.summary}\n")


# ---------------------------------------------------------------------------
# Pattern 6 — Inspecting the full result object
# ---------------------------------------------------------------------------

async def example_inspect_result():
    transport = StreamableHTTP("http://localhost:8000")
    client    = BrainClient(transport)

    async with client:
        result = await client.run("run full ml pipeline")

    print(f"Status:   {result.status.name}")
    print(f"Summary:  {result.summary}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"\nCapabilities run ({len(result.capabilities)}):")
    for cap in result.capabilities:
        status = "✓" if cap.success else f"✗ {cap.error}"
        print(f"  {status}  {cap.name}  ({cap.duration:.1f}s)")

    if result.elicitations:
        print(f"\nElicitations ({len(result.elicitations)}):")
        for req, resp in result.elicitations:
            print(f"  Q: {req.question}")
            print(f"  A: {resp.value}")

    print(f"\nOutput: {result.output}")


if __name__ == "__main__":
    # Change to whichever example you want to run
    asyncio.run(example_streaming())