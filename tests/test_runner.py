"""Status derivation edge cases that only show up with real, slow agents."""
import asyncio
from types import SimpleNamespace

from app.runner import Runner


class StubGraph:
    def __init__(self, snap):
        self.snap = snap

    async def aget_state(self, config):
        return self.snap


async def test_running_plan_is_never_reported_terminal():
    # Mid-step window: the finished step's writes are visible but the next checkpoint is not saved yet.
    snap = SimpleNamespace(values={"events": [{"message": "validated"}]}, next=(), interrupts=(), tasks=())
    runner = Runner(StubGraph(snap))
    runner.tasks["p"] = asyncio.create_task(asyncio.sleep(10))
    assert (await runner.status("p"))["status"] == "processing"

    runner.tasks["p"].cancel()
    del runner.tasks["p"]
    assert (await runner.status("p"))["status"] == "invalid"  # same snapshot, nothing running: terminal


async def test_running_plan_reports_its_current_node():
    snap = SimpleNamespace(values={"events": []}, next=("research",), interrupts=(), tasks=())
    runner = Runner(StubGraph(snap))
    runner.tasks["p"] = asyncio.create_task(asyncio.sleep(10))
    assert (await runner.status("p"))["status"] == "researching"
    runner.tasks["p"].cancel()
    del runner.tasks["p"]
    assert (await runner.status("p"))["status"] == "stalled"
