"""Orchestrator tests: routing, HITL interrupts and checkpoint persistence (fake agents)."""
from datetime import date, timedelta

import pytest
from langgraph.types import Command

from app.graph import build_graph, open_checkpointer
from conftest import trip_request

CFG = {"configurable": {"thread_id": "t1"}}


@pytest.fixture
async def graph(tmp_path):
    async with open_checkpointer(str(tmp_path / "cp.sqlite")) as saver:
        yield build_graph(saver)


async def start(graph, **req):
    await graph.ainvoke({"request": trip_request(**req)}, CFG)
    return await graph.aget_state(CFG)


async def resume(graph, review: dict):
    await graph.ainvoke(Command(resume=review), CFG)
    return await graph.aget_state(CFG)


async def test_pauses_for_review_then_approve_finalizes(graph, fake):
    snap = await start(graph)
    assert snap.next == ("review",) and snap.interrupts
    assert fake.names() == ["research", "plan"]
    assert snap.values["draft"]["budget"]["status"] == "within"

    snap = await resume(graph, {"action": "approve"})
    final = snap.values["final_plan"]
    assert not snap.next
    assert final["plan_id"] == "t1" and final["approved_revision"] == 1
    assert final["itinerary"] == snap.values["draft"]  # exactly what was approved
    assert final["review_history"][0]["action"] == "approve"


async def test_invalid_request_stops_before_agents(graph, fake):
    past = (date.today() - timedelta(days=5)).isoformat()
    snap = await start(graph, start_date=past, end_date=past)
    assert not snap.next and "past" in snap.values["error"]
    assert fake.calls == []


async def test_reject_routed_by_llm_to_planner(graph, fake):
    await start(graph)
    snap = await resume(graph, {"action": "reject", "feedback": "Too rushed, fewer stops per day"})
    assert fake.names() == ["research", "plan", "route", "plan"]
    assert fake.calls[-1] == ("plan", "Too rushed, fewer stops per day")
    assert snap.interrupts and snap.values["revision"] == 2
    assert snap.values["reviews"][0]["routed_to"] == "itinerary"


async def test_reject_routed_by_llm_to_research(graph, fake):
    await start(graph)
    await resume(graph, {"action": "reject", "feedback": "Add a modern art museum"})
    assert fake.names() == ["research", "plan", "route", "research", "plan"]
    # both agents see the feedback
    assert fake.calls[3] == ("research", "Add a modern art museum")
    assert fake.calls[4] == ("plan", "Add a modern art museum")


async def test_reject_with_explicit_target_skips_router(graph, fake):
    await start(graph)
    await resume(graph, {"action": "reject", "feedback": "Need more options", "revise": "research"})
    assert "route" not in fake.names()
    assert fake.names()[-2:] == ["research", "plan"]


async def test_modify_changes_only_named_days(graph, fake):
    before = (await start(graph)).values["draft"]
    snap = await resume(graph, {"action": "modify", "modifications": [{"day": 2, "change": "Beach afternoon"}]})
    after = snap.values["draft"]
    assert fake.names()[-1] == "modify"
    assert after["days"][1]["theme"] == "Changed 2"
    assert after["days"][0] == before["days"][0] and after["days"][2] == before["days"][2]
    assert after["lodging"] == before["lodging"]  # the fake changed the hotel; it was restored
    assert snap.interrupts and snap.values["revision"] == 2


async def test_plan_wide_modify_accepts_new_draft(graph, fake):
    await start(graph)
    snap = await resume(graph, {"action": "modify", "modifications": [{"change": "Cheaper hotel"}]})
    assert snap.values["draft"]["lodging"]["cost_per_night"] == 999


async def test_over_budget_draft_goes_back_to_planner_once(graph, fake):
    fake.plan_costs = [500, 400]  # both drafts over budget
    snap = await start(graph)
    assert fake.names() == ["research", "plan", "plan"]
    assert "over the maximum budget" in fake.calls[2][1]
    assert snap.interrupts and snap.values["draft"]["budget"]["status"] == "over"


async def test_review_survives_restart(tmp_path, fake):
    db = str(tmp_path / "cp.sqlite")
    async with open_checkpointer(db) as saver:
        await build_graph(saver).ainvoke({"request": trip_request()}, CFG)
    # new connection and new graph object = a restarted server
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        assert (await graph.aget_state(CFG)).interrupts
        await graph.ainvoke(Command(resume={"action": "approve"}), CFG)
        assert (await graph.aget_state(CFG)).values["final_plan"]
    assert fake.names() == ["research", "plan"]  # nothing re-ran after the restart


async def test_failure_is_checkpointed_and_retry_resumes(graph, fake):
    fake.fail_planner = True
    with pytest.raises(RuntimeError):
        await graph.ainvoke({"request": trip_request()}, CFG)
    snap = await graph.aget_state(CFG)
    assert snap.next == ("plan",) and any(t.error for t in snap.tasks)

    fake.fail_planner = False
    await graph.ainvoke(None, CFG)
    snap = await graph.aget_state(CFG)
    assert snap.interrupts
    assert fake.names() == ["research", "plan", "plan"]  # research was not repeated
