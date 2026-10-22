"""The real agents (middleware, tools, structured output) inside the real graph and runner,
driven by a scripted fake chat model. Catches wiring bugs the faked-agent tests cannot."""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import agents, config
from app.graph import build_graph, open_checkpointer
from app.runner import Runner
from app.schemas import load_request
from conftest import itinerary, research_report, trip_request


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


async def test_real_agents_run_inside_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXA_API_KEY", "")  # web_search answers with an error dict, no network
    req = load_request(trip_request())
    model = ScriptedModel(messages=iter([
        tool_call("web_search", {"query": "Lisbon food"}, "1"),
        tool_call("ResearchReport", research_report().model_dump(mode="json"), "2"),
        tool_call("Itinerary", itinerary(req).model_dump(mode="json"), "3"),
    ]))
    monkeypatch.setattr(agents, "models", lambda: [model])
    agents.research_agent.cache_clear()
    agents.planner_agent.cache_clear()
    try:
        async with open_checkpointer(str(tmp_path / "cp.sqlite")) as saver:
            runner = Runner(build_graph(saver))
            runner.start("p", {"request": trip_request()})
            await runner.tasks["p"]
            status = await runner.status("p")
    finally:
        agents.research_agent.cache_clear()
        agents.planner_agent.cache_clear()

    assert status["status"] == "awaiting_review", status["error"]
    assert "web_search x1" in status["events"][1]["message"]
    assert len(status["draft"]["days"]) == 3 and status["draft"]["budget"]["status"] == "within"
