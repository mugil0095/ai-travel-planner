"""The two specialist agents (research, itinerary planner) and the feedback router.

Each agent is a LangChain tool-calling agent (a small LangGraph ReAct loop) that ends by
returning a Pydantic object, so the orchestrator never parses free text.
"""
import json
from collections import Counter
from functools import lru_cache
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app import config
from app.schemas import Itinerary, ResearchReport, TravelRequest
from app.tools import find_restaurants, optimize_schedule, weather_outlook, web_search

RESEARCH_TOOLS = [web_search, weather_outlook]
PLANNER_TOOLS = [optimize_schedule, find_restaurants]
TOOL_NAMES = {t.name for t in RESEARCH_TOOLS + PLANNER_TOOLS}
# Hard caps per agent run. Over the cap a call is blocked and the model must answer with what it has
# (in live testing a model otherwise kept "verifying" every attraction, one search at a time).
TOOL_CALL_LIMITS = {"web_search": 8, "weather_outlook": 1, "optimize_schedule": 2, "find_restaurants": 10}
RUN_LIMITS = {"recursion_limit": 40}  # backstop for the whole reasoning/tool loop

RESEARCH_PROMPT = """You are a destination research specialist preparing facts for an itinerary planner.

Use web_search for focused queries, one topic each. Issue all of them, plus weather_outlook, together
in your first turn as parallel tool calls; add at most one follow-up round if something is missing.
Topics:
- top attractions matching the travellers' interests, with opening hours and entry prices
- the best neighbourhoods to stay in for these travellers
- practical local tips: getting around, etiquette, tipping, scams
- current safety advice
- events, festivals, public holidays, closures or crowds on the exact travel dates
- typical prices: mid-range hotel per night, a meal, local transport per day
Call weather_outlook once with the trip dates.
If the message includes reviewer feedback, focus the searches on what the feedback needs.

Then return the research report:
- 8-15 attractions, most relevant to the interests first. Use the official name of each place as
  it would appear on a map. Give realistic visit durations and per-person entry prices converted
  to the trip currency (0 if free).
- Prices in the trip currency; if you converted, use a sensible current rate.
- sources: only URLs you actually used. Do not invent facts the searches did not support."""

PLANNER_PROMPT = """You are an itinerary planner. Build a realistic day-by-day plan from the research.

1. Pick attractions that fit the interests and pace. Call optimize_schedule ONCE with the city,
   the candidate places in priority order (include a few more than fit), the number of days, the
   lodging area, and day_start/day_end suited to the travellers' pace.
2. Call find_restaurants once per day, near that day's main area, and pick lunch and dinner from
   the results. Make these calls together in one turn, not one at a time.
3. Return the itinerary:
   - exactly one entry per trip day, with the correct date
   - follow the optimiser's grouping, order, times and travel minutes; skip 'unscheduled' places
   - use real restaurant names from find_restaurants when available
   - costs in the trip currency: activities and meals per person, lodging per night for the
     whole group
   - estimated total = lodging x nights + (activities + meals + local transport) per person
     x travellers. Keep it at or under budget_max and, where sensible, above budget_min.
   - practical tips: weather, packing, transport, safety from the research"""

ROUTER_PROMPT = """A traveller rejected a draft trip plan with this feedback:

{feedback}

Decide which specialist must redo its work:
- "research": the feedback needs new information about the destination (different kinds of
  places or interests, areas not covered, events, more or other options to choose from)
- "itinerary": the feedback is about arrangement of what is already known (pacing, order,
  timing, budget, hotel, restaurants, swapping or dropping items)"""


class Route(BaseModel):
    target: Literal["research", "itinerary"]
    reason: str


@lru_cache
def models() -> list[ChatGoogleGenerativeAI]:
    """The primary model, then fallbacks used when it is overloaded (503) or rate limited (429).
    Each model fails over to the next after 2 retries; the last one backs off and retries 6 times."""
    names = list(dict.fromkeys([config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS]))
    return [
        ChatGoogleGenerativeAI(model=name, google_api_key=config.GOOGLE_API_KEY, temperature=0.3,
                               max_retries=6 if i == len(names) - 1 else 2, timeout=180)
        for i, name in enumerate(names)
    ]


def _agent(tools, prompt: str, schema, name: str):
    primary, *fallbacks = models()
    limits = [ToolCallLimitMiddleware(tool_name=t.name, run_limit=TOOL_CALL_LIMITS[t.name]) for t in tools]
    return create_agent(
        primary, tools, system_prompt=prompt, response_format=ToolStrategy(schema), name=name,
        middleware=[*limits, *([ModelFallbackMiddleware(*fallbacks)] if fallbacks else [])],
    )


@lru_cache
def research_agent():
    return _agent(RESEARCH_TOOLS, RESEARCH_PROMPT, ResearchReport, "research_agent")


@lru_cache
def planner_agent():
    return _agent(PLANNER_TOOLS, PLANNER_PROMPT, Itinerary, "planner_agent")


def _trip(req: TravelRequest) -> str:
    return json.dumps(req.model_dump(mode="json") | {"days": req.days, "nights": req.nights}, indent=1)


async def _run(agent, message: str):
    """Returns (structured output, {tool name: calls})."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message}]}, RUN_LIMITS)
    used = Counter(m.name for m in result["messages"] if m.type == "tool" and m.name in TOOL_NAMES)
    return result["structured_response"], dict(used)


async def run_research(req: TravelRequest, feedback: str = "", previous: dict | None = None):
    msg = f"Trip request:\n{_trip(req)}"
    if feedback:
        msg += (f"\n\nThe traveller rejected the previous plan. Research what this feedback needs:\n{feedback}"
                f"\n\nPrevious research (keep what is still useful):\n{json.dumps(previous)}")
    return await _run(research_agent(), msg)


async def run_planner(req: TravelRequest, research: dict, previous: dict | None = None, feedback: str = ""):
    msg = f"Trip request:\n{_trip(req)}\n\nResearch:\n{json.dumps(research)}"
    if previous:
        msg += (f"\n\nPrevious draft:\n{json.dumps(previous)}\n\nRevise it to address every point of this "
                f"feedback, keeping what was not criticised:\n{feedback}")
    return await _run(planner_agent(), msg)


async def run_modifier(req: TravelRequest, research: dict, draft: dict, modifications: list[dict]):
    changes = "\n".join(f"- {'Day ' + str(m['day']) if m.get('day') else 'Whole plan'}: {m['change']}"
                        for m in modifications)
    msg = (f"Trip request:\n{_trip(req)}\n\nResearch:\n{json.dumps(research)}\n\nCurrent draft:\n{json.dumps(draft)}"
           f"\n\nApply ONLY these changes and return the full itinerary with everything else exactly as it is."
           f" Use the tools only if a change needs it (e.g. a restaurant near a new place):\n{changes}")
    return await _run(planner_agent(), msg)


async def route_feedback(feedback: str) -> tuple[str, str]:
    """Pick the agent that should handle a rejection. Falls back to the planner if the call fails."""
    primary, *fallbacks = [m.with_structured_output(Route) for m in models()]
    try:
        route = await primary.with_fallbacks(fallbacks).ainvoke(ROUTER_PROMPT.format(feedback=feedback))
        return route.target, route.reason
    except Exception as e:  # a routing hiccup should not fail the whole review
        return "itinerary", f"router unavailable ({type(e).__name__}); defaulted to the planner"
