"""Orchestrator: a LangGraph StateGraph that routes work between the agents and pauses for review.

    validate -> research -> plan -> review (interrupt)
    plan   --over budget, once--> plan
    review --approve--> finalize
    review --reject---> research or plan (picked by the reviewer or an LLM router)
    review --modify---> modify -> review

State is checkpointed after every node, so a plan waiting for review survives restarts.
"""
import operator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from app import agents
from app.schemas import ReviewRequest, TravelRequest, load_request


class PlanState(TypedDict, total=False):
    request: dict
    research: dict
    draft: dict                 # latest itinerary plus its computed budget
    revision: int               # drafts produced so far
    pending: dict | None        # instructions for the next agent run; kind = reject | modify | budget
    budget_retry_done: bool
    reviews: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]
    final_plan: dict
    error: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event(message: str) -> list[dict]:
    return [{"at": _now(), "message": message}]


def _tools(used: dict) -> str:
    return ", ".join(f"{name} x{n}" for name, n in used.items()) or "no tool calls"


def estimate_budget(draft: dict, req: TravelRequest, research: dict) -> dict:
    """Totals are computed in code from the agent's per-item estimates, never by the LLM."""
    activities = sum(a["cost_per_person"] for d in draft["days"] for a in d["activities"])
    meals = sum(m["cost_per_person"] for d in draft["days"] for m in d["meals"])
    transport = research["typical_costs"]["local_transport_per_person_per_day"] * req.days
    parts = {
        "lodging": draft["lodging"]["cost_per_night"] * req.nights,
        "activities": activities * req.travelers,
        "meals": meals * req.travelers,
        "local_transport": transport * req.travelers,
    }
    total = sum(parts.values())
    return {
        "currency": req.currency,
        **{k: round(v, 2) for k, v in parts.items()},
        "total": round(total, 2),
        "budget_min": req.budget_min,
        "budget_max": req.budget_max,
        "status": "over" if total > req.budget_max else "under" if total < req.budget_min else "within",
    }


def keep_unchanged(previous: dict, new: dict, modifications: list[dict]) -> dict:
    """Make a modify touch only what the reviewer named.
    If every change names a day, all other days, the lodging and the texts stay exactly as before.
    A plan-wide change (no day, e.g. the hotel) accepts the new draft as returned."""
    if any(m.get("day") is None for m in modifications):
        return new
    changed = {m["day"] for m in modifications}
    new_days = {d["day"]: d for d in new["days"]}
    merged = dict(previous)
    merged["days"] = [new_days.get(d["day"], d) if d["day"] in changed else d for d in previous["days"]]
    return merged


async def validate(state: PlanState) -> dict:
    try:
        req = TravelRequest.model_validate(state["request"])
    except ValidationError as e:
        reason = "; ".join(err["msg"] for err in e.errors())
        return {"error": f"invalid request: {reason}", "events": _event(f"Request rejected: {reason}")}
    return {
        "request": req.model_dump(mode="json"),
        "revision": 0,
        "events": _event(f"Request validated: {req.days} days in {req.destination} for {req.travelers}"),
    }


async def research(state: PlanState) -> dict:
    req = load_request(state["request"])
    feedback = (state.get("pending") or {}).get("feedback", "")
    report, used = await agents.run_research(req, feedback, state.get("research"))
    return {
        "research": report.model_dump(mode="json"),
        "events": _event(f"Research agent: {len(report.attractions)} attractions, "
                         f"{len(report.sources)} sources ({_tools(used)})"),
    }


async def plan(state: PlanState) -> dict:
    req = load_request(state["request"])
    pending = state.get("pending") or {}
    itinerary, used = await agents.run_planner(
        req, state["research"], state.get("draft") if pending else None, pending.get("feedback", "")
    )
    draft = itinerary.model_dump(mode="json")
    budget = draft["budget"] = estimate_budget(draft, req, state["research"])
    revision = state.get("revision", 0) + 1
    update = {
        "draft": draft,
        "revision": revision,
        "pending": None,
        "events": _event(f"Planner agent: draft {revision}, estimated {budget['total']} {req.currency} "
                         f"({budget['status']} budget; {_tools(used)})"),
    }
    if budget["status"] == "over" and not state.get("budget_retry_done"):
        over = round(budget["total"] - req.budget_max, 2)
        update["pending"] = {
            "kind": "budget",
            "feedback": f"The estimated total {budget['total']} {req.currency} is {over} over the maximum "
                        f"budget of {req.budget_max}. Cut costs (cheaper lodging, free sights, cheaper meals) "
                        "to fit the budget while keeping the plan's spirit.",
        }
        update["budget_retry_done"] = True
        update["events"] += _event("Over budget: sent back to the planner once before review")
    return update


def after_plan(state: PlanState) -> Literal["plan", "review"]:
    return "plan" if (state.get("pending") or {}).get("kind") == "budget" else "review"


async def review(state: PlanState) -> Command[Literal["finalize", "research", "plan", "modify"]]:
    # Execution stops here until the API resumes the thread with Command(resume=<review body>).
    decision = ReviewRequest.model_validate(interrupt({
        "revision": state["revision"],
        "message": "Approve, reject with feedback, or modify specific days.",
    }))
    record = {"revision": state["revision"], "at": _now(), **decision.model_dump(mode="json")}

    if decision.action == "approve":
        return Command(goto="finalize", update={
            "reviews": [record], "events": _event(f"Draft {state['revision']} approved")})

    if decision.action == "modify":
        return Command(goto="modify", update={
            "reviews": [record],
            "pending": {"kind": "modify", "modifications": record["modifications"]},
            "events": _event(f"Draft {state['revision']}: {len(decision.modifications)} modification(s) requested"),
        })

    if decision.revise:
        target, reason = decision.revise, "chosen by the reviewer"
    else:
        target, reason = await agents.route_feedback(decision.feedback)
    record |= {"routed_to": target, "routing_reason": reason}
    return Command(goto="research" if target == "research" else "plan", update={
        "reviews": [record],
        "pending": {"kind": "reject", "feedback": decision.feedback},
        "budget_retry_done": False,
        "events": _event(f"Draft {state['revision']} rejected; sent to the {target} agent ({reason})"),
    })


async def modify(state: PlanState) -> dict:
    req = load_request(state["request"])
    mods = state["pending"]["modifications"]
    itinerary, used = await agents.run_modifier(req, state["research"], state["draft"], mods)
    draft = keep_unchanged(state["draft"], itinerary.model_dump(mode="json"), mods)
    draft["budget"] = estimate_budget(draft, req, state["research"])
    revision = state["revision"] + 1
    return {
        "draft": draft,
        "revision": revision,
        "pending": None,
        "events": _event(f"Planner agent applied {len(mods)} modification(s): draft {revision}, estimated "
                         f"{draft['budget']['total']} {req.currency} ({_tools(used)})"),
    }


def finalize(state: PlanState, config: RunnableConfig) -> dict:
    """Deterministic: the final plan is exactly the approved draft plus context, no LLM rewrite."""
    r = state["research"]
    final = {
        "plan_id": config["configurable"]["thread_id"],
        "approved_at": _now(),
        "approved_revision": state["revision"],
        "request": state["request"],
        "itinerary": state["draft"],
        "destination_brief": {k: r[k] for k in (
            "summary", "weather", "neighbourhoods_to_stay", "local_tips", "safety", "seasonal_notes", "sources")},
        "review_history": state.get("reviews", []),
    }
    return {"final_plan": final, "events": _event("Final plan produced")}


@asynccontextmanager
async def open_checkpointer(path: str):
    """SQLite checkpointer. The agents run as subgraphs and checkpoint their Pydantic outputs,
    which LangGraph only deserialises for allow-listed types."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("app.schemas", "ResearchReport"), ("app.schemas", "Itinerary")])
    async with aiosqlite.connect(path) as conn:
        yield AsyncSqliteSaver(conn, serde=serde)


def build_graph(checkpointer=None):
    g = StateGraph(PlanState)
    g.add_node("validate", validate)
    g.add_node("research", research)
    g.add_node("plan", plan)
    g.add_node("review", review)
    g.add_node("modify", modify)
    g.add_node("finalize", finalize)
    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", lambda s: END if s.get("error") else "research", ["research", END])
    g.add_edge("research", "plan")
    g.add_conditional_edges("plan", after_plan, ["plan", "review"])
    g.add_edge("modify", "review")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
