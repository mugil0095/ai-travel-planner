"""Fake agents so graph and API tests run without LLM or search keys."""
from datetime import date, timedelta

import pytest

from app import agents
from app.schemas import Itinerary, ResearchReport


def trip_request(days: int = 3, **overrides) -> dict:
    start = date.today() + timedelta(days=30)
    return {
        "destination": "Lisbon, Portugal",
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=days - 1)).isoformat(),
        "budget_min": 300,
        "budget_max": 2000,
        "currency": "EUR",
        "travelers": 2,
        "interests": ["food", "history"],
        **overrides,
    }


def research_report() -> ResearchReport:
    return ResearchReport.model_validate({
        "summary": "Hilly, sunny capital.",
        "attractions": [{"name": "Belém Tower", "area": "Belém", "why": "history", "duration_hours": 1,
                         "cost_per_person": 10}],
        "neighbourhoods_to_stay": ["Baixa"],
        "local_tips": ["Wear good shoes"],
        "safety": ["Watch for pickpockets on tram 28"],
        "weather": "Mild, 20C",
        "seasonal_notes": [],
        "typical_costs": {"hotel_per_night": 120, "meal_per_person": 20, "local_transport_per_person_per_day": 5},
        "sources": ["https://example.com"],
    })


def itinerary(req, theme: str = "Explore", meal_cost: float = 20, hotel: float = 100) -> Itinerary:
    start = req.start_date
    return Itinerary.model_validate({
        "title": "Lisbon trip",
        "summary": "A relaxed trip",
        "lodging": {"name": "Hotel Baixa", "area": "Baixa", "cost_per_night": hotel, "why": "central"},
        "days": [{
            "day": i + 1,
            "date": (start + timedelta(days=i)).isoformat(),
            "theme": f"{theme} {i + 1}",
            "activities": [{"start": "09:00", "end": "10:00", "name": "Belém Tower", "description": "Tower",
                            "area": "Belém", "cost_per_person": 10}],
            "meals": [{"meal": "lunch", "name": "O Navegador", "cuisine": "portuguese", "near": "Belém",
                       "cost_per_person": meal_cost}],
        } for i in range(req.days)],
        "tips": ["Bring a jacket"],
    })


class FakeAgents:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.plan_costs: list[float] = []   # meal cost per planner call, consumed in order
        self.fail_planner = False

    async def run_research(self, req, feedback="", previous=None):
        self.calls.append(("research", feedback))
        return research_report(), {"web_search": 5, "weather_outlook": 1}

    async def run_planner(self, req, research, previous=None, feedback=""):
        self.calls.append(("plan", feedback))
        if self.fail_planner:
            raise RuntimeError("LLM quota exceeded")
        cost = self.plan_costs.pop(0) if self.plan_costs else 20
        return itinerary(req, meal_cost=cost), {"optimize_schedule": 1, "find_restaurants": 3}

    async def run_modifier(self, req, research, draft, modifications):
        self.calls.append(("modify", modifications))
        # Changes every day and the hotel, so tests can check that untouched parts are restored.
        return itinerary(req, theme="Changed", hotel=999), {}

    async def route_feedback(self, feedback):
        self.calls.append(("route", feedback))
        return ("research", "needs new places") if "museum" in feedback else ("itinerary", "pacing")

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def fake(monkeypatch) -> FakeAgents:
    f = FakeAgents()
    for name in ("run_research", "run_planner", "run_modifier", "route_feedback"):
        monkeypatch.setattr(agents, name, getattr(f, name))
    return f
