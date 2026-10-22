"""Smoke tests against the real external APIs. Skipped unless RUN_LIVE=1."""
import os
from datetime import date, timedelta

import pytest

from app import config
from app.tools import find_restaurants, optimize_schedule, weather_outlook, web_search

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="set RUN_LIVE=1")]


async def test_weather_outlook():
    start = date.today() + timedelta(days=3)
    w = await weather_outlook.ainvoke({"destination": "Lisbon, Portugal", "start_date": start.isoformat(),
                                       "end_date": (start + timedelta(days=1)).isoformat()})
    assert w["source"] == "forecast" and len(w["days"]) == 2


async def test_optimize_schedule():
    s = await optimize_schedule.ainvoke({"city": "Lisbon, Portugal", "days": 2, "places": [
        {"name": "Belém Tower"}, {"name": "Jerónimos Monastery"}, {"name": "São Jorge Castle"}]})
    assert not s["unlocated"] and sum(len(d["stops"]) for d in s["days"]) == 3


async def test_find_restaurants():
    r = await find_restaurants.ainvoke({"near": "Belém Tower", "city": "Lisbon, Portugal"})
    assert r["restaurants"]


@pytest.mark.skipif(not config.EXA_API_KEY, reason="EXA_API_KEY not set")
async def test_web_search():
    rows = await web_search.ainvoke({"query": "best viewpoints in Lisbon"})
    assert rows and rows[0]["url"].startswith("http")
