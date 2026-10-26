"""Tool logic tests. HTTP is replaced by httpx.MockTransport; no network."""
import json
from datetime import date

import httpx
import pytest

from app import config
from app.graph import estimate_budget, keep_unchanged
from app.schemas import load_request
from app.tools.restaurants import fetch_elements, nominatim_elements, pick_restaurants
from app.tools.scheduler import Stop, best_order, build_schedule, cluster, hhmm_to_min, leg
from app.tools.search import exa_search
from app.tools.weather import last_year, trip_weather
from conftest import trip_request


def stop(name, lat, lon, minutes=60, rank=0, opens=None, closes=None):
    return Stop(name, lat, lon, minutes, rank, hhmm_to_min(opens or ""), hhmm_to_min(closes or ""))


# ---------- scheduler ----------

def test_leg_picks_mode_by_distance():
    a = stop("a", 38.70, -9.14)
    assert leg(a, stop("b", 38.705, -9.14), "transit")[1] == "walk"      # ~0.7 km
    assert leg(a, stop("c", 38.75, -9.14), "transit")[1] == "transit"    # ~7 km
    assert leg(a, stop("d", 38.75, -9.14), "drive")[1] == "drive"
    assert leg(a, stop("e", 39.20, -9.14), "transit")[1] == "intercity"  # ~70 km


def test_cluster_separates_distant_areas():
    west = [stop(f"w{i}", 38.69, -9.21 + i * 0.002, rank=i) for i in range(3)]
    east = [stop(f"e{i}", 38.71, -9.13 + i * 0.002, rank=3 + i) for i in range(3)]
    groups = cluster(west + east, 2)
    assert sorted(sorted(s.name for s in g) for g in groups) == [["e0", "e1", "e2"], ["w0", "w1", "w2"]]


def test_order_respects_opening_hours():
    # The afternoon-only place must come last even though it is listed first.
    late = stop("market", 38.700, -9.140, opens="14:00", rank=0)
    a = stop("a", 38.701, -9.141, rank=1)
    b = stop("b", 38.702, -9.142, rank=2)
    order = best_order([late, a, b], start=9 * 60, mode="transit")
    assert order[-1].name == "market"


def test_full_days_overflow_to_neighbours_then_unscheduled():
    # Six 3-hour visits close together, two 8-hour days: two fit per day. No day has room to
    # take a stop, so each over-full day drops its own lowest-priority stop.
    stops = [stop(f"p{i}", 38.70, -9.14 + i * 0.001, minutes=180, rank=i) for i in range(6)]
    result = build_schedule(stops, days=2, start=9 * 60, end=17 * 60, mode="transit")
    days = [[s["name"] for s in d["stops"]] for d in result["days"]]
    assert sorted(map(sorted, days)) == [["p0", "p1"], ["p3", "p4"]]
    assert result["unscheduled"] == ["p2", "p5"]
    assert all(d["ends_at"] <= "17:00" for d in result["days"])


def test_over_full_day_moves_stop_to_day_with_room():
    # Day A has three 3-hour stops, day B one: A's outermost stop moves to B instead of being dropped.
    a = [stop(f"a{i}", 38.70, -9.14 + i * 0.002, minutes=180, rank=i) for i in range(3)]
    b = [stop("b0", 38.70, -9.10, minutes=60, rank=3)]
    result = build_schedule(a + b, days=2, start=9 * 60, end=17 * 60, mode="transit")
    assert result["unscheduled"] == []
    assert all(d["ends_at"] <= "17:00" for d in result["days"])


def test_more_days_than_places_leaves_free_days():
    result = build_schedule([stop("only", 38.7, -9.14)], days=3, start=540, end=1140, mode="transit")
    assert [len(d["stops"]) for d in result["days"]] == [1, 0, 0]


def test_large_day_uses_heuristic_and_visits_everything():
    stops = [stop(f"p{i}", 38.70 + (i % 3) * 0.003, -9.14 + i * 0.003, minutes=30, rank=i) for i in range(9)]
    order = best_order(stops, start=540, mode="transit")
    assert sorted(s.name for s in order) == sorted(s.name for s in stops)


# ---------- restaurants ----------

ELEMENTS = [
    {"type": "node", "lat": 38.6920, "lon": -9.2160, "tags": {"name": "Far Pizza", "amenity": "restaurant", "cuisine": "pizza"}},
    {"type": "node", "lat": 38.6917, "lon": -9.2160, "tags": {"name": "Near Tasca", "amenity": "restaurant", "cuisine": "portuguese;seafood"}},
    {"type": "way", "center": {"lat": 38.6930, "lon": -9.2160}, "tags": {"name": "Green", "amenity": "cafe", "diet:vegetarian": "only"}},
]


def test_pick_restaurants_sorts_and_filters():
    rows, matched = pick_restaurants(ELEMENTS, 38.6916, -9.2160, "", False, 5)
    assert [r["name"] for r in rows] == ["Near Tasca", "Far Pizza", "Green"] and matched
    assert rows[0]["cuisine"] == "portuguese, seafood"
    rows, matched = pick_restaurants(ELEMENTS, 38.6916, -9.2160, "seafood", False, 5)
    assert [r["name"] for r in rows] == ["Near Tasca"] and matched
    rows, _ = pick_restaurants(ELEMENTS, 38.6916, -9.2160, "", True, 5)
    assert [r["name"] for r in rows] == ["Green"]


def test_pick_restaurants_falls_back_when_cuisine_missing():
    rows, matched = pick_restaurants(ELEMENTS, 38.6916, -9.2160, "sushi", False, 2)
    assert not matched and len(rows) == 2


async def test_overpass_busy_server_falls_back_to_mirror():
    hits = []

    def handler(request):
        hits.append(request.url.host)
        if request.url.host == "overpass-api.de":
            return httpx.Response(502, text="at capacity")
        return httpx.Response(200, json={"elements": ELEMENTS})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        elements = await fetch_elements(client, 1.0, 2.0, 500, backoff=0)
    assert elements == ELEMENTS
    assert hits == ["overpass-api.de", "overpass-api.de", "maps.mail.ru"]


async def test_nominatim_fallback_maps_to_overpass_shape():
    def handler(request):
        assert request.url.params["q"] == "restaurant" and request.url.params["bounded"] == "1"
        return httpx.Response(200, json=[{
            "name": "Tasca", "type": "restaurant", "lat": "38.6950", "lon": "-9.2208",
            "extratags": {"cuisine": "regional", "opening_hours": "Mo-Su 12:00-23:00"},
            "address": {"road": "Rua A", "house_number": "5"}}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        elements = await nominatim_elements(client, 38.6916, -9.2160, 800)
    rows, _ = pick_restaurants(elements, 38.6916, -9.2160, "regional", False, 5)
    assert rows[0]["name"] == "Tasca" and rows[0]["address"] == "Rua A 5"
    assert rows[0]["opening_hours"] == "Mo-Su 12:00-23:00"


# ---------- weather ----------

def weather_transport(seen):
    def handler(request):
        seen.append(request)
        if "geocoding" in request.url.host:
            return httpx.Response(200, json={"results": [
                {"name": "Lisbon", "country": "United States", "latitude": 1, "longitude": 1},
                {"name": "Lisbon", "country": "Portugal", "latitude": 38.7, "longitude": -9.1}]})
        return httpx.Response(200, json={"daily": {
            "temperature_2m_max": [20.0, 22.0], "temperature_2m_min": [10.0, None], "precipitation_sum": [0.0, 3.2]}})
    return httpx.MockTransport(handler)


async def test_weather_uses_forecast_for_near_trips():
    seen = []
    async with httpx.AsyncClient(transport=weather_transport(seen)) as client:
        w = await trip_weather(client, "Lisbon, Portugal", date(2026, 1, 10), date(2026, 1, 11), today=date(2026, 1, 1))
    assert seen[1].url.host == "api.open-meteo.com"
    assert seen[1].url.params["latitude"] == "38.7"  # the Portuguese Lisbon, not the US one
    assert w["source"] == "forecast" and w["wet_days"] == 1 and w["avg_min_c"] == 10.0


async def test_weather_uses_last_year_for_far_trips():
    seen = []
    async with httpx.AsyncClient(transport=weather_transport(seen)) as client:
        w = await trip_weather(client, "Lisbon", date(2026, 6, 10), date(2026, 6, 11), today=date(2026, 1, 1))
    assert seen[1].url.host == "archive-api.open-meteo.com"
    assert seen[1].url.params["start_date"] == "2025-06-10"
    assert w["days"][0]["date"] == "2026-06-10" and "2025" in w["source"]


def test_last_year_handles_leap_day():
    assert last_year(date(2028, 2, 29)) == date(2027, 2, 28)


# ---------- search ----------

async def test_exa_request_and_formatting(monkeypatch):
    monkeypatch.setattr(config, "EXA_API_KEY", "k")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"results": [
            {"title": "Guide", "url": "https://x.test", "publishedDate": "2026-05-01T00:00:00Z",
             "highlights": ["one", "two"]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await exa_search(client, "lisbon food")
    body = json.loads(seen[0].content)
    assert seen[0].headers["x-api-key"] == "k"
    assert body["query"] == "lisbon food" and "highlights" in body["contents"]
    assert rows == [{"title": "Guide", "url": "https://x.test", "published": "2026-05-01", "highlights": "one ... two"}]


# ---------- budget and modify guard ----------

def test_estimate_budget_math():
    req = load_request(trip_request(days=3))  # 2 travellers, 2 nights
    draft = {
        "lodging": {"cost_per_night": 100},
        "days": [{"activities": [{"cost_per_person": 10}], "meals": [{"cost_per_person": 20}]}] * 3,
    }
    b = estimate_budget(draft, req, {"typical_costs": {"local_transport_per_person_per_day": 5}})
    assert b["lodging"] == 200 and b["activities"] == 60 and b["meals"] == 120 and b["local_transport"] == 30
    assert b["total"] == 410 and b["status"] == "within"


@pytest.mark.parametrize("mods, expected_themes, expected_lodging", [
    ([{"day": 2, "change": "x"}], ["old1", "new2", "old3"], "old"),
    # plan-wide (no day, e.g. the hotel): lodging may change, days never do
    ([{"day": None, "change": "cheaper hotel"}], ["old1", "old2", "old3"], "new"),
    ([{"day": None, "change": "cheaper hotel"}, {"day": 3, "change": "x"}], ["old1", "old2", "new3"], "new"),
])
def test_keep_unchanged(mods, expected_themes, expected_lodging):
    prev = {"lodging": "old", "days": [{"day": i, "theme": f"old{i}"} for i in (1, 2, 3)]}
    new = {"lodging": "new", "days": [{"day": i, "theme": f"new{i}"} for i in (1, 2, 3)]}
    merged = keep_unchanged(prev, new, mods)
    assert [d["theme"] for d in merged["days"]] == expected_themes
    assert merged["lodging"] == expected_lodging
