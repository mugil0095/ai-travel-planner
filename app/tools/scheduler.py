"""Planner tool 1: a scheduling optimiser.

Geocodes the candidate places on OpenStreetMap, groups them into days by proximity
(k-means, then moves stops out of over-full days), and orders each day to respect
opening hours and minimise travel (exhaustive search up to 6 stops, nearest neighbour +
2-opt above). Travel time = straight-line distance x 1.3 at a speed per mode: an
estimate, not turn-by-turn routing.
"""
import itertools
import math
from dataclasses import dataclass
from typing import Literal

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools.geo import Place, geocode_city, geocode_place, haversine_km, new_client

DETOUR = 1.3     # street distance / straight-line distance
WALK_MAX_KM = 1.5
EXACT_MAX = 6    # 6! = 720 orderings per day


@dataclass(frozen=True)
class Stop:
    name: str
    lat: float
    lon: float
    minutes: int
    rank: int                  # position in the input list = priority
    opens: int | None = None   # minutes after midnight
    closes: int | None = None


def hhmm_to_min(s: str) -> int | None:
    try:
        h, m = s.strip().split(":")
        return int(h) * 60 + int(m)
    except ValueError:
        return None


def min_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def leg(a: Stop | Place, b: Stop, mode: str) -> tuple[int, str, float]:
    """Minutes, mode used and street km between two points."""
    km = haversine_km(a.lat, a.lon, b.lat, b.lon) * DETOUR
    if km <= WALK_MAX_KM:
        return round(km / 4.5 * 60), "walk", km
    if km > 40:
        return round(km / 70 * 60) + 20, "intercity", km
    if mode == "drive":
        return round(km / 25 * 60) + 5, "drive", km
    speed = 18 if km <= 10 else 35  # city bus/metro vs suburban train
    return round(km / speed * 60) + 8, "transit", km


def simulate(order, start: int, mode: str, home: Place | None = None):
    """Walk through a day in order. Returns (late visits, end minute, travel minutes, timeline)."""
    t, prev, travel, late, rows = start, home, 0, 0, []
    for s in order:
        mins, how, km = leg(prev, s, mode) if prev else (0, "start", 0.0)
        t += mins
        travel += mins
        wait = max(0, s.opens - t) if s.opens is not None else 0
        t += wait
        is_late = s.closes is not None and t + s.minutes > s.closes
        late += is_late
        row = {
            "name": s.name,
            "start": min_to_hhmm(t),
            "end": min_to_hhmm(t + s.minutes),
            "travel_minutes_from_previous": mins,
            "travel_mode": how,
            "travel_km": round(km, 1),
        }
        if wait:
            row["wait_minutes"] = wait
        if is_late:
            row["warning"] = "closes before the visit would end"
        rows.append(row)
        t += s.minutes
        prev = s
    return late, t, travel, rows


def best_order(stops: list[Stop], start: int, mode: str, home: Place | None = None) -> list[Stop]:
    """Fewest late visits first, then earliest finish, then least travel."""
    def cost(order):
        return simulate(order, start, mode, home)[:3]

    if len(stops) <= EXACT_MAX:
        return list(min(itertools.permutations(stops), key=cost))

    def km(a, b):
        return haversine_km(a.lat, a.lon, b.lat, b.lon)

    first = min(stops, key=lambda s: km(home, s)) if home else min(stops, key=lambda s: s.rank)
    order, rest = [first], [s for s in stops if s is not first]
    while rest:
        nxt = min(rest, key=lambda s: km(order[-1], s))
        order.append(nxt)
        rest.remove(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if cost(cand) < cost(order):
                    order, improved = cand, True
    return order


def cluster(stops: list[Stop], k: int) -> list[list[Stop]]:
    """k-means on projected coordinates, seeded with the top-priority stop then farthest points."""
    k = min(k, len(stops))
    if k == 0:
        return []
    lat0 = sum(s.lat for s in stops) / len(stops)
    pts = [(s.lon * 111.32 * math.cos(math.radians(lat0)), s.lat * 110.57) for s in stops]
    centers = [pts[0]]
    while len(centers) < k:
        centers.append(max(pts, key=lambda p: min(math.dist(p, c) for c in centers)))
    for _ in range(50):
        groups = [[] for _ in centers]
        for i, p in enumerate(pts):
            groups[min(range(k), key=lambda c: math.dist(p, centers[c]))].append(i)
        new = [
            (sum(pts[i][0] for i in g) / len(g), sum(pts[i][1] for i in g) / len(g)) if g else centers[c]
            for c, g in enumerate(groups)
        ]
        if new == centers:
            break
        centers = new
    return [[stops[i] for i in g] for g in groups]


def balance(groups: list[list[Stop]], start: int, end: int, mode: str, home: Place | None) -> list[Stop]:
    """Move stops out of over-full days into the nearest day with room.
    When no day has room, drop the over-full day's lowest-priority stop. Returns dropped stops."""
    ends: dict[tuple, int] = {}

    def day_end(g):
        key = tuple(sorted(s.rank for s in g))
        if key not in ends:
            ends[key] = simulate(best_order(g, start, mode, home), start, mode, home)[1]
        return ends[key]

    def gap_km(s, g):
        return min((haversine_km(s.lat, s.lon, o.lat, o.lon) for o in g), default=0.0)

    dropped = []
    for _ in range(2 * sum(map(len, groups))):
        over = next((g for g in groups if len(g) > 1 and day_end(g) > end), None)
        if over is None:
            break
        clat = sum(s.lat for s in over) / len(over)
        clon = sum(s.lon for s in over) / len(over)
        s = max(over, key=lambda s: haversine_km(s.lat, s.lon, clat, clon))
        others = sorted((g for g in groups if g is not over), key=lambda g: gap_km(s, g))
        target = next((g for g in others if day_end(g + [s]) <= end), None)
        if target is not None:
            over.remove(s)
            target.append(s)
        else:
            worst = max(over, key=lambda s: s.rank)
            over.remove(worst)
            dropped.append(worst)
    return dropped


def build_schedule(stops: list[Stop], days: int, start: int, end: int, mode: str, home: Place | None = None) -> dict:
    groups = cluster(stops, days)
    dropped = balance(groups, start, end, mode, home)
    groups += [[] for _ in range(days - len(groups))]
    groups.sort(key=lambda g: min((s.rank for s in g), default=math.inf))  # top priorities first
    out = []
    for i, g in enumerate(groups, 1):
        _, finish, travel, rows = simulate(best_order(g, start, mode, home), start, mode, home)
        out.append({"day": i, "stops": rows, "travel_minutes": travel, "ends_at": min_to_hhmm(finish) if g else None})
    return {"days": out, "unscheduled": [s.name for s in sorted(dropped, key=lambda s: s.rank)]}


class PlaceIn(BaseModel):
    name: str
    duration_hours: float = Field(default=1.5, gt=0, le=10)
    opens: str = Field(default="", description="HH:MM (24h), empty if unknown")
    closes: str = Field(default="", description="HH:MM (24h), empty if unknown")


@tool
async def optimize_schedule(
    city: str,
    places: list[PlaceIn],
    days: int,
    lodging_area: str = "",
    day_start: str = "09:00",
    day_end: str = "19:00",
    mode: Literal["transit", "drive"] = "transit",
) -> dict:
    """Turn candidate places into a timed day-by-day schedule with minimal travel.
    Places are located on OpenStreetMap, grouped into `days` by proximity and ordered to
    respect opening hours and cut travel time; legs show minutes, km and walk/transit/drive.
    List places most important first and include more than fit: low-priority ones that do
    not fit come back as `unscheduled`. Call once with all candidates."""
    try:
        async with new_client() as client:
            center = await geocode_city(client, city)
            if center is None:
                return {"error": f"could not locate city '{city}'"}
            home = await geocode_place(client, lodging_area, center) if lodging_area else None
            stops, unlocated = [], []
            for rank, p in enumerate(places):
                loc = await geocode_place(client, p.name, center)
                if loc is None:
                    unlocated.append(p.name)
                    continue
                stops.append(Stop(p.name, loc.lat, loc.lon, round(p.duration_hours * 60), rank,
                                  hhmm_to_min(p.opens), hhmm_to_min(p.closes)))
    except httpx.HTTPError as e:
        return {"error": f"geocoding failed: {e}"}
    start = hhmm_to_min(day_start) or 9 * 60
    end = hhmm_to_min(day_end) or 19 * 60
    return {
        **build_schedule(stops, max(1, days), start, end, mode, home),
        "unlocated": unlocated,
        "first_leg_from_lodging": home is not None,
        "travel_model": "straight line x1.3; walk <=1.5 km at 4.5 km/h, transit 18 km/h (35 km/h over "
                        "10 km) + 8 min, drive 25 km/h + 5 min, >40 km intercity 70 km/h + 20 min",
    }
