"""OpenStreetMap geocoding (Nominatim) and distance helpers shared by the planner tools."""
import asyncio
import math
import time
from dataclasses import dataclass

import httpx

from app.config import USER_AGENT

NOMINATIM = "https://nominatim.openstreetmap.org/search"
MAX_DAY_TRIP_KM = 120  # places found further from the city than this are treated as wrong matches

_cache: dict[tuple, list] = {}
_lock: asyncio.Lock | None = None
_last_call = 0.0


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    bbox: tuple[float, float, float, float] | None = None  # south, north, west, east


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=35, headers={"User-Agent": USER_AGENT})


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


async def search(client: httpx.AsyncClient, **params) -> list[dict]:
    """Nominatim's usage policy allows one request per second, so calls are serialised and cached."""
    global _lock, _last_call
    key = tuple(sorted(params.items()))
    if key not in _cache:
        _lock = _lock or asyncio.Lock()
        async with _lock:
            wait = 1.05 - (time.monotonic() - _last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                r = await client.get(NOMINATIM, params={"format": "jsonv2", "limit": 1, **params})
            finally:
                _last_call = time.monotonic()
        r.raise_for_status()
        _cache[key] = r.json()
    return _cache[key]


def _place(hit: dict, name: str) -> Place:
    s, n, w, e = (float(v) for v in hit["boundingbox"])
    return Place(name, float(hit["lat"]), float(hit["lon"]), (s, n, w, e))


async def geocode_city(client: httpx.AsyncClient, city: str) -> Place | None:
    hits = await search(client, q=city)
    return _place(hits[0], city) if hits else None


async def geocode_place(client: httpx.AsyncClient, name: str, city: Place) -> Place | None:
    """Look inside the city's bounding box first, then anywhere within day-trip distance."""
    s, n, w, e = city.bbox
    hits = await search(client, q=name, viewbox=f"{w},{n},{e},{s}", bounded=1)
    if not hits:
        hits = await search(client, q=name)
    if not hits:
        return None
    place = _place(hits[0], name)
    if haversine_km(city.lat, city.lon, place.lat, place.lon) > MAX_DAY_TRIP_KM:
        return None
    return place
