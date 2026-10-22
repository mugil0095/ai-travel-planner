"""Planner tool 2: real restaurants near a place, from OpenStreetMap (free, no key).

Overpass gives the most complete radius search; when its public servers are overloaded the
tool falls back to Nominatim's restaurant search in a box around the place.
"""
import asyncio
import math

import httpx
from langchain_core.tools import tool

from app.tools import geo
from app.tools.geo import geocode_city, geocode_place, haversine_km, new_client

# Public Overpass servers are often at capacity: retry once with backoff, then use a mirror.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
BUSY = {429, 502, 503, 504}
_cache: dict[tuple, list[dict]] = {}


def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    return (
        "[out:json][timeout:25];"
        f'nwr["amenity"~"^(restaurant|cafe)$"]["name"](around:{radius_m},{lat},{lon});'
        "out center tags 100;"
    )


async def fetch_elements(client: httpx.AsyncClient, lat: float, lon: float, radius_m: int,
                         backoff: float = 3.0) -> list[dict]:
    key = (round(lat, 4), round(lon, 4), radius_m)
    if key in _cache:
        return _cache[key]
    error = None
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                r = await client.post(url, data={"data": overpass_query(lat, lon, radius_m)})
                if r.status_code in BUSY and attempt == 0:
                    await asyncio.sleep(backoff)
                    continue
                r.raise_for_status()
                _cache[key] = r.json()["elements"]
                return _cache[key]
            except httpx.HTTPError as e:
                error = e
                break
    raise error


async def nominatim_elements(client: httpx.AsyncClient, lat: float, lon: float, radius_m: int) -> list[dict]:
    """Same shape as Overpass elements, so pick_restaurants handles both."""
    dlat = radius_m / 111_000
    dlon = dlat / math.cos(math.radians(lat))
    hits = await geo.search(client, q="restaurant", viewbox=f"{lon - dlon},{lat + dlat},{lon + dlon},{lat - dlat}",
                            bounded=1, limit=40, extratags=1, addressdetails=1)
    return [{
        "lat": float(h["lat"]),
        "lon": float(h["lon"]),
        "tags": {
            **(h.get("extratags") or {}),
            "name": h["name"],
            "amenity": h.get("type", ""),
            "addr:street": (h.get("address") or {}).get("road", ""),
            "addr:housenumber": (h.get("address") or {}).get("house_number", ""),
        },
    } for h in hits if h.get("name")]


def pick_restaurants(elements: list[dict], lat: float, lon: float, cuisine: str,
                     vegetarian: bool, limit: int) -> tuple[list[dict], bool]:
    """Nearest matches first. Returns (rows, cuisine_matched); falls back to any cuisine."""
    rows = []
    for el in elements:
        tags = el.get("tags", {})
        if vegetarian and not ({tags.get("diet:vegetarian"), tags.get("diet:vegan")} & {"yes", "only"}):
            continue
        plat = el.get("lat", el.get("center", {}).get("lat"))
        plon = el.get("lon", el.get("center", {}).get("lon"))
        if plat is None or plon is None:
            continue
        rows.append({
            "name": tags["name"],
            "type": tags.get("amenity", ""),
            "cuisine": tags.get("cuisine", "").replace(";", ", "),
            "distance_m": round(haversine_km(lat, lon, plat, plon) * 1000),
            "opening_hours": tags.get("opening_hours", ""),
            "address": " ".join(filter(None, [tags.get("addr:street"), tags.get("addr:housenumber")])),
            "website": tags.get("website") or tags.get("contact:website", ""),
        })
    rows.sort(key=lambda r: r["distance_m"])
    if cuisine:
        matched = [r for r in rows if cuisine.lower() in r["cuisine"].lower()]
        if matched:
            return matched[:limit], True
        return rows[:limit], False
    return rows[:limit], True


@tool
async def find_restaurants(near: str, city: str, cuisine: str = "", vegetarian: bool = False,
                           radius_m: int = 800, limit: int = 5) -> dict:
    """Find real restaurants and cafes (OpenStreetMap) within radius_m of a place, nearest
    first, with cuisine and opening hours. Use it for each day's lunch and dinner near that
    day's activities. cuisine is a single word such as 'portuguese', 'seafood', 'pizza'."""
    radius_m = max(200, min(radius_m, 3000))
    try:
        async with new_client() as client:
            center = await geocode_city(client, city)
            place = await geocode_place(client, near, center) if center else None
            if place is None:
                return {"error": f"could not locate '{near}' in {city}"}
            try:
                elements = await fetch_elements(client, place.lat, place.lon, radius_m)
            except httpx.HTTPError:
                elements = await nominatim_elements(client, place.lat, place.lon, radius_m)
    except httpx.HTTPError as e:
        return {"error": f"restaurant lookup failed: {e!r}"}
    rows, matched = pick_restaurants(elements, place.lat, place.lon, cuisine, vegetarian, max(1, min(limit, 10)))
    result = {"near": near, "radius_m": radius_m, "restaurants": rows}
    if not matched:
        result["note"] = f"no '{cuisine}' restaurants found nearby; showing the nearest of any cuisine"
    if not rows:
        result["note"] = "nothing found; try a larger radius_m"
    return result
