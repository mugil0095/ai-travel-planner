"""Research tool 2: daily weather for the trip dates from Open-Meteo (free, no key).

Trips within the 16-day forecast window get a real forecast. Later trips get last year's
observed weather on the same dates, which is a better guide than a generic monthly average.
"""
from datetime import date, timedelta

import httpx
from langchain_core.tools import tool

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_DAYS = 16  # including today
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum"


def last_year(d: date) -> date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:  # 29 February
        return d.replace(year=d.year - 1, day=28)


def _avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


async def trip_weather(client: httpx.AsyncClient, destination: str, start: date, end: date, today: date) -> dict:
    name, _, country = (part.strip() for part in destination.partition(","))
    r = await client.get(GEOCODE, params={"name": name, "count": 5})
    r.raise_for_status()
    hits = r.json().get("results") or []
    if country:
        hits = [h for h in hits if country.lower() in h.get("country", "").lower()] or hits
    if not hits:
        return {"error": f"could not locate {destination}"}
    loc = hits[0]

    is_forecast = end < today + timedelta(days=FORECAST_DAYS)
    q_start, q_end = (start, end) if is_forecast else (last_year(start), last_year(end))
    r = await client.get(
        FORECAST if is_forecast else ARCHIVE,
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "start_date": q_start.isoformat(),
            "end_date": q_end.isoformat(),
            "daily": DAILY,
            "timezone": "auto",
        },
    )
    r.raise_for_status()
    daily = r.json()["daily"]
    highs, lows, rain = daily["temperature_2m_max"], daily["temperature_2m_min"], daily["precipitation_sum"]
    return {
        "location": f"{loc['name']}, {loc.get('country', '')}".strip(", "),
        "source": "forecast" if is_forecast
        else f"observed weather on the same dates in {q_start.year} (a climate guide, not a forecast)",
        "days": [
            {"date": (start + timedelta(days=i)).isoformat(), "max_c": highs[i], "min_c": lows[i], "rain_mm": rain[i]}
            for i in range(len(highs))
        ],
        "avg_max_c": _avg(highs),
        "avg_min_c": _avg(lows),
        "wet_days": sum(1 for v in rain if v is not None and v >= 1),
    }


@tool
async def weather_outlook(destination: str, start_date: str, end_date: str) -> dict:
    """Daily max/min temperature (C) and rain (mm) for the trip dates (YYYY-MM-DD).
    A real forecast when the trip is within 16 days, otherwise last year's observed weather
    on the same dates. Call once per trip."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            return await trip_weather(
                client, destination, date.fromisoformat(start_date), date.fromisoformat(end_date), date.today()
            )
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"error": f"weather lookup failed: {e}"}
