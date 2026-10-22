"""Research tool 1: real-time web search through the Exa API."""
import httpx
from langchain_core.tools import tool

from app import config

EXA_URL = "https://api.exa.ai/search"


async def exa_search(client: httpx.AsyncClient, query: str, num_results: int = 5) -> list[dict]:
    r = await client.post(
        EXA_URL,
        headers={"x-api-key": config.EXA_API_KEY},
        json={
            "query": query,
            "type": "auto",
            "numResults": num_results,
            "contents": {"highlights": {"maxCharacters": 1500}},
        },
    )
    r.raise_for_status()
    return [
        {
            "title": res.get("title") or "",
            "url": res["url"],
            "published": (res.get("publishedDate") or "")[:10],
            "highlights": " ... ".join(res.get("highlights") or []),
        }
        for res in r.json().get("results", [])
    ]


@tool
async def web_search(query: str) -> list[dict] | dict:
    """Search the web for up-to-date destination information: attractions, neighbourhoods,
    local tips, safety advisories, events on the travel dates, prices. Returns the top 5 pages
    with relevant highlights and URLs. Use specific queries, one topic per call."""
    if not config.EXA_API_KEY:
        return {"error": "EXA_API_KEY is not set; web search is unavailable"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            return await exa_search(client, query)
    except httpx.HTTPError as e:
        return {"error": f"web search failed: {e}"}
