# AI Travel Planner

[![CI](https://github.com/mugil0095/ai-travel-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/mugil0095/ai-travel-planner/actions/workflows/ci.yml)

AI Travel Planner turns a trip request (destination, dates, budget, interests) into a timed, day-by-day itinerary. A research agent gathers live destination facts and a planner agent builds the plan, then the workflow pauses so a person can approve, reject or modify the draft. It is built with LangGraph and FastAPI.

**Stack:** Python 3.12, LangGraph, FastAPI, Google Gemini (LLM), Exa (web search), SQLite (checkpoints), OpenStreetMap (Nominatim, Overpass), Open-Meteo.

[Design report (PDF)](docs/AI_Travel_Planner_Report.pdf), Word version: [docs/AI_Travel_Planner_Report.docx](docs/AI_Travel_Planner_Report.docx)

## How it works

```mermaid
flowchart LR
    A([POST /plan]) --> V[validate] --> R[research<br/>Research Agent] --> P[plan<br/>Planner Agent]
    P -- over budget, once --> P
    P --> H{{review<br/>human}}
    H -- approve --> F[finalize] --> Z([GET /final])
    H -- reject: new info --> R
    H -- reject: arrangement --> P
    H -- modify --> M[modify] --> H
```

- **Orchestrator** (`app/graph.py`): a LangGraph `StateGraph` that validates the request, runs the two agents and pauses at the review step with `interrupt()`. If the draft is over budget, the planner gets one extra try.
- **Research agent**: `web_search` (Exa) and `weather_outlook` (Open-Meteo forecast, or last year's weather on the same dates for trips too far ahead).
- **Planner agent**: `optimize_schedule` finds places on OpenStreetMap, groups them into days and orders each day by opening hours and travel time. `find_restaurants` returns real restaurants near a place.
- **Human review**: approve finalizes the draft as it is. Modify changes only the days you name, and a change without a day (such as the hotel) updates only the lodging and summary. The code enforces both. Reject needs feedback and goes back to the research agent (new information) or the planner (rearrange). An LLM router picks, and the reviewer can override it.
- **Checkpoints**: every step is saved to SQLite. A plan waiting for review survives a restart, and a failed run can be resumed.

## Project structure

```
app/
  main.py              FastAPI app and endpoints
  graph.py             LangGraph workflow: nodes, routing, review pause
  agents.py            Research and planner agents, prompts, feedback router
  runner.py            Runs the graph in the background, reports status, recovers after restart
  schemas.py           Pydantic models for requests, reviews and agent outputs
  config.py            Environment variables and defaults
  tools/
    search.py          web_search (Exa)
    weather.py         weather_outlook (Open-Meteo)
    scheduler.py       optimize_schedule (groups and orders places)
    restaurants.py     find_restaurants (OpenStreetMap)
    geo.py             Geocoding and distance helpers (Nominatim)
tests/                 45 offline tests, plus 4 live checks
docs/                  Design report (PDF and Word)
.github/workflows/     CI: runs the tests and builds the Docker image on every push
Dockerfile             Container image
requirements.txt       Runtime dependencies
requirements-dev.txt   Adds pytest
.env.example           Template for keys and settings
pytest.ini             Test settings
```

## Setup

Requires Python 3.12.

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
```

Then open `.env` and add your keys.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GOOGLE_API_KEY` | yes | none | From [Google AI Studio](https://aistudio.google.com/apikey) |
| `EXA_API_KEY` | yes | none | From the [Exa dashboard](https://dashboard.exa.ai/api-keys) |
| `GEMINI_MODEL` | no | `gemini-3.5-flash` | Main model |
| `GEMINI_FALLBACK_MODELS` | no | `gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite` | Tried in order when the main model is overloaded (503) or rate limited (429) |
| `DB_PATH` | no | `data/checkpoints.sqlite` | SQLite checkpoint file |
| `MAX_REVISIONS` | no | `5` | Reject or modify rounds allowed before only approve is accepted |

The Gemini free tier is limited (about 20 requests per model per day was seen), so the fallback models help.

## Run

```bash
uvicorn app.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

Docker (CI builds this image and checks `/health` on every push):

```bash
docker build -t travel-planner .
docker run -p 8000:8000 --env-file .env -v planner-data:/app/data travel-planner
```

## API

Work runs in the background, so POSTs return `202` at once. Poll `GET /plan/{id}` or listen on `/events` to follow progress.

### Endpoints

| Method | Path | Purpose | Status codes |
|---|---|---|---|
| POST | `/plan` | Submit a trip request | 202, 422 |
| GET | `/plan/{id}` | Status, current draft, research, review history, progress events | 200, 404 |
| POST | `/plan/{id}/review` | Approve, reject (with feedback) or modify (with changes) | 202, 404, 409, 422 |
| GET | `/plan/{id}/final` | The approved plan | 200, 404, 409 before approval |
| GET | `/plan/{id}/events` | Server-sent events (SSE), one event per status change | 200, 404 |
| POST | `/plan/{id}/retry` | Resume a failed or stalled run from its last checkpoint | 202, 404, 409 |
| GET | `/health` | Model in use and whether the keys are set | 200 |

### Request fields (`POST /plan`)

| Field | Type | Rules |
|---|---|---|
| `destination` | text | 2 to 100 characters |
| `start_date`, `end_date` | date (`YYYY-MM-DD`) | Start today or later, end on or after start, at most 14 days |
| `budget_min`, `budget_max` | number | Total for the whole group, `budget_min` not above `budget_max` |
| `currency` | text | 3 uppercase letters, default `USD` |
| `travelers` | integer | 1 to 20 |
| `interests` | list of text | 1 to 10 items |
| `notes` | text | Optional, up to 500 characters (pace, diet, mobility) |

### Review body (`POST /plan/{id}/review`)

| Field | Used with | Rules |
|---|---|---|
| `action` | all | `approve`, `reject` or `modify` |
| `feedback` | reject | Required, up to 2000 characters |
| `revise` | reject | Optional: `research` or `itinerary`. If left out, an LLM router decides |
| `modifications` | modify | 1 to 10 items of `{"day", "change"}`. `day` must exist in the draft; leave it out for a plan-wide change such as the hotel, which can update the lodging and summary but never the days. `change` is 3 to 500 characters |

### Plan statuses

| Status | Meaning | What to do |
|---|---|---|
| `queued`, `validating`, `researching`, `planning`, `modifying`, `finalizing`, `processing` | Work in progress | Keep polling |
| `awaiting_review` | A draft is ready | Send a review |
| `completed` | The plan was approved | `GET /plan/{id}/final` |
| `failed` | A step raised an error (see the `error` field) | `POST /plan/{id}/retry` |
| `stalled` | The server stopped mid-run (normally resumed on startup) | `POST /plan/{id}/retry` |
| `invalid` | The request failed validation | Fix it and submit a new plan |

### Example flow (real run)

These are real responses from one run (Lisbon, 3 days, 2 travellers), trimmed with `…`. The commands use bash. On Windows PowerShell, `curl` is an alias for another command, so use the `/docs` page or `Invoke-RestMethod` instead.

**1. Create a plan**

```bash
curl -X POST localhost:8000/plan -H "Content-Type: application/json" -d '{
  "destination": "Lisbon, Portugal", "start_date": "2026-11-20", "end_date": "2026-11-22",
  "budget_min": 900, "budget_max": 1600, "currency": "EUR", "travelers": 2,
  "interests": ["food", "history", "viewpoints"], "notes": "Relaxed pace; one of us is vegetarian."}'
```

`202 Accepted`

```json
{"plan_id": "0d12bf31…", "status": "queued",
 "links": {"status": "/plan/0d12bf31…", "events": "/plan/0d12bf31…/events",
           "review": "/plan/0d12bf31…/review", "final": "/plan/0d12bf31…/final"}}
```

**2. Poll until the draft is ready.** `curl localhost:8000/plan/<plan_id>` returns `200`. After a few minutes the status is `awaiting_review`:

```json
{"plan_id": "0d12bf31…", "status": "awaiting_review", "revision": 1,
 "allowed_actions": ["approve", "reject", "modify"],
 "draft": {"title": "Lisbon Autumn Weekend: History, Viewpoints & Culinary Flavors",
           "budget": {"total": 897.0, "currency": "EUR", "status": "under"}, "…": "…"}}
```

<details>
<summary>Full draft response (trimmed)</summary>

```json
{
  "plan_id": "0d12bf31…",
  "status": "awaiting_review",
  "revision": 1,
  "allowed_actions": ["approve", "reject", "modify"],
  "draft": {
    "title": "Lisbon Autumn Weekend: History, Viewpoints & Culinary Flavors",
    "lodging": {"name": "Boutique Hotel in Chiado", "area": "Chiado", "cost_per_night": 180.0},
    "days": [
      {
        "day": 1,
        "date": "2026-11-20",
        "theme": "Panoramic Viewpoints, Moorish Fortress & Historic Baixa",
        "activities": [
          {"start": "09:43", "end": "10:43", "name": "Miradouro da Senhora do Monte",
           "cost_per_person": 0.0, "travel_minutes_from_previous": 13},
          "…"
        ],
        "meals": [
          {"meal": "breakfast", "name": "Pastelaria Bénard", "cuisine": "Portuguese Bakery & Cafe",
           "cost_per_person": 8.0},
          "…"
        ]
      },
      "… days 2-3"
    ],
    "budget": {"currency": "EUR", "lodging": 360.0, "activities": 138.0, "meals": 354.0,
               "local_transport": 45.0, "total": 897.0, "budget_min": 900.0, "budget_max": 1600.0,
               "status": "under"}
  },
  "research": "{ … }",
  "reviews": [],
  "events": [
    "Request validated: 3 days in Lisbon, Portugal for 2",
    "Research agent: 10 attractions, 9 sources (weather_outlook x1, web_search x6)",
    "Planner agent: draft 1, estimated 897.0 EUR (under budget; optimize_schedule x1, find_restaurants x5)"
  ]
}
```

In the real response each event is an object with `at` (timestamp) and `message`.
</details>

**3. Review the draft.** Each review returns `202` right away:

```bash
curl -X POST localhost:8000/plan/<plan_id>/review -H "Content-Type: application/json" \
  -d '{"action": "reject", "feedback": "Too much time on public transport. Keep each day in one area and add a fado evening."}'
```

```json
{"plan_id": "0d12bf31…", "status": "processing", "action": "reject", "links": {"…": "…"}}
```

When the new draft is ready, `GET /plan/<plan_id>` shows where the feedback went:

```json
{"status": "awaiting_review", "revision": 2,
 "reviews": [{"action": "reject", "routed_to": "research",
              "routing_reason": "The feedback requests adding a fado evening, which introduces a new cultural interest and event type requiring new destination options and information."}],
 "draft": {"budget": {"total": 1417.0, "status": "within"}, "…": "…"}}
```

Modify changes only the named day. Body, then the result:

```json
{"action": "modify", "modifications": [{"day": 2, "change": "Replace the afternoon with a sunset at a viewpoint and a vegetarian dinner nearby."}]}
```

```json
{"status": "awaiting_review", "revision": 3,
 "events": ["…", "Planner agent applied 1 modification(s): draft 3, estimated 1221.0 EUR (find_restaurants x2)"]}
```

Approve with `{"action": "approve"}`. The response is `202` with `"status": "processing"`, and the plan becomes `completed`.

**4. Get the final plan.** `curl localhost:8000/plan/<plan_id>/final` returns `200`:

```json
{"plan_id": "0d12bf31…", "approved_at": "2026-10-22T18:08:03+00:00", "approved_revision": 3,
 "itinerary": {"title": "Lisbon Alfama & Fado Immersion: History, Viewpoints & Culinary Evenings",
               "budget": {"total": 1221.0, "status": "within"}, "…": "…"},
 "destination_brief": {"sources": ["https://www.clubedefado.pt/en/", "https://fadoaocarmo.pt/en/", "…"], "…": "…"},
 "review_history": [{"revision": 1, "action": "reject", "routed_to": "research"},
                    {"revision": 2, "action": "modify"},
                    {"revision": 3, "action": "approve"}]}
```

`itinerary` is the approved draft unchanged. `destination_brief` also holds the weather, local tips, safety notes and seasonal notes.

**Live status.** `curl -N localhost:8000/plan/<plan_id>/events` streams one event per status change and closes when the plan is finished:

```text
event: status
data: {"status": "completed", "revision": 3, "allowed_actions": [], "latest": {"at": "2026-10-22T18:08:03+00:00", "message": "Final plan produced"}}
```

**Health.** `curl localhost:8000/health`:

```json
{"status": "ok", "model": "gemini-3.8-flash", "keys_configured": {"google": true, "exa": true}}
```

### Errors (real responses)

| Code | When | Response |
|---|---|---|
| 404 | Unknown plan id | `{"detail": "plan does-not-exist not found"}` |
| 409 | Review when the plan is not waiting for one | `{"detail": {"message": "plan is not awaiting review", "status": "completed", "allowed_actions": []}}` |
| 409 | Final plan before approval | `{"detail": {"message": "the final plan is only available after approval", "status": "awaiting_review", …}}` |
| 409 | Retry a plan that has not failed | `{"detail": {"message": "only failed or stalled plans can be retried", "status": "completed", …}}` |
| 422 | Invalid request | `"msg": "Value error, end_date must be on or after start_date"` |
| 422 | Invalid review body | `"msg": "Value error, reject requires feedback"` |

A `409` body always includes the current `status` and the `allowed_actions`, so a client knows what it can do next.

## Tests

```bash
pytest
```

This runs 45 offline tests and needs no API keys. The agents are faked, and one wiring test runs the real agents with a scripted model. GitHub Actions runs the same tests, then builds and smoke-tests the Docker image, on every push.

Live checks call Open-Meteo, Nominatim, Overpass and Exa (Exa needs a key). Run them with `RUN_LIVE=1 pytest -m live` in bash, or `$env:RUN_LIVE=1; pytest -m live` in PowerShell.

## Design decisions and tradeoffs

| Decision | Why | Tradeoff |
|---|---|---|
| Async API (202, then polling or SSE) | A run takes minutes; requests never wait on the LLM | Clients must poll or keep an SSE connection open |
| The checkpoint is the single source of truth | Restart recovery and `/retry` need no extra bookkeeping | Startup recovery scans all checkpoints, which gets slower as history grows |
| Typed agent outputs, plain Python for totals, the modify guard and the final plan | The exact parts cannot be changed by the LLM | Schema checks cannot catch content mistakes (the Lisbon "vegetarian" dinner was not confirmed vegetarian) |
| Reject routed to research or the planner | Feedback reaches the agent that can act on it | One extra LLM call; it can misroute (the reviewer can override, and a failed call falls back to the planner) |
| Model fallback chain, per-tool call caps, OpenStreetMap fallbacks | Live runs hit 503s, quota limits and runaway searches | A fallback model can produce a weaker draft |
| Free, keyless data for 3 of 4 tools | The project runs with two API keys | Travel times are estimates (straight line x 1.3, a fixed speed per mode), and OpenStreetMap coverage varies |
| One process | Simple to run and test | No safe multi-worker setup yet; scaling needs a job queue and Postgres |

## What I would improve with more time

- An evaluation set (a single fado evening, a verified vegetarian meal, narrow budgets, closed attractions) to compare prompts and Gemini models.
- Better tools: real road routes (OSRM), parsing of OpenStreetMap `opening_hours`, and a currency tool with dated exchange rates.
- Stream agent tool calls into the SSE feed, not only status changes.
- A small review UI that shows changed days next to the previous draft.

## Production concerns

- **Scale:** run graphs from a job queue, use a shared checkpointer such as Postgres, and lock plans in the database so two workers cannot resume the same one.
- **Reliability:** idempotency keys for submissions and reviews, per-step timeouts, circuit breakers and caching for external APIs.
- **Security and cost:** authentication, per-user plan ownership and quotas, token and cost tracking per plan. The 5-round limit only caps reject and modify.
- **Observability:** logs with `plan_id`, traces of agent steps, latency and error metrics, and no personal details in logs.
- **Safety:** treat web search results as untrusted input to the LLM (prompt injection), and respect the OpenStreetMap usage policies.
- **Data:** retention, cleanup and backups for the SQLite checkpoints, which hold trip requests and review history.

## Assumptions

- The budget is the total for the whole group: lodging, food, activities and local transport. It excludes travel to the destination, such as flights.
- Costs are in the request currency, and Gemini converts local prices.
- Trips are 1 to 14 days, start today or later, and have 1 to 20 travellers. Nights = days minus 1.
- Modify means targeted changes to named days (or a plan-wide change such as the hotel). Reject means the draft needs a new research or planning pass.
- Prices, opening hours and events come from web search and Gemini, so travellers should confirm them before booking.
