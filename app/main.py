"""FastAPI service for the travel-planning workflow.

Graph runs happen in background tasks, so every POST returns 202 at once; clients follow
progress with GET /plan/{id} or the server-sent events stream at GET /plan/{id}/events.
"""
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from app import config
from app.graph import build_graph, open_checkpointer
from app.runner import TERMINAL, Runner
from app.schemas import ReviewRequest, TravelRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
router = APIRouter()


def get_runner(request: Request) -> Runner:
    return request.app.state.runner


def links(plan_id: str) -> dict:
    base = f"/plan/{plan_id}"
    return {"status": base, "events": f"{base}/events", "review": f"{base}/review", "final": f"{base}/final"}


async def require_status(runner: Runner, plan_id: str) -> dict:
    s = await runner.status(plan_id)
    if s is None:
        raise HTTPException(404, f"plan {plan_id} not found")
    return s


def conflict(message: str, s: dict) -> HTTPException:
    return HTTPException(409, {"message": message, "status": s["status"], "allowed_actions": s["allowed_actions"]})


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": config.GEMINI_MODEL,
            "keys_configured": {"google": bool(config.GOOGLE_API_KEY), "exa": bool(config.EXA_API_KEY)}}


@router.post("/plan", status_code=202)
async def create_plan(body: TravelRequest, runner: Runner = Depends(get_runner)) -> dict:
    plan_id = uuid.uuid4().hex
    runner.start(plan_id, {"request": body.model_dump(mode="json")})
    return {"plan_id": plan_id, "status": "queued", "links": links(plan_id)}


@router.get("/plan/{plan_id}")
async def get_plan(plan_id: str, runner: Runner = Depends(get_runner)) -> dict:
    s = await require_status(runner, plan_id)
    s.pop("final_plan", None)
    return s | {"links": links(plan_id)}


@router.post("/plan/{plan_id}/review", status_code=202)
async def review_plan(plan_id: str, body: ReviewRequest, runner: Runner = Depends(get_runner)) -> dict:
    async with runner.lock:
        s = await require_status(runner, plan_id)
        if s["status"] != "awaiting_review":
            raise conflict("plan is not awaiting review", s)
        if body.action not in s["allowed_actions"]:
            raise conflict(f"revision limit ({config.MAX_REVISIONS}) reached: approve the plan", s)
        n_days = len(s["draft"]["days"])
        bad = [m.day for m in body.modifications if m.day and m.day > n_days]
        if bad:
            raise HTTPException(422, f"day(s) {bad} not in the plan (days 1-{n_days})")
        runner.start(plan_id, Command(resume=body.model_dump(mode="json")))
    return {"plan_id": plan_id, "status": "processing", "action": body.action, "links": links(plan_id)}


@router.get("/plan/{plan_id}/final")
async def final_plan(plan_id: str, runner: Runner = Depends(get_runner)) -> dict:
    s = await require_status(runner, plan_id)
    if s["status"] != "completed":
        raise conflict("the final plan is only available after approval", s)
    return s["final_plan"]


@router.post("/plan/{plan_id}/retry", status_code=202)
async def retry_plan(plan_id: str, runner: Runner = Depends(get_runner)) -> dict:
    async with runner.lock:
        s = await require_status(runner, plan_id)
        if "retry" not in s["allowed_actions"]:
            raise conflict("only failed or stalled plans can be retried", s)
        runner.start(plan_id, None)  # resumes from the last checkpoint, re-running the failed step
    return {"plan_id": plan_id, "status": "processing", "links": links(plan_id)}


@router.get("/plan/{plan_id}/events")
async def plan_events(plan_id: str, request: Request, runner: Runner = Depends(get_runner)) -> StreamingResponse:
    """Server-sent events: one `status` event per change; the stream ends when the plan is terminal."""
    await require_status(runner, plan_id)

    async def stream():
        last = None
        while not await request.is_disconnected():
            s = await runner.status(plan_id)
            events = s.get("events", [])
            key = (s["status"], s.get("revision"), len(events))
            if key != last:
                last = key
                data = {"status": s["status"], "revision": s.get("revision"),
                        "allowed_actions": s.get("allowed_actions"), "latest": events[-1] if events else None}
                yield f"event: status\ndata: {json.dumps(data)}\n\n"
            if s["status"] in TERMINAL:
                break
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


def create_app(db_path: str = config.DB_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        async with open_checkpointer(db_path) as saver:
            runner = Runner(build_graph(saver))
            await runner.recover(saver)
            app.state.runner = runner
            yield
            await runner.shutdown()

    app = FastAPI(
        title="AI Travel Planner",
        description="Multi-agent trip planning (LangGraph) with human-in-the-loop review.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
