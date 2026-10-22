"""Runs graph executions as background tasks and derives each plan's status from its checkpoint."""
import asyncio
import logging

from app import config

log = logging.getLogger(__name__)

NODE_STATUS = {
    "validate": "validating",
    "research": "researching",
    "plan": "planning",
    "modify": "modifying",
    "finalize": "finalizing",
}
TERMINAL = {"completed", "failed", "invalid"}


class Runner:
    def __init__(self, graph):
        self.graph = graph
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()  # makes "check status, then start a run" atomic per process

    @staticmethod
    def config(plan_id: str) -> dict:
        return {"configurable": {"thread_id": plan_id}}

    def running(self, plan_id: str) -> bool:
        task = self.tasks.get(plan_id)
        return task is not None and not task.done()

    def start(self, plan_id: str, payload) -> None:
        """payload: initial state, Command(resume=...) after a review, or None to resume a checkpoint."""
        self.tasks[plan_id] = asyncio.create_task(self._run(plan_id, payload))

    async def _run(self, plan_id: str, payload) -> None:
        try:
            # "sync": each checkpoint is on disk before the next step starts, so a crash loses no finished step.
            await self.graph.ainvoke(payload, self.config(plan_id), durability="sync")
        except Exception:
            # The checkpointer keeps the failed task and its error: status() reports it, /retry resumes.
            log.exception("plan %s failed", plan_id)

    async def status(self, plan_id: str) -> dict | None:
        snap = await self.graph.aget_state(self.config(plan_id))
        if not snap.values:
            return {"plan_id": plan_id, "status": "queued"} if self.running(plan_id) else None
        values = snap.values
        error = next((t.error for t in snap.tasks if t.error), None)

        if self.running(plan_id):
            # Between a step's writes and its checkpoint the snapshot can show no next node, and right
            # after a resume it still shows the old interrupt or error: never report those as final.
            status = NODE_STATUS.get(snap.next[0], snap.next[0]) if snap.next and not (snap.interrupts or error) \
                else "processing"
        elif snap.interrupts:
            status = "awaiting_review"
        elif error:
            status = "failed"
        elif not snap.next:
            status = "completed" if values.get("final_plan") else "invalid"
        else:
            status = "stalled"  # the process stopped mid-run; recover() or /retry resumes it

        rounds = sum(1 for r in values.get("reviews", []) if r["action"] != "approve")
        actions = {
            "awaiting_review": ["approve"] + (["reject", "modify"] if rounds < config.MAX_REVISIONS else []),
            "failed": ["retry"],
            "stalled": ["retry"],
        }.get(status, [])
        return {
            "plan_id": plan_id,
            "status": status,
            "revision": values.get("revision", 0),
            "allowed_actions": actions,
            "error": str(error)[:500] if error else values.get("error"),  # checkpointed errors are repr strings
            "request": values.get("request"),
            "draft": values.get("draft"),
            "research": values.get("research"),
            "reviews": values.get("reviews", []),
            "events": values.get("events", []),
            "final_plan": values.get("final_plan"),
        }

    async def recover(self, checkpointer) -> None:
        """On startup, resume runs that were in progress when the previous process stopped."""
        thread_ids = set()
        async for cp in checkpointer.alist(None):
            thread_ids.add(cp.config["configurable"]["thread_id"])
        for plan_id in thread_ids:
            s = await self.status(plan_id)
            if s and s["status"] == "stalled":
                log.info("resuming plan %s", plan_id)
                self.start(plan_id, None)

    async def shutdown(self) -> None:
        """Cancelled runs keep their last checkpoint and are resumed by recover() on the next start."""
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
