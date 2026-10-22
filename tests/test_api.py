"""API tests: status codes, the full HITL flow over HTTP, SSE and restart persistence (fake agents)."""
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import create_app
from conftest import trip_request


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "api.sqlite")


@pytest.fixture
def client(db, fake):
    with TestClient(create_app(db)) as c:
        yield c


def wait_for(client, plan_id, status, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/plan/{plan_id}").json()
        if body["status"] == status:
            return body
        time.sleep(0.05)
    raise AssertionError(f"plan stayed in {body['status']}, expected {status}")


def create(client, **req) -> str:
    r = client.post("/plan", json=trip_request(**req))
    assert r.status_code == 202, r.text
    return r.json()["plan_id"]


@pytest.mark.parametrize("override", [
    {"end_date": "2000-01-01"},                    # before start
    {"start_date": "2001-01-01", "end_date": "2001-01-02"},  # in the past
    {"interests": []},
    {"travelers": 0},
    {"budget_min": 5000},                          # above budget_max
    {"currency": "euro"},
])
def test_invalid_requests_are_422(client, override):
    assert client.post("/plan", json=trip_request() | override).status_code == 422


def test_trip_length_limit(client):
    assert client.post("/plan", json=trip_request(days=15)).status_code == 422


def test_unknown_plan_is_404(client):
    assert client.get("/plan/nope").status_code == 404
    assert client.get("/plan/nope/final").status_code == 404
    assert client.post("/plan/nope/review", json={"action": "approve"}).status_code == 404


def test_full_flow_reject_modify_approve(client, fake):
    plan_id = create(client)
    body = wait_for(client, plan_id, "awaiting_review")
    assert body["revision"] == 1 and body["allowed_actions"] == ["approve", "reject", "modify"]
    assert len(body["draft"]["days"]) == 3 and body["research"]["summary"]

    r = client.get(f"/plan/{plan_id}/final")
    assert r.status_code == 409 and r.json()["detail"]["status"] == "awaiting_review"

    # body validation
    assert client.post(f"/plan/{plan_id}/review", json={"action": "reject"}).status_code == 422
    assert client.post(f"/plan/{plan_id}/review", json={"action": "modify"}).status_code == 422
    r = client.post(f"/plan/{plan_id}/review",
                    json={"action": "modify", "modifications": [{"day": 9, "change": "beach"}]})
    assert r.status_code == 422 and "not in the plan" in r.text

    r = client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "Slower pace please"})
    assert r.status_code == 202
    body = wait_for(client, plan_id, "awaiting_review")
    assert body["revision"] == 2 and body["reviews"][0]["routed_to"] == "itinerary"

    client.post(f"/plan/{plan_id}/review", json={"action": "modify", "modifications": [{"day": 1, "change": "Later start"}]})
    body = wait_for(client, plan_id, "awaiting_review")
    assert body["revision"] == 3 and body["draft"]["days"][0]["theme"] == "Changed 1"

    assert client.post(f"/plan/{plan_id}/review", json={"action": "approve"}).status_code == 202
    wait_for(client, plan_id, "completed")
    final = client.get(f"/plan/{plan_id}/final").json()
    assert final["approved_revision"] == 3
    assert [r["action"] for r in final["review_history"]] == ["reject", "modify", "approve"]
    assert "final_plan" not in client.get(f"/plan/{plan_id}").json()

    r = client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    assert r.status_code == 409 and r.json()["detail"]["status"] == "completed"


def test_revision_limit(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_REVISIONS", 1)
    plan_id = create(client)
    wait_for(client, plan_id, "awaiting_review")
    client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "again"})
    body = wait_for(client, plan_id, "awaiting_review")
    assert body["allowed_actions"] == ["approve"]
    r = client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "again"})
    assert r.status_code == 409 and "revision limit" in r.text


def test_failed_run_reports_error_and_retry_recovers(client, fake):
    fake.fail_planner = True
    plan_id = create(client)
    body = wait_for(client, plan_id, "failed")
    assert "LLM quota exceeded" in body["error"] and body["allowed_actions"] == ["retry"]
    assert client.post(f"/plan/{plan_id}/review", json={"action": "approve"}).status_code == 409

    fake.fail_planner = False
    assert client.post(f"/plan/{plan_id}/retry").status_code == 202
    wait_for(client, plan_id, "awaiting_review")
    assert fake.names().count("research") == 1


def test_event_stream(client):
    plan_id = create(client)
    wait_for(client, plan_id, "awaiting_review")
    client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    wait_for(client, plan_id, "completed")
    with client.stream("GET", f"/plan/{plan_id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert "event: status" in text and '"status": "completed"' in text


def test_plan_awaiting_review_survives_restart(db, fake):
    with TestClient(create_app(db)) as c:
        plan_id = create(c)
        wait_for(c, plan_id, "awaiting_review")
    with TestClient(create_app(db)) as c:  # a new process on the same database
        assert c.get(f"/plan/{plan_id}").json()["status"] == "awaiting_review"
        c.post(f"/plan/{plan_id}/review", json={"action": "approve"})
        wait_for(c, plan_id, "completed")
        assert c.get(f"/plan/{plan_id}/final").status_code == 200
