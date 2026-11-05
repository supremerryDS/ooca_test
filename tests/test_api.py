from __future__ import annotations

import pytest
from fakes import ScriptedLLM, decision_payload, respond, tool_call
from fastapi.testclient import TestClient

from triage_agent.api import create_app
from triage_agent.config import Settings


def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM(
        [
            respond(tool_call("search_knowledge_base", {"query": "dark mode", "category": None, "top_k": None})),
            respond(tool_call("submit_triage", decision_payload())),
        ]
    )


@pytest.fixture
def client(make_agent) -> TestClient:
    return TestClient(create_app(Settings(), agent=make_agent(scripted_llm())))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model" in body


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_list_samples(client):
    ids = [t["ticket_id"] for t in client.get("/samples").json()]
    assert ids == ["T-1001", "T-1002", "T-1003"]


def test_triage_sample_by_id(client):
    response = client.post("/samples/T-1003/triage")
    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "T-1003"
    assert body["decision"]["action"] == "auto_respond"
    assert body["tool_calls"][0]["tool_name"] == "search_knowledge_base"


def test_triage_custom_ticket(client):
    ticket = {
        "ticket_id": "T-9000",
        "customer_id": "cus_3003",
        "channel": "email",
        "messages": [{"author": "customer", "sent_at": "2026-11-03T09:00:00+07:00", "text": "Do you have dark mode?"}],
    }
    response = client.post("/triage", json=ticket)
    assert response.status_code == 200
    assert response.json()["ticket_id"] == "T-9000"


def test_invalid_ticket_is_rejected(client):
    response = client.post("/triage", json={"ticket_id": "T-1", "customer_id": "cus_1", "messages": []})
    assert response.status_code == 422


def ticket_body(**overrides) -> dict:
    body = {
        "ticket_id": "T-9100",
        "customer_id": "cus_3003",
        "messages": [{"sent_at": "2026-11-03T09:00:00+07:00", "text": "Do you have dark mode?"}],
    }
    return {**body, **overrides}


@pytest.mark.parametrize(
    "body",
    [
        ticket_body(ticket_id=""),
        ticket_body(customer_id="   "),
        ticket_body(messages=[{"sent_at": "2026-11-03T09:00:00+07:00", "text": "   "}]),
        ticket_body(messages=[{"sent_at": "2026-11-03T09:00:00+07:00", "text": "x" * 10_001}]),
        ticket_body(messages=[{"sent_at": "2026-11-03T09:00:00+07:00", "text": "hi"}] * 101),
    ],
    ids=["empty-id", "blank-customer", "blank-text", "text-too-long", "too-many-messages"],
)
def test_abnormal_tickets_are_rejected(client, body):
    assert client.post("/triage", json=body).status_code == 422


def test_unknown_sample_is_404(client):
    assert client.post("/samples/T-404/triage").status_code == 404


def test_missing_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app(Settings()))  # no injected agent: must build a real one
    response = client.post("/samples/T-1001/triage")
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_docs_show_sample_tickets_as_examples(client):
    schema = client.get("/openapi.json").json()
    examples = schema["paths"]["/triage"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert set(examples) == {"T-1001", "T-1002", "T-1003"}
