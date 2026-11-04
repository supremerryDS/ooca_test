"""HTTP API (FastAPI) around the same `TriageAgent` the CLI uses.

    triage-agent serve            # then open http://127.0.0.1:8000/docs

Endpoints
---------
GET  /health                        liveness + configured model
GET  /samples                       the sample tickets (copy one into POST /triage)
POST /samples/{ticket_id}/triage    triage a sample ticket by ID (one click in /docs)
POST /triage                        triage any ticket sent in the request body

The agent is built on the first triage request, so the server (and /docs) starts
even without an OpenAI key; triage requests then return 503 with a clear message.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from triage_agent import __version__
from triage_agent.agent import TriageAgent
from triage_agent.bootstrap import build_agent, load_tickets
from triage_agent.config import PROJECT_ROOT, Settings
from triage_agent.schemas import Ticket, TriageResult

router = APIRouter()


# --------------------------------------------------------------------------- #
# Dependencies                                                                 #
# --------------------------------------------------------------------------- #
class AgentProvider:
    """Builds the agent lazily and reuses it for every request."""

    def __init__(self, settings: Settings, agent: TriageAgent | None = None) -> None:
        self._settings = settings
        self._agent = agent

    def __call__(self) -> TriageAgent:
        if self._agent is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise HTTPException(
                    status_code=503,
                    detail="OPENAI_API_KEY is not set on the server. Add it to .env and restart.",
                )
            self._agent = build_agent(self._settings)
        return self._agent


def get_agent(request: Request) -> TriageAgent:
    return request.app.state.agent_provider()


def get_samples(request: Request) -> dict[str, Ticket]:
    return request.app.state.samples


AgentDep = Annotated[TriageAgent, Depends(get_agent)]
SamplesDep = Annotated[dict[str, Ticket], Depends(get_samples)]


def _sample_examples() -> dict[str, dict[str, Any]]:
    """Sample tickets shown in the /docs example dropdown of POST /triage."""
    path = PROJECT_ROOT / "data" / "sample_tickets.json"
    if not path.exists():
        return {}
    return {
        t.ticket_id: {"summary": f"{t.ticket_id}: {t.messages[0].text[:60]}", "value": t.model_dump(mode="json")}
        for t in load_tickets(path)
    }


TICKET_EXAMPLES = _sample_examples()

_NO_KEY = {503: {"description": "OPENAI_API_KEY is not configured on the server"}}


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# Plain `def` (not `async def`) on purpose: FastAPI runs them in a thread pool, #
# so the blocking OpenAI call does not block the event loop.                   #
# --------------------------------------------------------------------------- #
@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/health", summary="Liveness check")
def health(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    return {"status": "ok", "model": settings.model, "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))}


@router.get("/samples", summary="List the sample tickets")
def list_samples(samples: SamplesDep) -> list[Ticket]:
    return list(samples.values())


@router.post(
    "/samples/{ticket_id}/triage",
    summary="Triage a sample ticket by ID",
    responses={404: {"description": "Unknown sample ticket ID"}, **_NO_KEY},
)
def triage_sample(ticket_id: str, samples: SamplesDep, agent: AgentDep) -> TriageResult:
    ticket = samples.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample ticket '{ticket_id}'. Try: {', '.join(samples)}")
    return agent.triage(ticket)


@router.post("/triage", summary="Triage a ticket", responses=_NO_KEY)
def triage_ticket(ticket: Annotated[Ticket, Body(openapi_examples=TICKET_EXAMPLES)], agent: AgentDep) -> TriageResult:
    """Classify urgency, extract key info, search the knowledge base and decide the next action.

    The response contains the final decision (after policy guardrails), the LLM's original
    decision, any policy overrides, every tool call, token usage and latency.
    """
    return agent.triage(ticket)


# --------------------------------------------------------------------------- #
# App factory                                                                  #
# --------------------------------------------------------------------------- #
def create_app(settings: Settings | None = None, agent: TriageAgent | None = None) -> FastAPI:
    """Build the app. Pass `agent` to inject a pre-built (or fake) agent, e.g. in tests."""
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Support Ticket Triage Agent",
        version=__version__,
        description="AI agent that triages customer-support tickets. "
        "Try **POST /samples/{ticket_id}/triage** with `T-1001`, `T-1002` or `T-1003`.",
    )
    app.state.settings = settings
    app.state.agent_provider = AgentProvider(settings, agent)
    app.state.samples = {t.ticket_id: t for t in load_tickets(settings.data_dir / "sample_tickets.json")}
    app.include_router(router)
    return app
