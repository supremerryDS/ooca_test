from __future__ import annotations

from collections.abc import Callable

import pytest
from fakes import DATA_DIR

from triage_agent.agent import TriageAgent
from triage_agent.bootstrap import build_tools, load_tickets
from triage_agent.llm.base import LLMClient
from triage_agent.policy import PolicyEngine, default_rules
from triage_agent.prompts import load_system_prompt
from triage_agent.schemas import Ticket
from triage_agent.tools import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return build_tools(DATA_DIR)


@pytest.fixture
def tickets() -> dict[str, Ticket]:
    return {t.ticket_id: t for t in load_tickets(DATA_DIR / "sample_tickets.json")}


@pytest.fixture
def make_agent(registry: ToolRegistry) -> Callable[..., TriageAgent]:
    def _make(llm: LLMClient, max_steps: int = 6) -> TriageAgent:
        return TriageAgent(
            llm=llm,
            tools=registry,
            policy=PolicyEngine(default_rules(confidence_threshold=0.6)),
            system_prompt=load_system_prompt(),
            max_steps=max_steps,
        )

    return _make
