"""Composition root: the only place that knows which concrete classes are wired together."""

from __future__ import annotations

import json
from pathlib import Path

from triage_agent.agent import TriageAgent
from triage_agent.config import Settings
from triage_agent.datasources import MockBackend
from triage_agent.llm.base import LLMClient
from triage_agent.policy import PolicyEngine, default_rules
from triage_agent.prompts import load_system_prompt
from triage_agent.schemas import Ticket
from triage_agent.tools import (
    BM25KnowledgeBase,
    CheckServiceStatusTool,
    GetBillingTransactionsTool,
    GetCustomerHistoryTool,
    SearchKnowledgeBaseTool,
    ToolRegistry,
)


def build_tools(data_dir: Path) -> ToolRegistry:
    """Create the four lookup tools backed by the mock data.

    Args:
        data_dir: Folder with `customers.json`, `billing.json`, `service_status.json`
            and `knowledge_base.json`.

    Returns:
        A registry holding every tool the agent may call.
    """
    backend = MockBackend(data_dir)
    knowledge_base = BM25KnowledgeBase.from_json(data_dir / "knowledge_base.json")
    return ToolRegistry(
        [
            GetCustomerHistoryTool(backend),
            SearchKnowledgeBaseTool(knowledge_base),
            GetBillingTransactionsTool(backend),
            CheckServiceStatusTool(backend),
        ]
    )


def build_llm(settings: Settings) -> LLMClient:
    """Create the OpenAI client. The API key is read from `OPENAI_API_KEY` by the SDK.

    Args:
        settings: Model name, reasoning effort and request timeout.

    Returns:
        A client ready to call the configured model.
    """
    from triage_agent.llm.openai_client import OpenAIChatClient  # imported lazily: tests never need it

    return OpenAIChatClient(
        settings.model,
        reasoning_effort=settings.reasoning_effort,
        timeout_s=settings.request_timeout_s,
    )


def build_agent(settings: Settings, llm: LLMClient | None = None) -> TriageAgent:
    """Wire the agent: model client, tools, guardrails and system prompt.

    Args:
        settings: Runtime settings.
        llm: Client to use instead of OpenAI, for example a fake one in tests.

    Returns:
        A ready-to-use agent.
    """
    return TriageAgent(
        llm=llm or build_llm(settings),
        tools=build_tools(settings.data_dir),
        policy=PolicyEngine(default_rules(settings.confidence_threshold)),
        system_prompt=load_system_prompt(),
        max_steps=settings.max_steps,
    )


def load_tickets(path: Path) -> list[Ticket]:
    """Read and validate tickets from a JSON file holding a list of tickets.

    Args:
        path: The JSON file.

    Returns:
        The tickets in file order.

    Raises:
        pydantic.ValidationError: If a ticket does not match the `Ticket` schema.
    """
    with path.open(encoding="utf-8") as fh:
        return [Ticket.model_validate(item) for item in json.load(fh)]
