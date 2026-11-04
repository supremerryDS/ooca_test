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
    from triage_agent.llm.openai_client import OpenAIChatClient  # imported lazily: tests never need it

    return OpenAIChatClient(
        settings.model,
        reasoning_effort=settings.reasoning_effort,
        timeout_s=settings.request_timeout_s,
    )


def build_agent(settings: Settings, llm: LLMClient | None = None) -> TriageAgent:
    return TriageAgent(
        llm=llm or build_llm(settings),
        tools=build_tools(settings.data_dir),
        policy=PolicyEngine(default_rules(settings.confidence_threshold)),
        system_prompt=load_system_prompt(),
        max_steps=settings.max_steps,
    )


def load_tickets(path: Path) -> list[Ticket]:
    with path.open(encoding="utf-8") as fh:
        return [Ticket.model_validate(item) for item in json.load(fh)]
