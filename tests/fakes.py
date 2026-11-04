"""Test doubles: a scripted LLM so the agent loop can be tested without an API key."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

from triage_agent.llm.base import ChatMessage, LLMError, LLMResponse, ToolCall, ToolSpec
from triage_agent.schemas import TokenUsage

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_ids = count(1)


@dataclass
class RecordedCall:
    messages: list[ChatMessage]
    tool_names: list[str]
    force_tool: str | None


class ScriptedLLM:
    """Returns pre-defined responses in order. An `Exception` in the script is raised instead."""

    model = "scripted-fake"

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self._script = deque(script)
        self.calls: list[RecordedCall] = []

    def chat(self, messages: list[ChatMessage], tools: list[ToolSpec], force_tool: str | None = None) -> LLMResponse:
        self.calls.append(RecordedCall(list(messages), [t.name for t in tools], force_tool))
        if not self._script:
            raise LLMError("script exhausted")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item


def tool_call(name: str, args: dict[str, Any] | str) -> ToolCall:
    raw = args if isinstance(args, str) else json.dumps(args)
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=raw)


def respond(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=list(calls), usage=TokenUsage(prompt_tokens=100, completion_tokens=20))


def decision_payload(**overrides: Any) -> dict[str, Any]:
    """A valid `submit_triage` payload (low-urgency auto-response) with optional overrides."""
    payload: dict[str, Any] = {
        "urgency": "low",
        "urgency_rationale": "Calm how-to question.",
        "extracted": {
            "product_area": "macOS desktop app / Appearance",
            "issue_type": "how_to",
            "secondary_issue_types": ["bug", "feature_request"],
            "summary": "Customer asks about dark mode.",
            "key_facts": ["App version 4.1.3"],
            "customer_language": "en",
            "sentiment": "positive",
            "sentiment_trend": "stable",
            "risk_flags": [],
        },
        "kb_article_ids": ["KB-APP-001"],
        "action": "auto_respond",
        "route_to": None,
        "action_rationale": "KB fully answers it.",
        "customer_reply": "Update to 4.2 and choose Dark in Settings > Appearance.",
        "internal_note": "Answered from KB-APP-001.",
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload
