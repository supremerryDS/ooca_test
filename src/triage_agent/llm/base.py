"""Provider-agnostic LLM interface.

The agent only talks to `LLMClient`. Today the implementation is OpenAI
(`OpenAIChatClient`); tests use a scripted fake. Swapping providers means writing
one new adapter, not touching the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel

from triage_agent.schemas import TokenUsage

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the LLM sees it: name, description and a Pydantic model for its arguments."""

    name: str
    description: str
    parameters: type[BaseModel]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, validated later by the tool's Pydantic model


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: list[ToolCall]) -> ChatMessage:
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None


class LLMError(RuntimeError):
    """Raised when the LLM provider fails after its own retries."""


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        force_tool: str | None = None,
    ) -> LLMResponse:
        """Run one completion.

        The model must answer with tool calls. If `force_tool` is given, it must call
        exactly that tool (used to force a final answer on the last step).
        """
        ...
