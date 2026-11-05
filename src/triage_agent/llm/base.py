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
    """One tool call requested by the model.

    Attributes:
        id: Provider-assigned ID; the tool result must be sent back with the same ID.
        name: Name of the tool to run.
        arguments: Raw JSON string, validated later by the tool's Pydantic model.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatMessage:
    """One message in the conversation with the model, independent of any provider.

    Build messages with the `system`, `user`, `assistant` and `tool_result` constructors.
    """

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
    """What the model returned for one completion.

    Attributes:
        content: Free text, if any (normally empty because the model must call tools).
        tool_calls: Tool calls to run, in the order the model asked for them.
        usage: Tokens used by this completion, when the provider reports it.
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None


class LLMError(RuntimeError):
    """Raised when the LLM provider fails after its own retries."""


class LLMClient(Protocol):
    """Anything that can run one tool-calling completion. Implemented by `OpenAIChatClient`.

    Attributes:
        model: Name of the model, recorded in every triage result.
    """

    model: str

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        force_tool: str | None = None,
    ) -> LLMResponse:
        """Run one completion in which the model must answer with tool calls.

        Args:
            messages: The conversation so far, oldest first.
            tools: Tools the model may call.
            force_tool: If given, the model must call exactly this tool (used to force a
                final answer on the last step).

        Returns:
            The model's tool calls, any free text and token usage.

        Raises:
            LLMError: If the provider fails after its own retries.
        """
        ...
