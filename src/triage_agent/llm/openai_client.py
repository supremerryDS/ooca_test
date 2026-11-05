"""OpenAI adapter for `LLMClient` (Chat Completions + strict function calling)."""

from __future__ import annotations

import logging
from typing import Any

import openai
from openai import OpenAI

from triage_agent.llm.base import ChatMessage, LLMError, LLMResponse, ToolCall, ToolSpec
from triage_agent.schemas import TokenUsage

logger = logging.getLogger(__name__)


class OpenAIChatClient:
    """`LLMClient` for OpenAI GPT models (Chat Completions with strict function calling).

    The API key is read from `OPENAI_API_KEY` by the SDK. The SDK also retries
    timeouts, rate limits and server errors with exponential backoff.

    Args:
        model: OpenAI model name.
        reasoning_effort: "low", "medium" or "high" for reasoning models, or None to
            leave the model default.
        timeout_s: Timeout for one request, in seconds.
        max_retries: How many times the SDK retries a failed request.
        client: A ready-made OpenAI client to use instead, for example a fake one in tests.
    """

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self._reasoning_effort = reasoning_effort
        # The SDK retries 408/409/429/5xx with exponential backoff.
        self._client = client or OpenAI(timeout=timeout_s, max_retries=max_retries)

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        force_tool: str | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in messages],
            "tools": [to_openai_tool(t) for t in tools],
            "tool_choice": ({"type": "function", "function": {"name": force_tool}} if force_tool else "required"),
        }
        if self._reasoning_effort:
            request["reasoning_effort"] = self._reasoning_effort

        try:
            completion = self._client.chat.completions.create(**request)
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        message = completion.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (message.tool_calls or [])
            if tc.type == "function"
        ]
        usage = (
            TokenUsage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
            if completion.usage
            else None
        )
        return LLMResponse(content=message.content, tool_calls=tool_calls, usage=usage)


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    """Convert a tool to OpenAI's function-tool format with a strict JSON schema.

    Strict means every field is required and no extra fields are allowed, so the
    model's arguments always match our Pydantic models.

    Args:
        spec: The tool to convert.

    Returns:
        The tool definition sent to OpenAI (what `triage-agent tools` prints).
    """
    return openai.pydantic_function_tool(spec.parameters, name=spec.name, description=spec.description)


def _to_openai_message(message: ChatMessage) -> dict[str, Any]:
    """Convert a provider-independent message to the Chat Completions format."""
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload
