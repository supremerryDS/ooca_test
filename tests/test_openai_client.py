"""Checks the OpenAI adapter's request/response mapping without network access."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from triage_agent.llm.base import ChatMessage, LLMError, ToolCall
from triage_agent.llm.openai_client import OpenAIChatClient


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def fake_client(completions: FakeCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def completion_with_tool_call() -> SimpleNamespace:
    call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="get_customer_history", arguments='{"customer_id": "cus_1001"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=7),
    )


def test_request_and_response_mapping(registry):
    completions = FakeCompletions(response=completion_with_tool_call())
    client = OpenAIChatClient("gpt-test", reasoning_effort="low", client=fake_client(completions))
    history = [
        ChatMessage.system("sys"),
        ChatMessage.user("ticket"),
        ChatMessage.assistant(None, [ToolCall(id="c0", name="get_customer_history", arguments="{}")]),
        ChatMessage.tool_result("c0", '{"plan": "free"}'),
    ]

    response = client.chat(history, registry.specs)

    sent = completions.kwargs
    assert sent["model"] == "gpt-test"
    assert sent["tool_choice"] == "required"
    assert sent["reasoning_effort"] == "low"
    assert sent["messages"][2]["tool_calls"][0]["function"]["name"] == "get_customer_history"
    assert sent["messages"][3] == {"role": "tool", "content": '{"plan": "free"}', "tool_call_id": "c0"}
    assert response.tool_calls == [ToolCall("call_1", "get_customer_history", '{"customer_id": "cus_1001"}')]
    assert response.usage.total_tokens == 57


def test_force_tool_sets_named_tool_choice(registry):
    completions = FakeCompletions(response=completion_with_tool_call())
    OpenAIChatClient("gpt-test", client=fake_client(completions)).chat([], registry.specs, force_tool="submit_triage")
    assert completions.kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit_triage"}}
    assert "reasoning_effort" not in completions.kwargs


def test_provider_errors_become_llm_errors(registry):
    completions = FakeCompletions(error=openai.APIConnectionError(request=SimpleNamespace(method="POST", url="x")))
    with pytest.raises(LLMError):
        OpenAIChatClient("gpt-test", client=fake_client(completions)).chat([], registry.specs)
