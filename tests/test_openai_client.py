"""Checks the OpenAI adapter's request/response mapping without network access."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from triage_agent.llm.base import ChatMessage, LLMError, ToolCall
from triage_agent.llm.openai_client import OpenAIChatClient


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response, self.error, self.kwargs, self.call_count = response, error, None, 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.call_count += 1
        if self.error:
            raise self.error
        return self.response


class RejectsTemperature:
    """Behaves like a reasoning model: any request with `temperature` is a 400."""

    def __init__(self, response) -> None:
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise bad_request(param="temperature")
        return self.response


def bad_request(param: str) -> openai.BadRequestError:
    response = SimpleNamespace(request=SimpleNamespace(method="POST", url="x"), status_code=400, headers={})
    return openai.BadRequestError("unsupported value", response=response, body={"param": param})


def fake_client(completions) -> SimpleNamespace:
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
    client = OpenAIChatClient("gpt-test", reasoning_effort="low", temperature=0, client=fake_client(completions))
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
    assert sent["temperature"] == 0
    assert sent["messages"][2]["tool_calls"][0]["function"]["name"] == "get_customer_history"
    assert sent["messages"][3] == {"role": "tool", "content": '{"plan": "free"}', "tool_call_id": "c0"}
    assert response.tool_calls == [ToolCall("call_1", "get_customer_history", '{"customer_id": "cus_1001"}')]
    assert response.usage.total_tokens == 57


def test_force_tool_sets_named_tool_choice(registry):
    completions = FakeCompletions(response=completion_with_tool_call())
    OpenAIChatClient("gpt-test", client=fake_client(completions)).chat([], registry.specs, force_tool="submit_triage")
    assert completions.kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit_triage"}}
    assert "reasoning_effort" not in completions.kwargs
    assert "temperature" not in completions.kwargs


def test_provider_errors_become_llm_errors(registry):
    completions = FakeCompletions(error=openai.APIConnectionError(request=SimpleNamespace(method="POST", url="x")))
    with pytest.raises(LLMError):
        OpenAIChatClient("gpt-test", client=fake_client(completions)).chat([], registry.specs)


def test_rejected_temperature_is_dropped_and_the_request_sent_again(registry):
    completions = RejectsTemperature(completion_with_tool_call())
    client = OpenAIChatClient("gpt-test", temperature=0, client=fake_client(completions))

    first = client.chat([], registry.specs)
    client.chat([], registry.specs)

    assert first.tool_calls[0].name == "get_customer_history"
    assert ["temperature" in call for call in completions.calls] == [True, False, False]


def test_other_bad_requests_are_not_retried(registry):
    completions = FakeCompletions(error=bad_request(param="messages"))
    with pytest.raises(LLMError):
        OpenAIChatClient("gpt-test", temperature=0, client=fake_client(completions)).chat([], registry.specs)
    assert completions.call_count == 1
