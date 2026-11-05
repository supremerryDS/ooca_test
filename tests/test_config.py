from __future__ import annotations

import pytest

from triage_agent.config import Settings


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    # from_env() would otherwise load the developer's real .env, API key included.
    monkeypatch.setattr("triage_agent.config.load_dotenv", lambda: None)


def test_temperature_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0")
    assert Settings.from_env().temperature == 0.0


def test_empty_temperature_means_model_default(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "")
    assert Settings.from_env().temperature is None
