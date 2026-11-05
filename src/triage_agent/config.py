"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Each field can be set with the environment variable in brackets.

    Attributes:
        model: OpenAI model name (`OPENAI_MODEL`).
        reasoning_effort: "low", "medium" or "high" for reasoning models, or None for the
            model default (`OPENAI_REASONING_EFFORT`).
        max_steps: Maximum model turns per ticket (`TRIAGE_MAX_STEPS`).
        confidence_threshold: Below this, an automatic reply goes to human review
            (`TRIAGE_CONFIDENCE_THRESHOLD`).
        request_timeout_s: Timeout for one OpenAI request, in seconds (`OPENAI_TIMEOUT_S`).
        data_dir: Folder with the mock data and knowledge base (`TRIAGE_DATA_DIR`).
    """

    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = None
    max_steps: int = 8
    confidence_threshold: float = 0.6
    request_timeout_s: float = 60.0
    data_dir: Path = PROJECT_ROOT / "data"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, loading a `.env` file first if there is one.

        Raises:
            ValueError: If a numeric variable such as `TRIAGE_MAX_STEPS` is not a number.
        """
        load_dotenv()
        defaults = cls()
        return cls(
            model=os.getenv("OPENAI_MODEL", defaults.model),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT") or defaults.reasoning_effort,
            max_steps=int(os.getenv("TRIAGE_MAX_STEPS", defaults.max_steps)),
            confidence_threshold=float(os.getenv("TRIAGE_CONFIDENCE_THRESHOLD", defaults.confidence_threshold)),
            request_timeout_s=float(os.getenv("OPENAI_TIMEOUT_S", defaults.request_timeout_s)),
            data_dir=Path(os.getenv("TRIAGE_DATA_DIR", str(defaults.data_dir))),
        )
