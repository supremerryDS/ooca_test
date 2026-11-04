"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = None  # e.g. "low" | "medium" for reasoning models; None = model default
    max_steps: int = 8
    confidence_threshold: float = 0.6
    request_timeout_s: float = 60.0
    data_dir: Path = PROJECT_ROOT / "data"

    @classmethod
    def from_env(cls) -> Settings:
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
