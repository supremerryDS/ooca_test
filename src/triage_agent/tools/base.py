"""Tool abstraction + registry.

A tool is: a name, a description, a Pydantic model for its arguments, and a `run`
method. The registry turns raw LLM tool calls into validated arguments, runs the
tool, and *never* lets an exception escape: errors are returned to the LLM as
data so it can correct itself (e.g. fix a bad argument) instead of crashing the run.

Adding a tool = subclass `Tool` and register it in `bootstrap.build_tools`.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from triage_agent.llm.base import ToolSpec
from triage_agent.schemas import ToolCallRecord

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Expected, user-facing tool failure (e.g. 'customer not found')."""


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, args: Any) -> dict[str, Any]:
        """Execute the tool with already-validated arguments and return JSON-able data."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.args_model)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> ToolCallRecord:
        started = time.perf_counter()
        parsed_args: dict[str, Any] | str = _safe_json(raw_arguments)

        def record(ok: bool, output: Any = None, error: str | None = None) -> ToolCallRecord:
            return ToolCallRecord(
                tool_name=name,
                arguments=parsed_args,
                ok=ok,
                output=output,
                error=error,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        tool = self._tools.get(name)
        if tool is None:
            return record(False, error=f"Unknown tool '{name}'. Available: {', '.join(self.names)}")

        try:
            args = tool.args_model.model_validate_json(raw_arguments or "{}")
        except ValidationError as exc:
            return record(False, error=f"Invalid arguments: {exc.errors(include_url=False)}")

        try:
            return record(True, output=tool.run(args))
        except ToolError as exc:
            return record(False, error=str(exc))
        except Exception:  # noqa: BLE001 - a broken tool must not kill the whole triage run
            logger.exception("Tool %s crashed", name)
            return record(False, error="Internal tool error. Continue with the information you have.")


def tool_output_for_llm(record: ToolCallRecord) -> str:
    """Serialize a tool result into the string we send back to the model."""
    payload = record.output if record.ok else {"error": record.error}
    return json.dumps(payload, ensure_ascii=False, default=str)


def _safe_json(raw: str) -> dict[str, Any] | str:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    return value if isinstance(value, dict) else raw
