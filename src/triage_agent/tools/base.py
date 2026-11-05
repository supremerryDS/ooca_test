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
    """Base class for a tool the model can call.

    Attributes:
        name: Name the model uses to call the tool.
        description: What the tool does and when to use it, written for the model.
        args_model: Pydantic model of the arguments; the JSON schema sent to the model
            is generated from it.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, args: Any) -> dict[str, Any]:
        """Execute the tool.

        Args:
            args: An instance of `args_model`, already validated.

        Returns:
            JSON-serializable data sent back to the model.

        Raises:
            ToolError: For expected failures the model should hear about, such as an
                unknown customer ID.
        """

    @property
    def spec(self) -> ToolSpec:
        """The tool as the model sees it: name, description and argument model."""
        return ToolSpec(name=self.name, description=self.description, parameters=self.args_model)


class ToolRegistry:
    """Holds the tools by name and runs the calls the model asks for.

    Args:
        tools: The tools to register.

    Raises:
        ValueError: If two tools have the same name.
    """

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        """Names of the registered tools, in registration order."""
        return list(self._tools)

    @property
    def specs(self) -> list[ToolSpec]:
        """Definitions of every registered tool, to send to the model."""
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> ToolCallRecord:
        """Validate the arguments, run the tool and record the outcome. Never raises.

        An unknown tool, invalid arguments, a `ToolError` or a crash inside the tool all
        come back as a record with `ok=False` and an error message, so the model can
        correct itself instead of the run failing.

        Args:
            name: Tool name requested by the model.
            raw_arguments: Arguments as the raw JSON string from the model.

        Returns:
            What was called, with which arguments, the output or error, and how long it took.
        """
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
    """Serialize a tool result into the JSON string sent back to the model.

    Args:
        record: The tool call to report.

    Returns:
        The output as JSON, or `{"error": ...}` if the call failed.
    """
    payload = record.output if record.ok else {"error": record.error}
    return json.dumps(payload, ensure_ascii=False, default=str)


def _safe_json(raw: str) -> dict[str, Any] | str:
    """Parse arguments for the audit record; keep the raw string if it is not a JSON object."""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    return value if isinstance(value, dict) else raw
