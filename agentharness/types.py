"""Core data types used across the harness."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    """Outcome of a single loop iteration."""

    TOOL_CALLED = "tool_called"
    FINAL_ANSWER = "final_answer"
    TOOL_FAILED = "tool_failed"
    STOPPED = "stopped"  # forced stop (max steps, timeout, etc.)


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class ToolResult:
    """The outcome of executing a ToolCall."""

    call_id: str
    success: bool
    output: Any = None
    error: str | None = None
    attempts: int = 1


@dataclass
class StepRecord:
    """One iteration of the agent loop, kept for logging/debugging."""

    step_number: int
    status: StepStatus
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    model_text: str | None = None


@dataclass
class RunResult:
    """Final output of a full agent run."""

    final_answer: str | None
    steps: list[StepRecord] = field(default_factory=list)
    stopped_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
