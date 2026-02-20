"""
Base skill infrastructure for the Neuro-Coscientist agent.

Skills are multi-step pipelines that chain atomic tools together.
They inherit from BaseTool so they register into the agent's tool dispatch,
but also expose run_direct(**kwargs) -> SkillResult for Python-first usage.
"""

import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..tools.base import BaseTool


@dataclass
class SkillStep:
    """Record of a single atomic step within a skill."""
    tool_name: str
    description: str
    status: str = "pending"  # pending | running | success | error
    duration_s: float = 0.0
    output: Any = None
    error: Optional[str] = None


@dataclass
class SkillResult:
    """Structured output from a skill execution."""
    skill_name: str
    success: bool = False
    results: Dict[str, Any] = field(default_factory=dict)
    figures: List[str] = field(default_factory=list)
    csv_paths: List[str] = field(default_factory=list)
    summary: str = ""
    steps: List[SkillStep] = field(default_factory=list)
    error: Optional[str] = None

    def to_tool_output(self) -> str:
        """Format for agent conversation interface."""
        lines = [f"=== {self.skill_name} ==="]
        lines.append(f"Status: {'SUCCESS' if self.success else 'FAILED'}")

        if self.error:
            lines.append(f"Error: {self.error}")

        lines.append(f"\nSteps ({len(self.steps)}):")
        for i, step in enumerate(self.steps, 1):
            status_icon = {"success": "+", "error": "!", "pending": ".", "running": "~"}.get(step.status, "?")
            lines.append(f"  [{status_icon}] {i}. {step.description} ({step.duration_s:.1f}s)")
            if step.error:
                lines.append(f"      Error: {step.error}")

        if self.summary:
            lines.append(f"\nSummary:\n{self.summary}")

        if self.figures:
            lines.append(f"\nFigures: {', '.join(self.figures)}")

        if self.csv_paths:
            lines.append(f"CSVs: {', '.join(self.csv_paths)}")

        return "\n".join(lines)


class BaseSkill(BaseTool):
    """
    Abstract base for multi-step skills.

    Inherits from BaseTool so skills auto-register into the agent's tool
    dispatch. Each skill also exposes run_direct(**kwargs) -> SkillResult
    for direct Python usage (Jupyter, CLI, tests).
    """

    def __init__(self, command_name: str):
        super().__init__(command_name)
        self._steps: List[SkillStep] = []

    @abstractmethod
    def run_direct(self, **kwargs) -> SkillResult:
        """Execute the skill pipeline and return structured results."""
        ...

    def __call__(self, params: Dict[str, Any]) -> str:
        """BaseTool interface — delegates to run_direct, returns string."""
        try:
            result = self.run_direct(**params)
            return result.to_tool_output()
        except Exception as e:
            return f"ERROR: {self.command_name} failed: {e}"

    def _run_step(
        self,
        tool_name: str,
        description: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> SkillStep:
        """Execute a single step with timing and error handling."""
        step = SkillStep(tool_name=tool_name, description=description, status="running")
        self._steps.append(step)

        t0 = time.time()
        try:
            step.output = func(*args, **kwargs)
            step.status = "success"
        except Exception as e:
            step.status = "error"
            step.error = f"{type(e).__name__}: {e}"
            step.output = None
        finally:
            step.duration_s = time.time() - t0

        return step

    def _reset_steps(self):
        """Clear step log for a fresh run."""
        self._steps = []

    def prompt_block(self) -> str:
        """Format for system prompt — prefixed SKILL: instead of TOOL:."""
        return f"SKILL: {self.command_name}\n{self.description}\n"
