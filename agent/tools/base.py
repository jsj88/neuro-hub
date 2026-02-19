"""
Base tool protocol and Stop sentinel for the Neuro-Coscientist agent.

Every tool inherits from BaseTool and implements:
    - command_name: str used to dispatch from LLM output
    - description: str injected into the system prompt
    - __call__(input_json: dict) -> str: executes the tool and returns a result string
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base for all agent tools."""

    def __init__(self, command_name: str):
        self.command_name = command_name

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description injected into the system prompt."""
        ...

    @abstractmethod
    def __call__(self, params: Dict[str, Any]) -> str:
        """Execute the tool and return a result string for the conversation."""
        ...

    def prompt_block(self) -> str:
        """Format for system prompt injection."""
        return f"TOOL: {self.command_name}\n{self.description}\n"


class Stop(BaseTool):
    """Sentinel tool that terminates the agent loop."""

    def __init__(self):
        super().__init__("STOP")

    @property
    def description(self) -> str:
        return (
            "Call STOP when the task is complete. "
            "Provide a JSON with a 'summary' key containing a brief summary of what was accomplished."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        raise StopIteration(params.get("summary", "Task completed."))
