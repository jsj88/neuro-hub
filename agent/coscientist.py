"""
Neuro-Coscientist: Autonomous AI agent for neuroscience and behavioral data.

Inspired by Boiko et al. (2023) "Autonomous chemical research with large language models"
(Nature, doi:10.1038/s41586-023-06792-0 / github.com/gomesgroup/coscientist).

Adapted for:
  - Reward learning model simulation and fitting (RW, CK, VPP, ORL)
  - EEG temporal decoding (TMaze REWP analysis)
  - Neuro-model correlation (RPE ↔ EEG amplitude)
  - Parameter recovery and model comparison
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from .tools.base import BaseTool, Stop
from .tools.simulate import SimulateBehavior, SimulateNeural
from .tools.model_fitting import FitModel, ParameterRecovery, CompareModels
from .tools.neural import RunTemporalDecoding, RunREWP
from .tools.correlate import CorrelateNeuroModel
from .tools.plot import PlotAndSave
from .skills import BehavioralPipeline, NeuralPipeline, NeuroModelFusion, FullPipeline
from .prompts.system_prompt import build_system_prompt


class NeuroCoscientist:
    """
    Autonomous agent that orchestrates neuroscience analysis tools via an LLM.

    Architecture (per Coscientist):
    1. System prompt lists all tools + domain context
    2. User task injected into conversation
    3. LLM responds with reasoning + exactly one TOOL_NAME: {json}
    4. Agent dispatches to the matching tool
    5. Tool output appended to conversation history
    6. Repeat until STOP or max_steps
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_steps: int = 15,
        verbose: bool = True,
        output_dir: str = "./results",
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self.max_steps = max_steps
        self.verbose = verbose
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize tools
        self.tools: Dict[str, BaseTool] = {}
        self._shared_state: Dict[str, Any] = {}  # shared memory across tools

        self._register_default_tools()

        # Conversation history
        self.messages: List[Dict[str, str]] = []

    def _register_default_tools(self):
        """Register all available tools."""
        tool_instances = [
            # Atomic tools
            SimulateBehavior(),
            SimulateNeural(),
            FitModel(),
            ParameterRecovery(),
            CompareModels(),
            RunTemporalDecoding(),
            RunREWP(),
            CorrelateNeuroModel(),
            PlotAndSave(),
            # Skills (multi-step pipelines)
            BehavioralPipeline(),
            NeuralPipeline(),
            NeuroModelFusion(),
            FullPipeline(),
            # Sentinel
            Stop(),
        ]
        for tool in tool_instances:
            self.tools[tool.command_name] = tool

    def register_tool(self, tool: BaseTool):
        """Register a custom tool."""
        self.tools[tool.command_name] = tool

    def _build_system_prompt(self) -> str:
        """Build system prompt with all tool descriptions."""
        return build_system_prompt(self.tools)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM and return the response text."""
        if self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, messages: List[Dict[str, str]]) -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=self.api_key)

        # Extract system message
        system = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=chat_messages,
        )
        return response.content[0].text

    def _parse_tool_call(self, response: str) -> Optional[tuple]:
        """
        Parse a tool call from the LLM response.

        Expected format:
            TOOL_NAME: {"key": "value", ...}
        """
        # Try to find TOOL_NAME: {json} pattern
        for tool_name in self.tools:
            pattern = rf"{tool_name}\s*:\s*(\{{.*?\}})"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    params = json.loads(match.group(1))
                    return tool_name, params
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    raw = match.group(1)
                    raw = raw.replace("'", '"')
                    try:
                        params = json.loads(raw)
                        return tool_name, params
                    except json.JSONDecodeError:
                        continue

        # Check for bare STOP
        if "STOP" in response:
            return "STOP", {"summary": "Task completed."}

        return None

    def _dispatch_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Dispatch to the named tool, injecting shared state."""
        tool = self.tools[tool_name]

        # Inject shared state for atomic tools that need cross-tool data
        if tool_name in ("RUN_TEMPORAL_DECODING", "RUN_REWP", "CORRELATE_NEURO_MODEL"):
            if "_simulated_data" not in params and "simulated_eeg" in self._shared_state:
                params["_simulated_data"] = self._shared_state["simulated_eeg"]

        # Inject shared state for skills that need data from prior steps
        if tool_name == "NEURAL_PIPELINE":
            if "rpes" not in params and "last_rpes" in self._shared_state:
                params["rpes"] = self._shared_state["last_rpes"]

        if tool_name == "NEURO_MODEL_FUSION":
            if "simulated_eeg" not in params and "simulated_eeg" in self._shared_state:
                params["simulated_eeg"] = self._shared_state["simulated_eeg"]
            if "rpes" not in params and "last_rpes" in self._shared_state:
                params["rpes"] = self._shared_state["last_rpes"]

        result = tool(params)

        # Capture shared state from simulation tools
        if tool_name == "SIMULATE_NEURAL":
            self._shared_state["simulated_eeg"] = tool._last_data

        if tool_name == "SIMULATE_BEHAVIOR":
            self._shared_state["last_behavior_csv"] = result

        # Capture shared state from skills
        if tool_name == "BEHAVIORAL_PIPELINE" and hasattr(tool, '_steps'):
            # Parse RPEs from the skill's last run (stored in run_direct result)
            # The string output contains the SkillResult; we also store via tool internals
            pass

        if tool_name == "NEURAL_PIPELINE" and hasattr(tool, '_sim_tool'):
            if hasattr(tool._sim_tool, '_last_data') and tool._sim_tool._last_data is not None:
                self._shared_state["simulated_eeg"] = tool._sim_tool._last_data

        return result

    def run(self, task: str) -> str:
        """
        Run the agent on a task.

        Parameters
        ----------
        task : str
            Natural-language task description

        Returns
        -------
        str
            Final summary of what was accomplished
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🧠 NEURO-COSCIENTIST — Starting task")
            print(f"{'='*70}")
            print(f"Task: {task}\n")

        # Initialize conversation
        system_prompt = self._build_system_prompt()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TASK: {task}"},
        ]

        summary = ""

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step}/{self.max_steps} ---")

            # Call LLM
            try:
                response = self._call_llm(self.messages)
            except Exception as e:
                if self.verbose:
                    print(f"LLM error: {e}")
                break

            self.messages.append({"role": "assistant", "content": response})

            if self.verbose:
                # Print reasoning (truncated)
                preview = response[:500] + ("..." if len(response) > 500 else "")
                print(f"🤖 Agent:\n{preview}\n")

            # Parse tool call
            parsed = self._parse_tool_call(response)

            if parsed is None:
                if self.verbose:
                    print("⚠️  No tool call detected. Asking agent to use a tool.")
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Please use exactly one tool. Format: TOOL_NAME: {\"key\": \"value\"}\n"
                        f"Available tools: {', '.join(self.tools.keys())}"
                    ),
                })
                continue

            tool_name, tool_params = parsed

            if self.verbose:
                print(f"🔧 Tool: {tool_name}")
                print(f"   Params: {json.dumps(tool_params, indent=2)[:300]}")

            # Dispatch
            try:
                result = self._dispatch_tool(tool_name, tool_params)
            except StopIteration as e:
                summary = str(e)
                if self.verbose:
                    print(f"\n✅ STOP — {summary}")
                break
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
                if self.verbose:
                    print(f"❌ Tool error: {result}")

            if self.verbose:
                print(f"📊 Result:\n{result[:500]}\n")

            # Append result to conversation
            self.messages.append({
                "role": "user",
                "content": f"TOOL RESULT ({tool_name}):\n{result}",
            })
        else:
            summary = "Max steps reached."
            if self.verbose:
                print(f"\n⚠️ Max steps ({self.max_steps}) reached.")

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🏁 Task complete. Steps used: {step}")
            print(f"{'='*70}\n")

        return summary

    def run_offline(self, task: str) -> str:
        """
        Run the agent WITHOUT an LLM — executes a deterministic script.
        Useful for testing tools directly.
        """
        if self.verbose:
            print("Running in OFFLINE mode (no LLM, sequential tool execution)\n")

        results = []

        # Parse task for keywords and run corresponding tools
        task_lower = task.lower()

        if "simulate" in task_lower and "behav" in task_lower:
            r = self._dispatch_tool("SIMULATE_BEHAVIOR", {
                "model": "rw", "params": {"alpha": 0.3, "beta": 5.0},
                "n_trials": 200, "n_subjects": 1, "output_dir": self.output_dir,
            })
            results.append(f"[SIMULATE_BEHAVIOR] {r}")

        if "simulate" in task_lower and ("neural" in task_lower or "eeg" in task_lower):
            r = self._dispatch_tool("SIMULATE_NEURAL", {
                "n_epochs": 200, "rewp_amplitude": 3.0,
            })
            results.append(f"[SIMULATE_NEURAL] {r}")

        if "fit" in task_lower:
            # Need a CSV path
            csv_path = os.path.join(self.output_dir, "sim_rw_1subj.csv")
            if os.path.exists(csv_path):
                r = self._dispatch_tool("FIT_MODEL", {
                    "model": "rw", "data_path": csv_path,
                })
                results.append(f"[FIT_MODEL] {r}")

        if "compare" in task_lower:
            csv_path = os.path.join(self.output_dir, "sim_rw_1subj.csv")
            if os.path.exists(csv_path):
                r = self._dispatch_tool("COMPARE_MODELS", {
                    "models": ["rw", "ck", "rwck"], "data_path": csv_path,
                })
                results.append(f"[COMPARE_MODELS] {r}")

        if "decode" in task_lower or "decod" in task_lower:
            if "simulated_eeg" in self._shared_state:
                r = self._dispatch_tool("RUN_TEMPORAL_DECODING", {"source": "simulated"})
                results.append(f"[RUN_TEMPORAL_DECODING] {r}")

        return "\n\n".join(results) if results else "No matching tools for the task."
