"""
Neuro-Coscientist CLI entry point.

Usage:
    # With LLM (requires OPENAI_API_KEY or ANTHROPIC_API_KEY):
    python run.py --task "Simulate RW model and run temporal decoding"
    python run.py --preset full_pipeline_simulated
    python run.py --provider anthropic --model claude-sonnet-4-20250514

    # Offline mode (no LLM, tests tools directly):
    python run.py --offline --task "simulate behavioral and neural data, fit and compare"

    # Direct skill execution (no LLM):
    python run.py --skill full --output-dir ./results/test
    python run.py --skill behavioral --skill-config config.json
"""

import argparse
import json
import os
import sys

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.coscientist import NeuroCoscientist
from agent.tasks.example_tasks import TASKS

SKILL_CHOICES = ["behavioral", "neural", "fusion", "full"]


def _run_skill(skill_name: str, config: dict, output_dir: str):
    """Execute a skill directly without LLM."""
    from agent.skills import BehavioralPipeline, NeuralPipeline, NeuroModelFusion, FullPipeline

    skill_map = {
        "behavioral": BehavioralPipeline,
        "neural": NeuralPipeline,
        "fusion": NeuroModelFusion,
        "full": FullPipeline,
    }

    skill_cls = skill_map[skill_name]
    skill = skill_cls()

    config.setdefault("output_dir", output_dir)

    print(f"\nRunning skill: {skill.command_name}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}\n")

    result = skill.run_direct(**config)

    print(result.to_tool_output())

    if result.figures:
        print(f"\nFigures saved:")
        for fig in result.figures:
            print(f"  {fig}")

    if result.csv_paths:
        print(f"\nCSVs saved:")
        for csv in result.csv_paths:
            print(f"  {csv}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Coscientist: Autonomous AI agent for neuroscience",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --preset simulate_rw
  python run.py --preset full_pipeline_simulated
  python run.py --task "Run parameter recovery for RW model with alpha=0.5, beta=3"
  python run.py --offline --task "simulate behavior and neural data, decode and compare"
  python run.py --skill full --output-dir ./results/test
  python run.py --skill behavioral --skill-config '{"generating_model": "rwck", "n_trials": 300}'
  python run.py --list-presets

Available presets: %(presets)s
        """ % {"presets": ", ".join(TASKS.keys())},
    )

    parser.add_argument(
        "--task", type=str, default=None,
        help="Natural-language task for the agent",
    )
    parser.add_argument(
        "--preset", type=str, default=None, choices=list(TASKS.keys()),
        help="Run a pre-defined task",
    )
    parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model name (default: gpt-4o for openai, claude-sonnet-4-20250514 for anthropic)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=15,
        help="Maximum agent steps (default: 15)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory for results and figures",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run in offline mode (no LLM, tests tools directly)",
    )
    parser.add_argument(
        "--skill", type=str, default=None, choices=SKILL_CHOICES,
        help="Run a skill directly without LLM (behavioral, neural, fusion, full)",
    )
    parser.add_argument(
        "--skill-config", type=str, default=None,
        help="JSON config for --skill: inline JSON string or path to .json file",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="List all available preset tasks and exit",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\nAvailable Preset Tasks:\n")
        for name, desc in TASKS.items():
            print(f"  {name}:")
            for line in desc.strip().split("\n"):
                print(f"    {line}")
            print()
        print("Available Skills (--skill):")
        for s in SKILL_CHOICES:
            print(f"  {s}")
        return

    # Direct skill execution
    if args.skill:
        config = {}
        if args.skill_config:
            if os.path.isfile(args.skill_config):
                with open(args.skill_config) as f:
                    config = json.load(f)
            else:
                config = json.loads(args.skill_config)

        _run_skill(args.skill, config, args.output_dir)
        return

    # Determine task
    if args.preset:
        task = TASKS[args.preset]
    elif args.task:
        task = args.task
    else:
        parser.error("Provide --task, --preset, or --skill (or use --list-presets)")

    # Determine model
    if args.model:
        model = args.model
    else:
        model = "gpt-4o" if args.provider == "openai" else "claude-sonnet-4-20250514"

    # Create agent
    agent = NeuroCoscientist(
        provider=args.provider,
        model=model,
        max_steps=args.max_steps,
        verbose=not args.quiet,
        output_dir=args.output_dir,
    )

    # Run
    if args.offline:
        result = agent.run_offline(task)
    else:
        result = agent.run(task)

    print(f"\nFinal Result:\n{result}")


if __name__ == "__main__":
    main()
