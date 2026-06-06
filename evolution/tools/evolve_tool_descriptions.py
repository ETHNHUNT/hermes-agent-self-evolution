"""Main entry point for Phase 2: Tool Description Evolution.

Orchestrates the full optimization loop:
  1. Extract current tool descriptions from hermes-agent
  2. Generate tool selection evaluation dataset
  3. Run GEPA optimization to evolve descriptions
  4. Validate evolved descriptions against constraints
  5. Output results (diff + metrics + PR-ready files)

Usage:
    python -m evolution.tools.evolve_tool_descriptions --dry-run
    python -m evolution.tools.evolve_tool_descriptions --iterations 20
    python -m evolution.tools.evolve_tool_descriptions --tools read_file,write_file --iterations 10
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path, setup_dspy_lm
from evolution.core.constraints import ConstraintValidator
from evolution.tools.tool_description_module import (
    ToolDescriptionModule,
    tool_selection_metric,
)
from evolution.tools.tool_selection_evaluator import (
    SyntheticToolSelectionBuilder,
    SessionDBToolMiner,
    ToolSelectionDataset,
    ToolSelectionExample,
    ToolSelectionEvaluator,
)

console = Console()


def extract_tool_descriptions(hermes_agent_path: Path) -> dict[str, str]:
    """Extract current tool descriptions from hermes-agent source code.

    Parses tool schema files to find the 'description' field for each tool.
    Returns a dict of tool_name -> description text.
    """
    descriptions = {}
    tools_dir = hermes_agent_path / "tools"

    if not tools_dir.exists():
        console.print(f"[red]Tools directory not found: {tools_dir}[/red]")
        return descriptions

    # Key tool files to extract from
    tool_files = {
        "file_tools.py": ["read_file", "write_file", "patch", "search_files"],
        "terminal_tool.py": ["terminal"],
        "memory_tool.py": ["memory"],
        "session_search_tool.py": ["session_search"],
        "todo_tool.py": ["todo"],
        "delegate_tool.py": ["delegate_task"],
        "code_execution_tool.py": ["execute_code"],
        "web_tools.py": ["web_fetch", "web_search"],
        "browser_tool.py": ["browser_navigate", "browser_click", "browser_snapshot", "browser_scroll", "browser_type", "browser_vision", "browser_console", "browser_back", "browser_press", "browser_get_images"],
        "send_message_tool.py": ["send_message"],
        "cronjob_tools.py": ["cronjob"],
        "skill_manager_tool.py": ["skill_manage", "skill_view", "skills_list"],
        "vision_tools.py": ["vision_analyze"],
        "tts_tool.py": ["text_to_speech"],
        "image_generation_tool.py": ["image_gen"],
        "x_search_tool.py": ["x_search"],
        "process_registry.py": ["process"],
        "clarify_tool.py": ["clarify"],
        "spotify_tool.py": ["spotify"],
        "obsidian_tool.py": ["obsidian"],
        "notion_tool.py": ["notion"],
        "github_code_review_tool.py": ["github_code_review"],
        "arxiv_tool.py": ["arxiv"],
        "youtube_content_tool.py": ["youtube_content"],
        "gif_search_tool.py": ["gif_search"],
        "songsee_tool.py": ["songsee"],
        "heartmula_tool.py": ["heartmula"],
        "maps_tool.py": ["maps"],
        "linear_tool.py": ["linear"],
        "airtable_tool.py": ["airtable"],
        "google_workspace_tool.py": ["google_workspace"],
        "powerpoint_tool.py": ["powerpoint"],
        "nano_pdf_tool.py": ["nano_pdf"],
        "ocr_documents_tool.py": ["ocr_documents"],
        "hue_tool.py": ["openhue"],
        "xurl_tool.py": ["xurl"],
        "yuanbao_tool.py": ["yuanbao"],
        "feishu_doc_tool.py": ["feishu_doc"],
        "feishu_drive_tool.py": ["feishu_drive"],
        "homeassistant_tool.py": ["homeassistant"],
        "discord_tool.py": ["discord"],
        "mixture_of_agents_tool.py": ["mixture_of_agents"],
        "lazy_deps_tool.py": ["lazy_deps"],
        "transcription_tools.py": ["transcription"],
        "video_generation_tool.py": ["video_generation"],
        "computer_use_tool.py": ["computer_use"],
        "mcp_tool.py": ["mcp"],
        "kanban_tools.py": ["kanban"],
        "hermes_agent_tool.py": ["hermes_agent"],
        "plan_tool.py": ["plan"],
        "spike_tool.py": ["spike"],
        "tdd_tool.py": ["test_driven_development"],
        "debugging_tool.py": ["debugging"],
        "requesting_code_review_tool.py": ["requesting_code_review"],
        "systematic_debugging_tool.py": ["systematic_debugging"],
        "subagent_driven_development_tool.py": ["subagent_driven_development"],
        "writing_plans_tool.py": ["writing_plans"],
        "hermes_agent_skill_authoring_tool.py": ["hermes_agent_skill_authoring"],
        "debugging_hermes_tui_commands_tool.py": ["debugging_hermes_tui_commands"],
        "hermes_s6_container_supervision_tool.py": ["hermes_s6_container_supervision"],
        "node_inspect_debugger_tool.py": ["node_inspect_debugger"],
        "python_debugpy_tool.py": ["python_debugpy"],
        "dogfood_tool.py": ["dogfood"],
        "webhook_subscriptions_tool.py": ["webhook_subscriptions"],
        "kanban_orchestrator_tool.py": ["kanban_orchestrator"],
        "kanban_worker_tool.py": ["kanban_worker"],
        "claude_code_tool.py": ["claude_code"],
        "codex_tool.py": ["codex"],
        "opencode_tool.py": ["opencode"],
        "kanban_codex_lane_tool.py": ["kanban_codex_lane"],
        "huggingface_hub_tool.py": ["huggingface_hub"],
        "evaluating_llms_harness_tool.py": ["evaluating_llms_harness"],
        "weights_and_biases_tool.py": ["weights_and_biases"],
        "llama_cpp_tool.py": ["llama_cpp"],
        "obliteratus_tool.py": ["obliteratus"],
        "serving_llms_vllm_tool.py": ["serving_llms_vllm"],
        "audiocraft_audio_generation_tool.py": ["audiocraft_audio_generation"],
        "segment_anything_model_tool.py": ["segment_anything_model"],
        "dspy_tool.py": ["dspy"],
        "native_mcp_tool.py": ["native_mcp"],
        "apple_notes_tool.py": ["apple_notes"],
        "apple_reminders_tool.py": ["apple_reminders"],
        "findmy_tool.py": ["findmy"],
        "imessage_tool.py": ["imessage"],
        "macos_computer_use_tool.py": ["macos_computer_use"],
        "himalaya_tool.py": ["himalaya"],
        "minecraft_modpack_server_tool.py": ["minecraft_modpack_server"],
        "pokemon_player_tool.py": ["pokemon_player"],
        "codebase_inspection_tool.py": ["codebase_inspection"],
        "github_auth_tool.py": ["github_auth"],
        "github_issues_tool.py": ["github_issues"],
        "github_pr_workflow_tool.py": ["github_pr_workflow"],
        "github_repo_management_tool.py": ["github_repo_management"],
        "blogwatcher_tool.py": ["blogwatcher"],
        "llm_wiki_tool.py": ["llm_wiki"],
        "polymarket_tool.py": ["polymarket"],
        "godmode_tool.py": ["godmode"],
        "teams_meeting_pipeline_tool.py": ["teams_meeting_pipeline"],
    }

    for tool_file, expected_tools in tool_files.items():
        file_path = tools_dir / tool_file
        if not file_path.exists():
            continue

        content = file_path.read_text()

        # Extract description from registry.register calls
        for tool_name in expected_tools:
            # Pattern 1: description in schema dict after tool name
            # Match: "description": "..." within 2000 chars of the tool name
            name_pos = content.find(f'"{tool_name}"')
            if name_pos == -1:
                name_pos = content.find(f"'{tool_name}'")
            if name_pos == -1:
                continue

            # Look for "description" within 2000 chars after the tool name
            window = content[name_pos:name_pos + 2000]
            desc_match = re.search(r'"description"\s*:\s*"([^"]{10,600})"', window)
            if desc_match:
                desc = desc_match.group(1).strip()
                if len(desc) > 10:
                    descriptions[tool_name] = desc

    # Also extract from description constants (multi-line triple-quoted strings)
    for tool_file in tools_dir.glob("*.py"):
        if tool_file.name in ("__init__.py", "registry.py", "mcp_tool.py"):
            continue
        content = tool_file.read_text()
        # Find all DESCRIPTION constants: FOO_DESCRIPTION = """..."""
        const_pattern = r'([A-Z_]+_DESCRIPTION)\s*=\s*"""(.*?)"""'
        for match in re.finditer(const_pattern, content, re.DOTALL):
            const_name = match.group(1)
            desc_text = match.group(3).strip() if match.lastindex and match.lastindex >= 3 else match.group(2).strip()
            # Map constant name to known tool names
            const_lower = const_name.lower()
            if "terminal" in const_lower:
                if "terminal" not in descriptions or len(descriptions["terminal"]) < 100:
                    descriptions["terminal"] = desc_text[:500]
            elif "browser" in const_lower:
                for bt in ["browser_navigate", "browser_click", "browser_snapshot"]:
                    if bt not in descriptions or len(descriptions[bt]) < 50:
                        descriptions[bt] = desc_text[:500]
                        break

    # Fallback: try to get descriptions from the running tool schemas
    # Fallback: try to get descriptions from the running tool schemas for any missing tools
    try:
        sys.path.insert(0, str(hermes_agent_path))
        from tools.registry import registry, discover_builtin_tools
        discover_builtin_tools()
        for name, entry in registry._tools.items():
            if name not in descriptions:
                schema = getattr(entry, "schema", {})
                if isinstance(schema, dict):
                    desc = schema.get("description", "")
                    if desc and len(desc) > 10:
                        descriptions[name] = desc
    except Exception as e:
        console.print(f"[dim]Runtime extraction failed: {e}[/dim]")

    return descriptions


def evolve_descriptions(
    tool_descriptions: dict[str, str],
    eval_dataset: ToolSelectionDataset,
    config: EvolutionConfig,
    max_metric_calls: int = 150,
    or_api_key: str = "",
    or_base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, str]:
    """Run GEPA optimization on tool descriptions.

    Optimizes each tool description individually using GEPA's text optimization.
    For each tool, creates a DSPy program where the description is the optimizable
    parameter, then runs GEPA to find better descriptions.

    Args:
        tool_descriptions: Current tool name -> description mapping
        eval_dataset: Evaluation dataset of (task, correct_tool) pairs
        config: Evolution configuration
        max_metric_calls: Maximum LLM calls for GEPA

    Returns:
        Dict of evolved tool name -> description mapping
    """
    console.print(f"\n[bold cyan]🧬 Starting GEPA optimization[/bold cyan]")
    console.print(f"  Tools: {len(tool_descriptions)}")
    console.print(f"  Eval examples: {eval_dataset.size}")
    console.print(f"  Max metric calls: {max_metric_calls}")
    console.print(f"  Optimizer model: {config.optimizer_model}")

    # Build DSPy dataset from eval examples
    dspy_examples = eval_dataset.to_dspy_examples()

    if not dspy_examples:
        console.print("[red]No evaluation examples — cannot optimize[/red]")
        return tool_descriptions

    # Split into train and val
    n_train = max(1, int(len(dspy_examples) * 0.6))
    trainset = dspy_examples[:n_train]
    valset = dspy_examples[n_train:]

    console.print(f"  Train: {len(trainset)} | Val: {len(valset)}")

    # Create the GEPA optimizer
    # reflection_lm must be an LM object, not a string
    reflection_lm = dspy.settings.lm or dspy.LM(
        model=config.optimizer_model,
        api_key=or_api_key,
        base_url=or_base_url,
    )
    # ── Direct LLM-based description optimization ──────────────────────
    # GEPA doesn't work well for optimizing dict-structured descriptions.
    # Instead, we use a direct approach: ask an LLM to improve descriptions
    # based on failure analysis, then evaluate the improved versions.

    max_iters = max(1, max_metric_calls // 10)
    console.print(f"\n[bold cyan]Starting description optimization[/bold]")
    console.print(f"  Tools: {len(tool_descriptions)}")
    console.print(f"  Eval examples: {len(dspy_examples)}")
    console.print(f"  Max iterations: {max_iters}")

    evolved = dict(tool_descriptions)

    # Identify which tools need improvement (accuracy < 100%)
    # from the baseline evaluation
    # We'll focus on the worst-performing tools first

    # Run baseline to get per-tool accuracy
    evaluator = ToolSelectionEvaluator(config)
    baseline_results = evaluator.evaluate(tool_descriptions, eval_dataset)

    # Sort tools by accuracy (worst first)
    tool_accuracies = []
    for tool_name in tool_descriptions:
        per_tool = baseline_results.get("per_tool", {}).get(tool_name, {})
        acc = per_tool.get("correct", 0) / max(1, per_tool.get("total", 1))
        tool_accuracies.append((tool_name, acc, per_tool.get("total", 0)))

    tool_accuracies.sort(key=lambda x: (x[1], -x[2]))  # Sort by accuracy asc, then by total desc

    console.print(f"\n[bold]Tools needing improvement:[/bold]")
    for tool_name, acc, total in tool_accuracies:
        if acc < 1.0:
            console.print(f"  {tool_name}: {acc:.0%} ({total} examples)")

    # For each tool that needs improvement, ask LLM to write a better description
    # based on the failure cases
    lm = dspy.settings.lm

    class ImproveDescription(dspy.Signature):
        """Improve a tool's description to make it more distinguishable from similar tools.

        Given the current description, the tool's name, examples of tasks where this tool
        was confused with other tools, and the competing tools' descriptions, write a
        better description that makes it clearer when to use this tool vs alternatives.

        The description should:
        1. Be clear and specific about what this tool does
        2. Explicitly mention what it does NOT do (if commonly confused)
        3. Include key phrases that match natural language task descriptions
        4. Be concise (under 500 chars)
        """
        tool_name: str = dspy.InputField(desc="Name of the tool")
        current_description: str = dspy.InputField(desc="Current description of the tool")
        competing_tools: str = dspy.InputField(desc="JSON of {name: description} for commonly confused tools")
        failure_examples: str = dspy.InputField(desc="Examples of tasks where the wrong tool was selected")
        improved_description: str = dspy.OutputField(desc="Improved description (under 500 chars)")

    improver = dspy.ChainOfThought(ImproveDescription)

    max_improvement_rounds = max(1, max_metric_calls // 20)  # Each round uses ~20 LLM calls
    for round_num in range(max_improvement_rounds):
        console.print(f"\n[bold]Improvement round {round_num + 1}/{max_improvement_rounds}[/bold]")

        # Re-evaluate current descriptions
        current_results = evaluator.evaluate(evolved, eval_dataset)
        current_acc = current_results["accuracy"]
        console.print(f"  Current accuracy: {current_acc:.1%}")

        # Find tools that still need improvement
        improved_this_round = False
        for tool_name, acc, total in tool_accuracies:
            per_tool = current_results.get("per_tool", {}).get(tool_name, {})
            current_tool_acc = per_tool.get("correct", 0) / max(1, per_tool.get("total", 1))

            if current_tool_acc >= 1.0:
                continue  # Already perfect

            # Get failure examples for this tool
            failures = [
                f for f in current_results.get("failures", [])
                if f.get("expected") == tool_name
            ]

            if not failures:
                continue

            # Get competing tools (tools that were incorrectly selected instead)
            competing = {}
            for f in failures:
                wrong_tool = f.get("got", "")
                if wrong_tool in evolved and wrong_tool != tool_name:
                    competing[wrong_tool] = evolved[wrong_tool][:200]

            # Ask LLM to improve the description
            try:
                with dspy.context(lm=lm):
                    result = improver(
                        tool_name=tool_name,
                        current_description=evolved[tool_name],
                        competing_tools=str(competing),
                        failure_examples=str([f["task"][:100] for f in failures[:5]]),
                    )

                new_desc = result.improved_description.strip()
                if len(new_desc) > 10 and len(new_desc) <= 500 and new_desc != evolved[tool_name]:
                    evolved[tool_name] = new_desc
                    console.print(f"  [green]✓[/green] {tool_name}: improved")
                    improved_this_round = True
            except Exception as e:
                console.print(f"  [dim]Error improving {tool_name}: {e}[/dim]")

        if not improved_this_round:
            console.print("  No improvements made this round. Stopping.")
            break

    return evolved


@click.command()
@click.option("--iterations", default=20, help="Number of GEPA iterations")
@click.option("--max-metric-calls", default=150, help="Max LLM calls for GEPA")
@click.option("--tools", default=None, help="Comma-separated list of specific tools to optimize")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "sessiondb", "combined"]))
@click.option("--tasks-per-tool", default=3, help="Synthetic tasks per tool")
@click.option("--eval-dataset", default=None, help="Path to curated eval dataset JSON file")
@click.option("--set-baseline", is_flag=True, help="Establish baseline (eval only, no optimization)")
@click.option("--compare-baseline", is_flag=True, help="Compare against saved baseline")
@click.option("--baseline-path", default=None, help="Path to baseline JSON file")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--output-dir", default=None, help="Output directory for results")
@click.option("--optimizer-model", default="openai/minimax-m3-free", help="Model for optimizing / GEPA reflections")
@click.option("--eval-model", default="openai/minimax-m3-free", help="Model for evaluations")
@click.option("--judge-model", default="openai/minimax-m3-free", help="Model for LLM-as-judge / dataset generation")
def main(
    iterations: int,
    max_metric_calls: int,
    tools: Optional[str],
    eval_source: str,
    tasks_per_tool: int,
    eval_dataset: Optional[str],
    set_baseline: bool,
    compare_baseline: bool,
    baseline_path: Optional[str],
    dry_run: bool,
    output_dir: Optional[str],
    optimizer_model: str,
    eval_model: str,
    judge_model: str,
):
    """Evolve tool descriptions using DSPy + GEPA.

    Optimizes the natural language descriptions in tool schemas so that
    the Hermes Agent selects the right tool more reliably.
    """

    console.print(Panel(
        "[bold cyan]🧬 Hermes Agent Self-Evolution — Phase 2[/bold cyan]\n"
        "Tool Description Evolution via DSPy + GEPA",
        title="Phase 2",
        border_style="cyan",
    ))

    # ── Configuration ──────────────────────────────────────────────────
    import os

    # Configure DSPy globally using setup_dspy_lm helper
    setup_dspy_lm(eval_model)
    console.print(f"[dim]Configured DSPy with model: {eval_model}[/dim]")

    or_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    or_base_url = "https://openrouter.ai/api/v1"

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
    )

    if output_dir:
        config.output_dir = Path(output_dir)

    hermes_agent_path = config.hermes_agent_path
    console.print(f"\n[bold]Setup:[/bold]")
    console.print(f"  hermes-agent: {hermes_agent_path}")
    console.print(f"  Iterations: {iterations}")
    console.print(f"  Max metric calls: {max_metric_calls}")
    console.print(f"  Optimizer: {config.optimizer_model}")

    # ── Extract current tool descriptions ──────────────────────────────
    console.print(f"\n[bold]Extracting tool descriptions...[/bold]")
    tool_descriptions = extract_tool_descriptions(hermes_agent_path)

    if tools:
        specific_tools = [t.strip() for t in tools.split(",")]
        tool_descriptions = {
            k: v for k, v in tool_descriptions.items()
            if k in specific_tools
        }

    if not tool_descriptions:
        console.print("[red]No tool descriptions found! Check hermes-agent path.[/red]")
        sys.exit(1)

    console.print(f"  Found {len(tool_descriptions)} tool descriptions:")
    for name, desc in sorted(tool_descriptions.items()):
        console.print(f"    {name}: {desc[:60]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would optimize {len(tool_descriptions)} tool descriptions")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        return

    # ── Build evaluation dataset ───────────────────────────────────────
    dataset = ToolSelectionDataset()

    if eval_dataset:
        curated_path = Path(eval_dataset)
        if curated_path.exists():
            curated_examples = []
            try:
                # Try JSON format first
                curated_data = json.loads(curated_path.read_text())
                if not isinstance(curated_data, list):
                    curated_data = [curated_data]
            except json.JSONDecodeError:
                # Fallback to JSONL format
                curated_data = []
                with open(curated_path) as f:
                    for line in f:
                        if line.strip():
                            curated_data.append(json.loads(line))

            for item in curated_data:
                task = item.get("task_description") or item.get("task_input") or ""
                correct_tool = item.get("correct_tool", "")
                if isinstance(correct_tool, list):
                    correct_tool = correct_tool[0] if correct_tool else ""

                curated_examples.append(ToolSelectionExample(
                    task_description=task,
                    correct_tool=correct_tool,
                    difficulty=item.get("difficulty", "medium"),
                    category=item.get("category", "general"),
                    source=item.get("source", "curated"),
                ))
            dataset.extend(curated_examples)
            console.print(f"\n[bold]Loaded curated eval dataset:[/bold] {dataset.size} examples")
            # Filter to only include tools we're optimizing
            if tools:
                specific_tools = set(t.strip() for t in tools.split(","))
                dataset.examples = [ex for ex in dataset.examples if ex.correct_tool in specific_tools]
                console.print(f"  Filtered to {dataset.size} examples for selected tools")
        else:
            console.print(f"[yellow]Curated dataset not found: {curated_path}[/yellow]")
    else:
        console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

        if eval_source in ("synthetic", "combined"):
            builder = SyntheticToolSelectionBuilder(config)
            synthetic = builder.generate(tool_descriptions, num_tasks_per_tool=tasks_per_tool)
            dataset.extend(synthetic.examples)

        if eval_source in ("sessiondb", "combined"):
            miner = SessionDBToolMiner(config)
            tool_names = list(tool_descriptions.keys())

            hermes_data = miner.mine_hermes_sessions(tool_names)
            dataset.extend(hermes_data.examples)

            codex_data = miner.mine_codex_sessions(tool_names)
            dataset.extend(codex_data.examples)

    if dataset.size == 0:
        console.print("[red]No evaluation examples generated![/red]")
        sys.exit(1)

    console.print(f"  Total evaluation examples: {dataset.size}")

    # Print dataset stats
    stats = dataset.stats()
    console.print(f"  Sources: {stats['sources']}")
    console.print(f"  Difficulties: {stats['difficulties']}")

    # ── Baseline evaluation ────────────────────────────────────────────
    console.print(f"\n[bold]Running baseline evaluation...[/bold]")
    evaluator = ToolSelectionEvaluator(config)
    baseline_results = evaluator.evaluate(tool_descriptions, dataset)
    evaluator.print_results(baseline_results)

    # ── Baseline management ────────────────────────────────────────────
    default_baseline_path = config.output_dir / "tools" / "baseline.json"
    bp = Path(baseline_path) if baseline_path else default_baseline_path

    if set_baseline:
        # Save baseline: descriptions + eval dataset + metrics
        bp.parent.mkdir(parents=True, exist_ok=True)
        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "tool_descriptions": tool_descriptions,
            "eval_dataset": [ex.to_dict() for ex in dataset.examples],
            "metrics": {
                "accuracy": baseline_results["accuracy"],
                "correct": baseline_results["correct"],
                "total": baseline_results["total"],
                "per_tool": baseline_results["per_tool"],
            },
            "config": {
                "tools": list(tool_descriptions.keys()),
                "eval_source": eval_source,
                "tasks_per_tool": tasks_per_tool,
            },
        }
        bp.write_text(json.dumps(baseline_data, indent=2))
        console.print(f"\n[bold green]✅ Baseline saved to: {bp}[/bold green]")
        console.print(f"  Accuracy: {baseline_results['accuracy']:.1%}")
        console.print(f"  Examples: {dataset.size}")
        console.print(f"  Tools: {len(tool_descriptions)}")
        return

    if compare_baseline:
        # Load and compare against saved baseline
        if not bp.exists():
            console.print(f"[red]No baseline found at {bp}. Run with --set-baseline first.[/red]")
            sys.exit(1)
        baseline_data = json.loads(bp.read_text())
        baseline_acc = baseline_data["metrics"]["accuracy"]
        current_acc = baseline_results["accuracy"]
        diff = current_acc - baseline_acc
        console.print(f"\n[bold]Baseline Comparison:[/bold]")
        console.print(f"  Baseline accuracy:  {baseline_acc:.1%} (from {baseline_data['timestamp'][:19]})")
        console.print(f"  Current accuracy:   {current_acc:.1%}")
        console.print(f"  Difference:         {diff:+.1%}")
        if diff > 0:
            console.print(f"  [bold green]✅ Improved![/bold green]")
        elif diff < 0:
            console.print(f"  [bold red]⚠️  Regressed![/bold red]")
        else:
            console.print(f"  [dim]No change[/dim]")
        return

    # ── Run GEPA optimization ──────────────────────────────────────────
    evolved_descriptions = evolve_descriptions(
        tool_descriptions,
        dataset,
        config,
        max_metric_calls=max_metric_calls,
        or_api_key=or_api_key,
        or_base_url=or_base_url,
    )

    # ── Evaluate evolved descriptions ──────────────────────────────────
    console.print(f"\n[bold]Evaluating evolved descriptions...[/bold]")

    # Validate constraints
    validator = ConstraintValidator(config)
    console.print(f"\n[bold]Constraint validation:[/bold]")
    all_passed = True
    for name, desc in evolved_descriptions.items():
        results = validator.validate_all(desc, "tool_description")
        for r in results:
            icon = "✅" if r.passed else "❌"
            console.print(f"  {icon} {name}: {r.message}")
            if not r.passed:
                all_passed = False

        # Factual accuracy: check description still mentions the tool's purpose
        tool_keywords = set(name.replace("_", " ").lower().split())
        desc_lower = desc.lower()
        if not any(kw in desc_lower for kw in tool_keywords if len(kw) > 2):
            console.print(f"  ⚠️  {name}: Description may not reflect tool purpose")
            all_passed = False

    if not all_passed:
        console.print("\n[yellow]WARNING: Some constraints failed. Review evolved descriptions.[/yellow]")

    evolved_results = evaluator.evaluate(evolved_descriptions, dataset)
    evaluator.print_results(evolved_results)

    # ── Output results ─────────────────────────────────────────────────
    baseline_acc = baseline_results["accuracy"]
    evolved_acc = evolved_results["accuracy"]
    improvement = evolved_acc - baseline_acc

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Baseline accuracy: {baseline_acc:.1%}")
    console.print(f"  Evolved accuracy:  {evolved_acc:.1%}")
    console.print(f"  Improvement:       {improvement:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = config.output_dir / "tools" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save descriptions
    baseline_path = results_dir / "baseline_descriptions.json"
    evolved_path = results_dir / "evolved_descriptions.json"
    baseline_path.write_text(json.dumps(tool_descriptions, indent=2))
    evolved_path.write_text(json.dumps(evolved_descriptions, indent=2))

    # Save diff
    diff_path = results_dir / "description_diff.txt"
    diff_lines = []
    for name in sorted(tool_descriptions.keys()):
        old = tool_descriptions.get(name, "")
        new = evolved_descriptions.get(name, "")
        if old != new:
            diff_lines.append(f"=== {name} ===")
            diff_lines.append(f"--- BEFORE ({len(old)} chars) ---")
            diff_lines.append(old)
            diff_lines.append(f"+++ AFTER ({len(new)} chars) +++")
            diff_lines.append(new)
            diff_lines.append("")
    diff_path.write_text("\n".join(diff_lines))

    # Save metrics
    metrics = {
        "timestamp": timestamp,
        "baseline_accuracy": baseline_acc,
        "evolved_accuracy": evolved_acc,
        "improvement": improvement,
        "num_tools": len(tool_descriptions),
        "eval_examples": dataset.size,
        "iterations": iterations,
        "max_metric_calls": max_metric_calls,
        "per_tool_baseline": baseline_results["per_tool"],
        "per_tool_evolved": evolved_results["per_tool"],
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Save eval dataset
    dataset.save(results_dir / "eval_dataset.jsonl")

    console.print(f"\n[bold green]Results saved to: {results_dir}[/bold green]")
    console.print(f"  Descriptions: {evolved_path}")
    console.print(f"  Diff:         {diff_path}")
    console.print(f"  Metrics:      {metrics_path}")

    if improvement > 0:
        console.print(f"\n[bold green]✅ Improvement achieved! Ready for human review.[/bold green]")
    elif improvement == 0:
        console.print(f"\n[yellow]No improvement. Consider more iterations or better eval data.[/yellow]")
    else:
        console.print(f"\n[red]⚠️  Accuracy decreased. Do not apply evolved descriptions.[/red]")


if __name__ == "__main__":
    main()
