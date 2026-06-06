"""Main entry point for Phase 3: System Prompt Evolution.

Optimizes individual prompt constants (like MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE)
in hermes-agent/agent/prompt_builder.py using GEPA or failure-driven LLM optimization.

Usage:
    python -m evolution.prompts.evolve_prompt_section --section MEMORY_GUIDANCE --dry-run
    python -m evolution.prompts.evolve_prompt_section --section SESSION_SEARCH_GUIDANCE --iterations 5
"""

import json
import sys
import re
import os
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, setup_dspy_lm
from evolution.core.constraints import ConstraintValidator
from evolution.prompts.prompt_section_module import (
    PromptSectionModule,
    prompt_behavior_metric,
)
from evolution.prompts.prompt_selection_evaluator import (
    PromptBehaviorExample,
    PromptBehaviorDataset,
    SyntheticPromptScenarioBuilder,
    SessionDBPromptMiner,
    PromptBehaviorEvaluator,
)

console = Console()


def extract_prompt_section(hermes_agent_path: Path, section_name: str) -> str:
    """Extract a system prompt section value from prompt_builder.py."""
    file_path = hermes_agent_path / "agent" / "prompt_builder.py"
    if not file_path.exists():
        console.print(f"[red]prompt_builder.py not found: {file_path}[/red]")
        sys.exit(1)

    content = file_path.read_text(encoding="utf-8")

    # Import the module to get the evaluated Python string constant
    try:
        sys.path.insert(0, str(hermes_agent_path))
        import agent.prompt_builder
        import importlib
        importlib.reload(agent.prompt_builder)
        val = getattr(agent.prompt_builder, section_name, None)
        if isinstance(val, str):
            return val
    except Exception as e:
        console.print(f"[dim]Runtime import of prompt_builder failed: {e}[/dim]")

    # Fallback to regex parser
    pattern = rf"{section_name}\s*=\s*\(([\s\S]*?)\n\s*\)"
    match = re.search(pattern, content)
    if match:
        # Clean up lines and join them
        lines = match.group(1).strip().split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                cleaned_lines.append(line[1:-1])
            elif line.startswith("'") and line.endswith("'"):
                cleaned_lines.append(line[1:-1])
        return "".join(cleaned_lines)

    raise ValueError(f"Could not extract system prompt section {section_name} from prompt_builder.py")


def patch_prompt_builder(hermes_agent_path: Path, section_name: str, new_text: str) -> None:
    """Safely patch prompt_builder.py by replacing the target constant declaration."""
    file_path = hermes_agent_path / "agent" / "prompt_builder.py"
    if not file_path.exists():
        raise FileNotFoundError(f"prompt_builder.py not found at {file_path}")

    content = file_path.read_text(encoding="utf-8")

    # Format text into parenthesized double-quoted strings (similar to prompt_builder.py styling)
    # Break long text at word boundaries around 74 characters (preserving spacing at the end)
    paragraphs = new_text.split("\n")
    formatted_lines = []
    
    for i, para in enumerate(paragraphs):
        # We append a literal "\n" to represent line breaks in MEMORY_GUIDANCE,
        # unless it is the last paragraph.
        suffix = "\\n" if i < len(paragraphs) - 1 else ""
        wrapped = textwrap.wrap(para, width=74, break_long_words=False)
        
        for j, line in enumerate(wrapped):
            # Escape double quotes
            line_esc = line.replace('"', '\\"')
            # Append trailing space for implicit concatenation (except last line or suffix)
            line_suffix = " " if j < len(wrapped) - 1 or suffix else ""
            formatted_lines.append(f'    "{line_esc}{line_suffix}{suffix}"')

    # Find the parenthesis start and end
    pattern = rf"({section_name}\s*=\s*\()([\s\S]*?)(\n\s*\))"
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Could not find parenthesis-wrapped constant {section_name} in prompt_builder.py")

    start_idx = match.start(2)
    end_idx = match.end(2)

    new_content = content[:start_idx] + "\n" + "\n".join(formatted_lines) + content[end_idx:]
    file_path.write_text(new_content, encoding="utf-8")


def write_evolved_prompts(
    hermes_agent_path: Path,
    section_name: str,
    new_text: str,
    branch_name: Optional[str] = None,
) -> str:
    """Patch the prompt builder file and commit changes on a dedicated git branch."""
    if branch_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"evolve/system-prompt-{section_name.lower()}-{ts}"

    # 1. Create a git branch
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=hermes_agent_path,
        capture_output=True,
    )

    # 2. Patch prompt_builder.py
    patch_prompt_builder(hermes_agent_path, section_name, new_text)

    # 3. Git commit
    subprocess.run(["git", "add", "agent/prompt_builder.py"], cwd=hermes_agent_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"evolve: update system prompt {section_name} ({branch_name})"],
        cwd=hermes_agent_path,
        capture_output=True,
    )

    return branch_name


def evolve_direct(
    section_name: str,
    original_text: str,
    dataset: PromptBehaviorDataset,
    config: EvolutionConfig,
    iterations: int,
) -> str:
    """Run failure-driven direct LLM prompt optimization."""
    evaluator = PromptBehaviorEvaluator(config)
    
    # Split examples
    splits = dataset.split(config.train_ratio, config.val_ratio)
    train_ex = splits["train"]
    val_ex = splits["val"]

    console.print(f"\n[bold cyan]🧬 Starting direct failure-driven prompt optimization[/bold cyan]")
    console.print(f"  Train set: {len(train_ex)} examples | Val set: {len(val_ex)} examples")

    class ImproveGuideline(dspy.Signature):
        """Improve a system prompt section's guidelines to prevent agent failures.

        Given the target section name, the current guideline text, and examples of tasks where the agent behaved incorrectly (failures),
        write a better guideline that clarifies the rules and corrects the behavior.

        The guideline should:
        1. Be concise, clear, and direct.
        2. Explicitly address the failures shown.
        3. Avoid adding unnecessary verbosity (stay within the character/length limit).
        4. Not contradict existing instructions.
        """
        section_name: str = dspy.InputField(desc="The system prompt section name")
        current_guideline: str = dspy.InputField(desc="The current guideline text")
        failure_examples: str = dspy.InputField(desc="JSON list of failures containing task context, incorrect tool/actions, and rubrics")
        improved_guideline: str = dspy.OutputField(desc="The improved system prompt section text")

    improver = dspy.ChainOfThought(ImproveGuideline)
    lm = dspy.settings.lm

    best_guideline = original_text
    
    # Baseline eval
    baseline_results = evaluator.evaluate(best_guideline, train_ex)
    best_acc = baseline_results["accuracy"]
    best_score = baseline_results["avg_score"]
    
    console.print(f"  Baseline Accuracy: {best_acc:.1%} | Avg Score: {best_score:.3f}")

    for round_num in range(iterations):
        console.print(f"\n[bold]Round {round_num + 1}/{iterations}[/bold]")
        
        # Evaluate to get failures
        current_results = evaluator.evaluate(best_guideline, train_ex)
        failures = current_results["failures"]
        
        if not failures:
            console.print("  No failures remaining in train set!")
            break

        # Ask LLM to improve guideline
        try:
            with dspy.context(lm=lm):
                result = improver(
                    section_name=section_name,
                    current_guideline=best_guideline,
                    failure_examples=json.dumps(failures[:5], indent=2),
                )
            
            candidate_guideline = result.improved_guideline.strip()
            
            # Remove string wrapping or markers if emitted
            if candidate_guideline.startswith('"') and candidate_guideline.endswith('"'):
                candidate_guideline = candidate_guideline[1:-1]
            elif candidate_guideline.startswith("```") and candidate_guideline.endswith("```"):
                lines = candidate_guideline.split("\n")
                candidate_guideline = "\n".join(lines[1:-1])

            # Evaluate candidate
            candidate_results = evaluator.evaluate(candidate_guideline, train_ex)
            candidate_acc = candidate_results["accuracy"]
            candidate_score = candidate_results["avg_score"]
            
            # Validate growth constraint
            growth = (len(candidate_guideline) - len(original_text)) / max(1, len(original_text))
            
            if growth > config.max_prompt_growth:
                console.print(f"  [red]✗[/red] Candidate rejected: growth too high ({growth:+.1%})")
                continue

            if candidate_score > best_score or (candidate_score == best_score and candidate_acc > best_acc):
                best_guideline = candidate_guideline
                best_acc = candidate_acc
                best_score = candidate_score
                console.print(f"  [green]✓[/green] Improved! Accuracy: {best_acc:.1%} | Avg Score: {best_score:.3f}")
            else:
                console.print(f"  [yellow]⚠[/yellow] No improvement (candidate: {candidate_acc:.1%} | {candidate_score:.3f})")

        except Exception as e:
            console.print(f"  [red]Error during round {round_num + 1}: {e}[/red]")

    return best_guideline


@click.command()
@click.option("--section", required=True, type=click.Choice(["MEMORY_GUIDANCE", "SESSION_SEARCH_GUIDANCE", "DEFAULT_AGENT_IDENTITY", "SKILLS_GUIDANCE"]))
@click.option("--iterations", default=5, help="Number of optimization iterations")
@click.option("--eval-source", default="combined", type=click.Choice(["synthetic", "sessiondb", "combined"]))
@click.option("--scenarios-count", default=15, help="Number of scenarios to generate if synthetic")
@click.option("--method", default="direct", type=click.Choice(["direct", "gepa"]))
@click.option("--run-tests", is_flag=True, help="Run all pytest unit tests as regression check")
@click.option("--dry-run", is_flag=True, help="Extract prompts and generate dataset without optimizing")
@click.option("--output-dir", default=None, help="Custom output directory")
@click.option("--optimizer-model", default="openai/minimax-m3-free", help="Model for optimizing / GEPA reflections")
@click.option("--eval-model", default="openai/minimax-m3-free", help="Model for evaluations")
@click.option("--judge-model", default="openai/minimax-m3-free", help="Model for LLM-as-judge / dataset generation")
def main(
    section: str,
    iterations: int,
    eval_source: str,
    scenarios_count: int,
    method: str,
    run_tests: bool,
    dry_run: bool,
    output_dir: Optional[str],
    optimizer_model: str,
    eval_model: str,
    judge_model: str,
):
    """Orchestrate Phase 3: System Prompt Evolution."""
    console.print(Panel(
        f"[bold cyan]🧬 Hermes Agent Self-Evolution — Phase 3[/bold cyan]\n"
        f"Optimizing System Prompt Section: [bold]{section}[/bold]",
        title="Phase 3",
        border_style="cyan",
    ))

    # Configure DSPy globally using setup_dspy_lm helper
    setup_dspy_lm(eval_model)
    console.print(f"[dim]Configured DSPy with model: {eval_model}[/dim]")

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        run_pytest=run_tests,
    )

    if output_dir:
        config.output_dir = Path(output_dir)

    hermes_agent_path = config.hermes_agent_path
    console.print(f"\n[bold]Setup:[/bold]")
    console.print(f"  hermes-agent: {hermes_agent_path}")
    console.print(f"  Iterations: {iterations}")
    console.print(f"  Evaluation Source: {eval_source}")

    # 1. Extract original prompt section text
    console.print(f"\n[bold]Extracting system prompt section...[/bold]")
    original_text = extract_prompt_section(hermes_agent_path, section)
    console.print(f"  Original Length: {len(original_text)} characters")
    console.print(f"  Baseline Text:\n  [dim]'{original_text[:120]}...'[/dim]")

    # 2. Build dataset
    dataset = PromptBehaviorDataset()
    
    if eval_source in ("synthetic", "combined"):
        builder = SyntheticPromptScenarioBuilder(config)
        synthetic_data = builder.generate(section, original_text, num_scenarios=scenarios_count)
        dataset.extend(synthetic_data.examples)

    if eval_source in ("sessiondb", "combined"):
        miner = SessionDBPromptMiner(config)
        mined_data = miner.mine_scenarios(section)
        dataset.extend(mined_data)

    if dataset.size == 0:
        console.print("[red]No evaluation scenarios built/mined![/red]")
        sys.exit(1)

    console.print(f"\n[bold]Evaluation dataset ready:[/bold] {dataset.size} examples total")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — Setup validated successfully.[/bold green]")
        return

    # 3. Baseline Evaluation on holdout set
    splits = dataset.split(config.train_ratio, config.val_ratio)
    holdout_ex = splits["holdout"]
    
    evaluator = PromptBehaviorEvaluator(config)
    console.print(f"\n[bold]Evaluating baseline on holdout set ({len(holdout_ex)} examples)...[/bold]")
    baseline_holdout = evaluator.evaluate(original_text, holdout_ex)
    console.print(f"  Baseline Holdout Accuracy: {baseline_holdout['accuracy']:.1%}")
    console.print(f"  Baseline Holdout Avg Score: {baseline_holdout['avg_score']:.3f}")

    # 4. Run Optimization
    if method == "direct":
        evolved_text = evolve_direct(section, original_text, dataset, config, iterations)
    else:
        # GEPA optimizer implementation
        console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]")
        trainset = [ex.to_dspy_example() for ex in splits["train"]]
        valset = [ex.to_dspy_example() for ex in splits["val"]]

        baseline_module = PromptSectionModule(original_text)
        try:
            optimizer = dspy.GEPA(
                metric=prompt_behavior_metric,
                max_steps=iterations,
            )
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
                valset=valset,
            )
            evolved_text = optimized_module.system_guideline
        except Exception as e:
            console.print(f"[yellow]GEPA compilation failed: {e}. Falling back to Direct method.[/yellow]")
            evolved_text = evolve_direct(section, original_text, dataset, config, iterations)

    # 5. Constraint Validation
    validator = ConstraintValidator(config)
    constraints = validator.validate_all(evolved_text, "system_prompt_section", baseline_text=original_text)
    
    passed_all = True
    console.print(f"\n[bold]Validating constraints...[/bold]")
    for r in constraints:
        icon = "✅" if r.passed else "❌"
        color = "green" if r.passed else "red"
        console.print(f"  {icon} [{color}]{r.constraint_name}[/{color}]: {r.message}")
        if not r.passed:
            passed_all = False

    if not passed_all:
        console.print("[red]✗ Evolved prompt failed constraints. Aborting deployment.[/red]")
        sys.exit(1)

    # 6. Evaluate Evolved prompt on holdout
    console.print(f"\n[bold]Evaluating evolved prompt on holdout set...[/bold]")
    evolved_holdout = evaluator.evaluate(evolved_text, holdout_ex)
    
    table = Table(title="Evolution Holdout Metrics")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Difference", justify="right")
    
    score_diff = evolved_holdout['avg_score'] - baseline_holdout['avg_score']
    diff_color = "green" if score_diff > 0 else ("red" if score_diff < 0 else "white")
    
    table.add_row(
        "Accuracy",
        f"{baseline_holdout['accuracy']:.1%}",
        f"{evolved_holdout['accuracy']:.1%}",
        f"[{diff_color}]{evolved_holdout['accuracy'] - baseline_holdout['accuracy']:+.1%}[/{diff_color}]"
    )
    table.add_row(
        "Avg Behavior Score",
        f"{baseline_holdout['avg_score']:.3f}",
        f"{evolved_holdout['avg_score']:.3f}",
        f"[{diff_color}]{score_diff:+.3f}[/{diff_color}]"
    )
    table.add_row(
        "Character Count",
        f"{len(original_text)}",
        f"{len(evolved_text)}",
        f"{len(evolved_text) - len(original_text):+}"
    )
    console.print(table)

    # 7. Run Safety Gates (Tests)
    if run_tests:
        console.print(f"\n[bold]Running regression tests on hermes-agent...[/bold]")
        # Apply temporary patches to run tests
        temp_file = hermes_agent_path / "agent" / "prompt_builder.py"
        backup_content = temp_file.read_text(encoding="utf-8")
        try:
            patch_prompt_builder(hermes_agent_path, section, evolved_text)
            test_result = validator.run_test_suite(hermes_agent_path)
            if test_result.passed:
                console.print("[green]✅ Regression tests passed successfully![/green]")
            else:
                console.print(f"[red]❌ Regression tests failed![/red]\n{test_result.details}")
                sys.exit(1)
        finally:
            # Restore backup after test suite runs
            temp_file.write_text(backup_content, encoding="utf-8")

    # 8. Deploy / Write back
    if score_diff > 0:
        console.print(f"\n[bold green]✅ Success! Evolved prompt improved holdout scores.[/bold green]")
        branch = write_evolved_prompts(hermes_agent_path, section, evolved_text)
        console.print(f"  Evolved changes written to branch: [bold cyan]{branch}[/bold cyan]")
        
        # Save output logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = config.output_dir / "prompts" / f"{section.lower()}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        (log_dir / "baseline.txt").write_text(original_text)
        (log_dir / "evolved.txt").write_text(evolved_text)
        
        metrics = {
            "section": section,
            "baseline_holdout_score": baseline_holdout['avg_score'],
            "evolved_holdout_score": evolved_holdout['avg_score'],
            "score_diff": score_diff,
            "branch": branch,
        }
        (log_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        console.print(f"  Metrics and logs saved to: {log_dir}/")
    else:
        console.print(f"\n[yellow]⚠ Optimization did not exceed baseline score. Evolved branch was not created.[/yellow]")


if __name__ == "__main__":
    main()
