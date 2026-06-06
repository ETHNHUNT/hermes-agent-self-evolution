"""Tool selection evaluation dataset builder for Phase 2.

Generates (task_description, correct_tool, correct_params) triples that
test whether the agent picks the right tool for a given task.

Data sources:
  1. Synthetic generation — LLM reads tool schemas and generates test tasks
  2. SessionDB mining — extracts real tool-call patterns from session history
  3. Combined — merges both sources for a robust evaluation dataset

The evaluator also runs the actual selection test: given a task and the
current tool descriptions, does the agent pick the correct tool?
"""

import json
import random
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalExample, EvalDataset
from evolution.tools.tool_description_module import ToolDescriptionModule

console = Console()


@dataclass
class ToolSelectionExample:
    """A single tool selection test case."""
    task_description: str   # What the agent is asked to do
    correct_tool: str       # The ground-truth tool that should be selected
    difficulty: str = "medium"
    category: str = "general"
    source: str = "synthetic"

    def to_eval_example(self) -> EvalExample:
        return EvalExample(
            task_input=self.task_description,
            expected_behavior=f"Should select the '{self.correct_tool}' tool",
            difficulty=self.difficulty,
            category=self.category,
            source=self.source,
        )

    def to_dict(self) -> dict:
        return {
            "task_description": self.task_description,
            "correct_tool": self.correct_tool,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolSelectionExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ToolSelectionDataset:
    """Collection of tool selection examples with train/val/holdout splits."""

    def __init__(self):
        self.examples: list[ToolSelectionExample] = []

    def add(self, example: ToolSelectionExample):
        self.examples.append(example)

    def extend(self, examples: list[ToolSelectionExample]):
        self.examples.extend(examples)

    @property
    def size(self) -> int:
        return len(self.examples)

    def split(self, train_ratio=0.5, val_ratio=0.25) -> EvalDataset:
        """Split into train/val/holdout and return as EvalDataset."""
        shuffled = list(self.examples)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_ex = [ex.to_eval_example() for ex in shuffled[:n_train]]
        val_ex = [ex.to_eval_example() for ex in shuffled[n_train:n_train + n_val]]
        holdout_ex = [ex.to_eval_example() for ex in shuffled[n_train + n_val:]]

        return EvalDataset(train=train_ex, val=val_ex, holdout=holdout_ex)

    def save(self, path: Path):
        """Save to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
        console.print(f"  Saved {len(self.examples)} examples to {path}")

    @classmethod
    def load(cls, path: Path) -> "ToolSelectionDataset":
        """Load from JSONL file."""
        dataset = cls()
        if not path.exists():
            return dataset
        with open(path) as f:
            for line in f:
                if line.strip():
                    dataset.add(ToolSelectionExample.from_dict(json.loads(line)))
        return dataset

    def to_dspy_examples(self) -> list[dspy.Example]:
        """Convert to DSPy Example objects for GEPA."""
        return [
            dspy.Example(
                task_description=ex.task_description,
                correct_tool=ex.correct_tool,
            ).with_inputs("task_description", "correct_tool")
            for ex in self.examples
        ]

    def stats(self) -> dict:
        """Return dataset statistics."""
        tools = {}
        difficulties = {}
        sources = {}
        for ex in self.examples:
            tools[ex.correct_tool] = tools.get(ex.correct_tool, 0) + 1
            difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1
            sources[ex.source] = sources.get(ex.source, 0) + 1
        return {
            "total": len(self.examples),
            "tools": tools,
            "difficulties": difficulties,
            "sources": sources,
        }


class SyntheticToolSelectionBuilder:
    """Generate synthetic tool selection test cases using an LLM.

    Reads the current tool descriptions and generates realistic tasks
    that should map to specific tools.
    """

    class GenerateSelectionTasks(dspy.Signature):
        """Generate tool selection test cases for agent evaluation.

        Given a list of available tools with their descriptions, generate
        diverse test cases where each test case is a task description that
        clearly maps to ONE specific tool.

        Guidelines:
        - Each task should unambiguously require one specific tool
        - Vary difficulty: easy (obvious tool match), medium (requires
          understanding tool nuances), hard (subtle distinction between
          similar tools)
        - Cover all tools evenly
        - Tasks should be realistic (things a user would actually ask)
        - Include tasks that test common misselection patterns
          (e.g., "search for text in files" -> search_files, NOT terminal+grep)
        """
        tool_list: str = dspy.InputField(desc="JSON array of {name, description} for each tool")
        num_tasks_per_tool: int = dspy.InputField(desc="Number of tasks to generate per tool")
        test_cases: str = dspy.OutputField(
            desc="JSON array of {task_description, correct_tool, difficulty, category}"
        )

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateSelectionTasks)

    def generate(
        self,
        tool_descriptions: dict[str, str],
        num_tasks_per_tool: int = 3,
    ) -> ToolSelectionDataset:
        """Generate synthetic tool selection dataset.

        Args:
            tool_descriptions: Dict mapping tool_name -> description
            num_tasks_per_tool: How many test cases per tool

        Returns:
            ToolSelectionDataset with generated examples
        """
        dataset = ToolSelectionDataset()

        tools_json = json.dumps(
            [{"name": n, "description": d} for n, d in tool_descriptions.items()],
            indent=2
        )

        # Use globally configured LM (has API key)
        lm = dspy.settings.lm
        if lm is None:
            lm = dspy.LM(self.config.judge_model)

        console.print(f"\n[bold]Generating synthetic tool selection tasks[/bold]")
        console.print(f"  Tools: {len(tool_descriptions)}")
        console.print(f"  Tasks per tool: {num_tasks_per_tool}")
        console.print(f"  Expected total: {len(tool_descriptions) * num_tasks_per_tool}")

        with dspy.context(lm=lm):
            result = self.generator(
                tool_list=tools_json,
                num_tasks_per_tool=num_tasks_per_tool,
            )

        try:
            cases_raw = json.loads(result.test_cases)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', result.test_cases, re.DOTALL)
            if match:
                cases_raw = json.loads(match.group())
            else:
                console.print(f"[red]Failed to parse generated tasks[/red]")
                return dataset

        for case in cases_raw:
            task = case.get("task_description", "").strip()
            tool = case.get("correct_tool", "").strip()
            if task and tool:
                dataset.add(ToolSelectionExample(
                    task_description=task,
                    correct_tool=tool,
                    difficulty=case.get("difficulty", "medium"),
                    category=case.get("category", "general"),
                    source="synthetic",
                ))

        console.print(f"  Generated {dataset.size} examples")
        return dataset


class SessionDBToolMiner:
    """Mine real tool-call patterns from Hermes session database.

    Extracts (task, tool_selected) pairs from actual usage to create
    evaluation examples based on real behavior. This captures misselection
    patterns that synthetic data might miss.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def mine_hermes_sessions(self, tool_names: list[str]) -> ToolSelectionDataset:
        """Mine tool selection examples from Hermes session database.

        Reads the state.db SQLite database to find real user requests
        and the tools that were selected for them.

        Args:
            tool_names: List of tool names to look for

        Returns:
            ToolSelectionDataset with mined examples
        """
        dataset = ToolSelectionDataset()
        db_path = Path.home() / ".hermes" / "state.db"

        if not db_path.exists():
            console.print(f"  [dim]No session DB at {db_path}[/dim]")
            return dataset

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get all sessions
            cursor.execute("SELECT session_id, system_prompt FROM sessions")
            sessions = cursor.fetchall()
            console.print(f"  Found {len(sessions)} sessions in state.db")

            for session_id, system_prompt in sessions:
                # Get messages for this session
                cursor.execute(
                    """SELECT role, content FROM messages
                       WHERE session_id = ?
                       ORDER BY rowid""",
                    (session_id,)
                )
                messages = cursor.fetchall()

                # Find user messages followed by tool calls
                for i, (role, content) in enumerate(messages):
                    if role != "user" or not content:
                        continue
                    content_str = str(content)
                    if len(content_str) < 10:
                        continue

                    # Look ahead for tool calls in assistant responses
                    for j in range(i + 1, min(i + 5, len(messages))):
                        next_role, next_content = messages[j]
                        if next_role == "assistant" and next_content:
                            next_str = str(next_content)
                            # Check if any tool name appears in the response
                            for tool_name in tool_names:
                                if tool_name.lower() in next_str.lower():
                                    dataset.add(ToolSelectionExample(
                                        task_description=content_str[:500],
                                        correct_tool=tool_name,
                                        difficulty="medium",
                                        category="session_mined",
                                        source="sessiondb",
                                    ))
                                    break
                            break

            conn.close()
        except Exception as e:
            console.print(f"  [yellow]Session mining error: {e}[/yellow]")

        console.print(f"  Mined {dataset.size} examples from sessions")
        return dataset

    def mine_codex_sessions(self, tool_names: list[str]) -> ToolSelectionDataset:
        """Mine tool selection examples from Codex session database."""
        dataset = ToolSelectionDataset()
        db_path = Path.home() / ".codex" / "state_5.sqlite"

        if not db_path.exists():
            console.print(f"  [dim]No Codex DB at {db_path}[/dim]")
            return dataset

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get thread data
            cursor.execute("SELECT * FROM threads LIMIT 50")
            threads = cursor.fetchall()
            console.print(f"  Found {len(threads)} Codex threads")

            # Extract user messages
            for thread in threads:
                thread_str = str(thread)
                for tool_name in tool_names:
                    if tool_name.lower() in thread_str.lower():
                        dataset.add(ToolSelectionExample(
                            task_description=f"Codex session involving {tool_name}",
                            correct_tool=tool_name,
                            difficulty="medium",
                            category="codex_mined",
                            source="codex",
                        ))
                        break

            conn.close()
        except Exception as e:
            console.print(f"  [yellow]Codex mining error: {e}[/yellow]")

        console.print(f"  Mined {dataset.size} examples from Codex")
        return dataset


class ToolSelectionEvaluator:
    """Evaluates tool selection accuracy given current tool descriptions.

    Runs the agent on a dataset of (task, correct_tool) pairs and
    measures how often it picks the right tool.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def evaluate(
        self,
        tool_descriptions: dict[str, str],
        dataset: ToolSelectionDataset,
    ) -> dict:
        """Run tool selection evaluation.

        Args:
            tool_descriptions: Current tool name -> description mapping
            dataset: Test dataset of (task, correct_tool) pairs

        Returns:
            Dict with accuracy, per-tool stats, and failure analysis
        """
        module = ToolDescriptionModule(tool_descriptions)
        # Use globally configured LM (has API key)
        lm = dspy.settings.lm
        if lm is None:
            lm = dspy.LM(self.config.eval_model)

        correct = 0
        total = 0
        per_tool = {}
        failures = []

        console.print(f"\n[bold]Evaluating tool selection[/bold] ({dataset.size} examples)")

        for example in dataset.examples:
            try:
                with dspy.context(lm=lm):
                    prediction = module.forward(
                        task_description=example.task_description,
                        correct_tool=example.correct_tool,
                    )

                is_correct = getattr(prediction, "is_correct", False)
                total += 1

                # Per-tool stats
                tool = example.correct_tool
                if tool not in per_tool:
                    per_tool[tool] = {"correct": 0, "total": 0}
                per_tool[tool]["total"] += 1
                if is_correct:
                    correct += 1
                    per_tool[tool]["correct"] += 1
                else:
                    failures.append({
                        "task": example.task_description[:100],
                        "expected": example.correct_tool,
                        "got": getattr(prediction, "selected_tool", "unknown"),
                        "reasoning": getattr(prediction, "reasoning", "")[:200],
                    })
            except Exception as e:
                console.print(f"  [dim]Error on example: {e}[/dim]")

        accuracy = correct / max(1, total)

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_tool": per_tool,
            "failures": failures[:20],  # Top 20 failures for analysis
        }

    def print_results(self, results: dict):
        """Pretty-print evaluation results."""
        console.print(f"\n[bold]Tool Selection Results[/bold]")
        console.print(f"  Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")

        if results["per_tool"]:
            table = Table(title="Per-Tool Accuracy")
            table.add_column("Tool", style="cyan")
            table.add_column("Correct", style="green")
            table.add_column("Total", style="white")
            table.add_column("Accuracy", style="bold")
            for tool, stats in sorted(results["per_tool"].items()):
                acc = stats["correct"] / max(1, stats["total"])
                table.add_row(tool, str(stats["correct"]), str(stats["total"]), f"{acc:.1%}")
            console.print(table)

        if results["failures"]:
            console.print(f"\n[bold red]Top Failures:[/bold red]")
            for f in results["failures"][:10]:
                console.print(f"  [red]✗[/red] Task: {f['task'][:80]}")
                console.print(f"    Expected: {f['expected']} | Got: {f['got']}")
