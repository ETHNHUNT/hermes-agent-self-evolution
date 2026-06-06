"""Dataset management and evaluators for Phase 3: System Prompt Evolution.

Generates evaluation scenarios synthetically using an LLM and mines real-world usage patterns
from state.db (SQLite) to test memory and session search behaviors under candidate prompt guidelines.
"""

import json
import random
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import dspy
from rich.console import Console

from evolution.core.config import EvolutionConfig
from evolution.prompts.prompt_section_module import PromptSectionModule, prompt_behavior_metric

console = Console()


@dataclass
class PromptBehaviorExample:
    """A single prompt behavior test case."""
    conversation_context: str
    correct_tool: str  # e.g., 'memory', 'session_search', or 'none'
    rubric: str
    difficulty: str = "medium"
    category: str = "general"
    source: str = "synthetic"

    def to_dspy_example(self) -> dspy.Example:
        return dspy.Example(
            conversation_context=self.conversation_context,
            correct_tool=self.correct_tool,
            rubric=self.rubric,
        ).with_inputs("conversation_context", "correct_tool", "rubric")

    def to_dict(self) -> dict:
        return {
            "conversation_context": self.conversation_context,
            "correct_tool": self.correct_tool,
            "rubric": self.rubric,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptBehaviorExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PromptBehaviorDataset:
    """Collection of prompt behavior examples with train/val/holdout splits."""

    def __init__(self):
        self.examples: list[PromptBehaviorExample] = []

    def add(self, example: PromptBehaviorExample):
        self.examples.append(example)

    def extend(self, examples: list[PromptBehaviorExample]):
        self.examples.extend(examples)

    @property
    def size(self) -> int:
        return len(self.examples)

    def split(self, train_ratio=0.5, val_ratio=0.25) -> dict[str, list[PromptBehaviorExample]]:
        """Split examples into train/val/holdout sets."""
        shuffled = list(self.examples)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "holdout": shuffled[n_train + n_val:],
        }

    def save(self, path: Path):
        """Save to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
        console.print(f"  Saved {len(self.examples)} examples to {path}")

    @classmethod
    def load(cls, path: Path) -> "PromptBehaviorDataset":
        """Load from JSONL file."""
        dataset = cls()
        if not path.exists():
            return dataset
        with open(path) as f:
            for line in f:
                if line.strip():
                    dataset.add(PromptBehaviorExample.from_dict(json.loads(line)))
        return dataset


class SyntheticPromptScenarioBuilder:
    """Generates synthetic behavioral scenarios using an LLM based on prompt section rules."""

    class GenerateScenariosSignature(dspy.Signature):
        """Generate diverse test scenarios for evaluating agent behavior under system prompt guidelines.

        Given the name of a prompt section (e.g., MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE) and its original text,
        generate diverse, realistic dialogue/task scenarios that specifically test whether the agent adheres
        to the rules in that prompt section.

        Guidelines:
        - Generate some scenarios where the agent SHOULD invoke the tool (positive cases).
        - Generate some scenarios where the agent SHOULD NOT invoke the tool, or should do something else (negative cases).
        - Include a specific expected_behavior_rubric that describes how the agent should behave and what to check.
        - Output a JSON list of objects: [{"conversation_context": "...", "correct_tool": "...", "rubric": "...", "difficulty": "..."}].
        """
        section_name = dspy.InputField(desc="The name of the target system prompt section")
        original_text = dspy.InputField(desc="The original text of the system prompt section")
        num_scenarios = dspy.InputField(desc="Number of test scenarios to generate")
        scenarios = dspy.OutputField(desc="JSON list of test scenarios")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateScenariosSignature)

    def generate(self, section_name: str, original_text: str, num_scenarios: int = 15) -> PromptBehaviorDataset:
        """Generate synthetic prompt behavior dataset."""
        dataset = PromptBehaviorDataset()
        lm = dspy.settings.lm or dspy.LM(self.config.judge_model)

        console.print(f"\n[bold]Generating synthetic scenarios for {section_name}[/bold]")
        console.print(f"  Expected scenarios: {num_scenarios}")

        with dspy.context(lm=lm):
            result = self.generator(
                section_name=section_name,
                original_text=original_text,
                num_scenarios=num_scenarios,
            )

        try:
            scenarios_raw = json.loads(result.scenarios)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', result.scenarios, re.DOTALL)
            if match:
                scenarios_raw = json.loads(match.group())
            else:
                console.print("[red]Failed to parse generated scenarios JSON[/red]")
                return dataset

        for item in scenarios_raw:
            context = item.get("conversation_context", "").strip()
            tool = item.get("correct_tool", "none").strip()
            rubric = item.get("rubric", "").strip()
            if context and rubric:
                dataset.add(PromptBehaviorExample(
                    conversation_context=context,
                    correct_tool=tool,
                    rubric=rubric,
                    difficulty=item.get("difficulty", "medium"),
                    category=section_name.lower(),
                    source="synthetic",
                ))

        console.print(f"  Generated {dataset.size} synthetic scenarios")
        return dataset


class SessionDBPromptMiner:
    """Mines real user messages and tools usage from state.db to find prompt failures/adherence."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def mine_scenarios(self, section_name: str) -> list[PromptBehaviorExample]:
        """Mine prompt behavior examples from SQLite session database."""
        examples = []
        db_path = Path.home() / ".hermes" / "state.db"

        if not db_path.exists():
            console.print(f"  [dim]No session DB found at {db_path} — skipping mining[/dim]")
            return examples

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Retrieve all messages sorted by session and execution order
            cursor.execute("SELECT session_id, role, content FROM messages ORDER BY rowid")
            all_msgs = cursor.fetchall()
            conn.close()

            # Group messages by session_id
            sessions = {}
            for session_id, role, content in all_msgs:
                if not content:
                    continue
                sessions.setdefault(session_id, []).append((role, str(content)))

            console.print(f"  Found {len(sessions)} historical sessions in state.db")

            for session_id, msgs in sessions.items():
                for i, (role, content) in enumerate(msgs):
                    if role != "user":
                        continue

                    content_lower = content.lower()
                    if section_name == "MEMORY_GUIDANCE":
                        # Look for user stated preferences or conventions
                        keywords = ["prefer", "always", "convention", "setup", "style", "usually"]
                        if any(kw in content_lower for kw in keywords) and len(content) > 15:
                            has_memory_tool = False
                            # Look ahead for memory tool calls in assistant messages
                            for j in range(i + 1, min(i + 4, len(msgs))):
                                next_role, next_content = msgs[j]
                                if next_role == "assistant" and "memory" in next_content:
                                    has_memory_tool = True
                                    break

                            rubric = (
                                "Agent should invoke the memory tool to save the stated user preference."
                                if has_memory_tool else "Evaluate if agent should save memory facts."
                            )
                            examples.append(PromptBehaviorExample(
                                conversation_context=content,
                                correct_tool="memory" if has_memory_tool else "none",
                                rubric=rubric,
                                difficulty="medium",
                                category="memory_mined",
                                source="sessiondb",
                            ))

                    elif section_name == "SESSION_SEARCH_GUIDANCE":
                        # Look for user referring to past sessions/context
                        keywords = ["yesterday", "last time", "we did", "remember", "previously", "past", "history"]
                        if any(kw in content_lower for kw in keywords) and len(content) > 15:
                            has_search_tool = False
                            for j in range(i + 1, min(i + 4, len(msgs))):
                                next_role, next_content = msgs[j]
                                if next_role == "assistant" and "session_search" in next_content:
                                    has_search_tool = True
                                    break

                            rubric = (
                                "Agent should use session_search to retrieve past conversation history."
                                if has_search_tool else "Evaluate if agent should search past sessions."
                            )
                            examples.append(PromptBehaviorExample(
                                conversation_context=content,
                                correct_tool="session_search" if has_search_tool else "none",
                                rubric=rubric,
                                difficulty="medium",
                                category="search_mined",
                                source="sessiondb",
                            ))

        except Exception as e:
            console.print(f"  [yellow]Session mining error: {e}[/yellow]")

        console.print(f"  Mined {len(examples)} examples from SQLite state.db")
        return examples


class PromptBehaviorEvaluator:
    """Evaluates prompt behavior scores using candidate system prompt section text."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def evaluate(self, system_guideline: str, examples: list[PromptBehaviorExample]) -> dict:
        """Evaluate a system prompt guideline text on a list of examples."""
        module = PromptSectionModule(system_guideline)
        lm = dspy.settings.lm or dspy.LM(self.config.eval_model)

        correct = 0
        total = 0
        scores = []
        failures = []

        for example in examples:
            try:
                with dspy.context(lm=lm):
                    prediction = module(
                        conversation_context=example.conversation_context,
                        correct_tool=example.correct_tool,
                        rubric=example.rubric,
                    )

                score = prompt_behavior_metric(example.to_dspy_example(), prediction)
                total += 1
                scores.append(score)

                # Score >= 0.8 is considered a correct pass
                if score >= 0.8:
                    correct += 1
                else:
                    failures.append({
                        "context": example.conversation_context[:100],
                        "expected_tool": example.correct_tool,
                        "rubric": example.rubric[:100],
                        "simulated_calls": prediction.tool_calls,
                        "simulated_response": prediction.agent_response[:150],
                        "score": score,
                    })

            except Exception as e:
                console.print(f"  [dim]Error evaluating example: {e}[/dim]")

        accuracy = correct / max(1, total)
        avg_score = sum(scores) / max(1, total)

        return {
            "accuracy": accuracy,
            "avg_score": avg_score,
            "correct": correct,
            "total": total,
            "failures": failures,
        }
