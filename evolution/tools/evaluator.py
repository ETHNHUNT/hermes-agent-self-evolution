"""Phase 2A evaluator with 3-dimension scoring.

Scores each tool selection example on:
  1. Tool choice accuracy (correct tool or correct no-tool decision)
  2. Parameter correctness (required arguments present and sensible)
  3. Efficiency penalty (subtract for using broad/expensive/redundant tools)

Scoring rubric:
  1.0 = correct tool + acceptable params
  0.7 = acceptable alternative tool
  0.3 = wrong but still task-progressing tool
  0.0 = wrong or unnecessary tool
  -0.1 to -0.3 = avoidable overuse of terminal or multi-step tooling
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SelectionScore:
    """Score for a single tool selection example."""
    tool_choice_score: float  # 0.0 to 1.0
    param_score: float        # 0.0 to 1.0
    efficiency_penalty: float # 0.0 to -0.3
    total_score: float        # tool_choice + param + efficiency, clamped to 0-1
    reasoning: str = ""       # Why this score was given


@dataclass
class EvalResult:
    """Full evaluation result for a dataset."""
    overall_accuracy: float
    overall_tool_choice: float
    overall_param: float
    overall_efficiency: float
    overall_total: float
    per_tool: dict
    per_category: dict
    failures: list
    scores: list  # List of SelectionScore


class DevToolEvaluator:
    """Evaluates tool selection with 3-dimension scoring.

    Uses an LLM judge to score tool selections since many cases
    require nuanced judgment (e.g., is terminal acceptable when
    search_files would be better?).
    """

    # Tools that are considered "expensive" or "broad" — using them
    # when a narrower tool exists triggers efficiency penalty.
    EFFICIENT_TOOL = {
        "search_files": 1.0,
        "read_file": 1.0,
        "write_file": 1.0,
        "patch": 1.0,
        "terminal": 0.7,   # Broad — can do anything but often overkill
        "process": 0.9,
        "NO_TOOL": 1.0,
    }

    # Known acceptable alternatives (tool -> list of acceptable alternatives)
    ACCEPTABLE_ALTERNATIVES = {
        "search_files": [],
        "read_file": [],
        "write_file": [],
        "patch": ["write_file"],  # write_file can do what patch does (but less precisely)
        "terminal": [],
        "process": [],
        "NO_TOOL": [],
    }

    def score_selection(
        self,
        task: str,
        correct_tool: str,
        selected_tool: str,
        selected_params: Optional[dict] = None,
        expected_params: Optional[dict] = None,
    ) -> SelectionScore:
        """Score a single tool selection.

        Args:
            task: The user's task description
            correct_tool: Ground truth tool
            selected_tool: What the agent chose
            selected_params: Parameters the agent used
            expected_params: Expected parameters

        Returns:
            SelectionScore with all dimensions
        """
        # ── Dimension 1: Tool choice ──
        if selected_tool == correct_tool:
            tool_score = 1.0
        elif correct_tool in self.ACCEPTABLE_ALTERNATIVES.get(selected_tool, []):
            tool_score = 0.7
        elif selected_tool == "NO_TOOL" and correct_tool != "NO_TOOL":
            tool_score = 0.0  # Should have used a tool
        elif selected_tool != "NO_TOOL" and correct_tool == "NO_TOOL":
            tool_score = 0.0  # Should NOT have used a tool
        else:
            # Check if it's at least in the right category
            file_tools = {"search_files", "read_file", "write_file", "patch"}
            if selected_tool in file_tools and correct_tool in file_tools:
                tool_score = 0.3  # Wrong file tool but at least file-oriented
            else:
                tool_score = 0.0

        # ── Dimension 2: Parameter correctness ──
        param_score = 1.0  # Default: assume correct
        if expected_params and selected_params:
            missing = []
            for key in expected_params:
                if key not in selected_params:
                    missing.append(key)
            if missing:
                param_score = max(0.0, 1.0 - 0.3 * len(missing))
        elif expected_params and not selected_params:
            param_score = 0.5  # Missing all params

        # ── Dimension 3: Efficiency penalty ─
        efficiency = 0.0
        if tool_score >= 0.7:  # Only penalize if tool choice was reasonable
            # Penalize using terminal when a file tool would be better
            if selected_tool == "terminal" and correct_tool in (
                "search_files", "read_file", "write_file", "patch"
            ):
                efficiency = -0.2
            # Penalize using write_file when patch would be more precise
            elif selected_tool == "write_file" and correct_tool == "patch":
                efficiency = -0.1
            # Penalize using search_files when read_file is enough
            elif selected_tool == "search_files" and correct_tool == "read_file":
                efficiency = -0.1

        total = max(0.0, min(1.0, tool_score + param_score - 1.0 + efficiency))

        reasoning = f"tool={tool_score}, params={param_score}, efficiency={efficiency}"

        return SelectionScore(
            tool_choice_score=tool_score,
            param_score=param_score,
            efficiency_penalty=efficiency,
            total_score=total,
            reasoning=reasoning,
        )

    def evaluate_dataset(
        self,
        selections: list[dict],
    ) -> EvalResult:
        """Evaluate a full dataset of tool selections.

        Args:
            selections: List of dicts with keys:
                task, correct_tool, selected_tool, selected_params, expected_params

        Returns:
            EvalResult with aggregate and per-tool stats
        """
        scores = []
        per_tool = {}
        per_category = {}
        failures = []

        for sel in selections:
            score = self.score_selection(
                task=sel.get("task", ""),
                correct_tool=sel.get("correct_tool", ""),
                selected_tool=sel.get("selected_tool", ""),
                selected_params=sel.get("selected_params"),
                expected_params=sel.get("expected_params"),
            )
            scores.append(score)

            # Per-tool stats
            tool = sel.get("correct_tool", "unknown")
            if tool not in per_tool:
                per_tool[tool] = {"correct": 0, "total": 0, "total_score": 0.0}
            per_tool[tool]["total"] += 1
            per_tool[tool]["total_score"] += score.total_score
            if score.tool_choice_score >= 0.7:
                per_tool[tool]["correct"] += 1
            else:
                failures.append({
                    "task": sel.get("task", "")[:100],
                    "expected": tool,
                    "got": sel.get("selected_tool", "unknown"),
                    "score": score.total_score,
                    "reasoning": score.reasoning,
                })

        n = max(1, len(scores))
        return EvalResult(
            overall_accuracy=sum(1 for s in scores if s.tool_choice_score >= 0.7) / n,
            overall_tool_choice=sum(s.tool_choice_score for s in scores) / n,
            overall_param=sum(s.param_score for s in scores) / n,
            overall_efficiency=sum(s.efficiency_penalty for s in scores) / n,
            overall_total=sum(s.total_score for s in scores) / n,
            per_tool=per_tool,
            per_category=per_category,
            failures=sorted(failures, key=lambda f: f["score"]),
            scores=scores,
        )

    def format_results(self, result: EvalResult) -> str:
        """Format evaluation results as a readable string."""
        lines = [
            "=" * 60,
            "Phase 2A Evaluation Results",
            "=" * 60,
            f"Overall Accuracy:     {result.overall_accuracy:.1%}",
            f"Tool Choice Score:    {result.overall_tool_choice:.3f}",
            f"Param Score:          {result.overall_param:.3f}",
            f"Efficiency Penalty:   {result.overall_efficiency:.3f}",
            f"Total Score:          {result.overall_total:.3f}",
            "",
            "Per-Tool Breakdown:",
            "-" * 40,
        ]
        for tool, stats in sorted(result.per_tool.items()):
            acc = stats["correct"] / max(1, stats["total"])
            avg = stats["total_score"] / max(1, stats["total"])
            lines.append(
                f"  {tool:20s}  acc={acc:.1%}  avg_score={avg:.3f}  n={stats['total']}"
            )

        if result.failures:
            lines.extend(["", "Top Failures:", "-" * 40])
            for f in result.failures[:10]:
                lines.append(
                    f"  [{f['score']:.1f}] {f['task'][:60]}..."
                )
                lines.append(f"       expected={f['expected']} got={f['got']}")

        return "\n".join(lines)
