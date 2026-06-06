"""Wraps tool descriptions as a DSPy module for GEPA optimization.

The key abstraction: each tool's description text becomes an optimizable
parameter. On each forward pass, the module presents a task to the LLM
(with the current tool descriptions) and checks whether the LLM picks
the right tool.

Cross-tool evaluation: ALL tool descriptions are optimized simultaneously,
not in isolation. This prevents one tool's description from "stealing"
selections from another.
"""

import json
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig


class ToolSelectionSignature(dspy.Signature):
    """Given a task and a list of available tools with their descriptions,
    select the most appropriate tool to complete the task.

    Consider:
    - What the task requires (reading, writing, searching, executing, etc.)
    - The specific capabilities of each tool as described
    - Whether the task requires file operations, terminal commands, web access, etc.
    """
    task_description: str = dspy.InputField(desc="The task the agent needs to complete")
    available_tools: str = dspy.InputField(desc="JSON array of {name, description} for each available tool")
    selected_tool: str = dspy.OutputField(desc="The name of the tool that should be used for this task")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why this tool was chosen")


class ToolDescriptionModule(dspy.Module):
    """DSPy module that wraps tool descriptions for optimization.

    The tool descriptions dict is the parameter that GEPA optimizes.
    On each forward pass:
    1. Format tool descriptions as context
    2. Present a task to the LLM
    3. Check if the LLM picks the correct tool
    4. Return a prediction with the selection result
    """

    def __init__(self, tool_descriptions: dict[str, str]):
        """
        Args:
            tool_descriptions: Dict mapping tool_name -> description text.
                             These are the parameters GEPA will optimize.
        """
        super().__init__()
        self.tool_descriptions = tool_descriptions
        self.predictor = dspy.ChainOfThought(ToolSelectionSignature)

    def forward(self, task_description: str, correct_tool: str) -> dspy.Prediction:
        """Run tool selection and return prediction with correctness signal.

        Args:
            task_description: The task the agent needs to complete
            correct_tool: The ground-truth correct tool name

        Returns:
            dspy.Prediction with selected_tool, reasoning, and correct fields
        """
        # Format available tools as JSON for the LLM
        tools_json = json.dumps(
            [{"name": name, "description": desc} for name, desc in self.tool_descriptions.items()],
            indent=2
        )

        result = self.predictor(
            task_description=task_description,
            available_tools=tools_json,
        )

        selected = result.selected_tool.strip().lower()
        correct = correct_tool.strip().lower()

        # Fuzzy match: the LLM might return "read_file" or "The tool is read_tool"
        is_correct = (
            selected == correct
            or correct in selected
            or selected in correct
        )

        return dspy.Prediction(
            selected_tool=result.selected_tool,
            reasoning=result.reasoning,
            correct_tool=correct_tool,
            is_correct=is_correct,
        )

    def format_tools_context(self) -> str:
        """Format tool descriptions as a readable context string."""
        lines = []
        for name, desc in sorted(self.tool_descriptions.items()):
            lines.append(f"  {name}: {desc}")
        return "\n".join(lines)


def tool_selection_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str = "",
    pred_trace: Optional[dict] = None,
) -> float:
    """DSPy-compatible metric function for tool selection optimization.

    This is what gets passed to dspy.GEPA(metric=...).
    Returns 1.0 if the correct tool was selected, 0.0 otherwise.

    GEPA reads the trace (execution logs) to understand WHY the wrong
    tool was picked, then proposes targeted description mutations.
    """
    is_correct = getattr(prediction, "is_correct", False)
    if is_correct:
        return 1.0

    # Partial credit: if the reasoning shows understanding but wrong selection
    reasoning = getattr(prediction, "reasoning", "") or ""
    correct_tool = getattr(prediction, "correct_tool", "") or ""

    # If the reasoning mentions the correct tool but selected wrong, partial credit
    if correct_tool.lower() in reasoning.lower():
        return 0.3

    return 0.0
