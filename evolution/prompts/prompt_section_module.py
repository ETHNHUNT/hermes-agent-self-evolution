"""Wraps system prompt sections as a DSPy module for GEPA optimization.

Each target system prompt section text becomes an optimizable parameter.
On each forward pass, the module simulates the agent's behavior (tool calls and response)
given a task scenario and the candidate system prompt section text.
"""

import json
from typing import Optional

import dspy


class AgentBehaviorSignature(dspy.Signature):
    """Simulate the agent's tool-calling behavior under a specific system prompt guideline.

    Given a system prompt guideline (e.g. MEMORY_GUIDANCE) and the conversation context,
    decide if the agent should invoke a tool (and if so, what parameters).
    """
    system_guideline = dspy.InputField(desc="The specific system prompt guideline or rule to follow")
    conversation_context = dspy.InputField(desc="The user request and surrounding conversation context")
    tool_calls = dspy.OutputField(desc="JSON list of tool calls to make (e.g., [{'name': 'memory', 'arguments': {...}}] or [] if no tool is appropriate)")
    agent_response = dspy.OutputField(desc="The text response from the agent explaining the action or response")


class PromptSectionModule(dspy.Module):
    """DSPy module that wraps system prompt sections for optimization.

    The system prompt section text is the optimizable parameter that GEPA mutates.
    On each forward pass:
    1. Pass the candidate system prompt section (system_guideline) to the predictor.
    2. Retrieve the agent's simulated tool calls and response.
    3. Return a prediction with the simulation results.
    """

    def __init__(self, system_guideline: str):
        super().__init__()
        self.system_guideline = system_guideline
        self.predictor = dspy.ChainOfThought(AgentBehaviorSignature)

    def forward(self, conversation_context: str, correct_tool: str, rubric: str) -> dspy.Prediction:
        """Run agent behavior simulation and return prediction.

        Args:
            conversation_context: The task/dialogue scenario to evaluate.
            correct_tool: Ground-truth tool name or "none".
            rubric: The expected behavior rubric.

        Returns:
            dspy.Prediction containing simulated outputs.
        """
        result = self.predictor(
            system_guideline=self.system_guideline,
            conversation_context=conversation_context,
        )
        return dspy.Prediction(
            tool_calls=result.tool_calls,
            agent_response=result.agent_response,
            system_guideline=self.system_guideline,
            correct_tool=correct_tool,
            rubric=rubric,
        )


class EvaluatePromptAdherence(dspy.Signature):
    """Evaluate how well an agent's simulated action matches the expected behavior rubric under the prompt rules.

    Analyze the conversation context, the rubric, the candidate system guideline, and the simulated response/tool calls.
    Score the adherence from 0.0 to 1.0.
    Ensure:
    - If the guideline forbids saving PR/issue/temporary numbers, and the simulated action saved them, score must be 0.0.
    - If the guideline requires declarative memory format, penalize imperative format (instructions to self).
    - If it followed all rules and met the rubric, score should be 1.0.
    """
    conversation_context = dspy.InputField(desc="The user query and context")
    system_guideline = dspy.InputField(desc="The system guideline text")
    rubric = dspy.InputField(desc="The expected behavior rubric")
    simulated_tool_calls = dspy.InputField(desc="Tool calls from the simulation")
    simulated_response = dspy.InputField(desc="Agent text response")
    score = dspy.OutputField(desc="Float score from 0.0 to 1.0 (e.g. 0.8)")
    reasoning = dspy.OutputField(desc="Reasoning for the score")


def prompt_behavior_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str = "",
    pred_trace: Optional[dict] = None,
) -> float:
    """DSPy-compatible metric function for prompt behavior optimization.

    Scores the candidate response against the expected tool selection and rubric.
    """
    # 1. Parse prediction tool calls
    try:
        pred_calls = json.loads(prediction.tool_calls)
        if not isinstance(pred_calls, list):
            # Try parsing if it's single dictionary or string-wrapped list
            if isinstance(pred_calls, dict):
                pred_calls = [pred_calls]
            else:
                pred_calls = []
    except Exception:
        pred_calls = []

    correct_tool = example.correct_tool.lower().strip()

    # 2. Check tool selection correctness
    tool_correct = False
    if correct_tool == "none" or not correct_tool:
        tool_correct = len(pred_calls) == 0
    else:
        # Check if the correct tool was called
        tool_correct = any(call.get("name", "").lower() == correct_tool for call in pred_calls)

    if not tool_correct:
        return 0.0

    # 3. LLM-as-judge scoring for rubric adherence
    lm = dspy.settings.lm
    if lm is None:
        return 1.0 if tool_correct else 0.0

    evaluator = dspy.ChainOfThought(EvaluatePromptAdherence)

    try:
        # We need the guideline that was used in this prediction.
        # This allows GEPA to know what mutated prompt was being evaluated.
        system_guideline = getattr(prediction, "system_guideline", "")

        with dspy.context(lm=lm):
            eval_result = evaluator(
                conversation_context=example.conversation_context,
                system_guideline=system_guideline,
                rubric=example.rubric,
                simulated_tool_calls=prediction.tool_calls,
                simulated_response=prediction.agent_response,
            )
        score = float(eval_result.score)
        return min(max(score, 0.0), 1.0)
    except Exception:
        # Fallback if judge model fails
        return 1.0 if tool_correct else 0.0
