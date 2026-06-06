"""System prompt evolution package for Hermes Agent Self-Evolution (Phase 3).

Optimizes target system prompt sections (e.g., MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE)
using DSPy + GEPA or failure-driven LLM optimization.
"""

from evolution.prompts.prompt_section_module import PromptSectionModule
from evolution.prompts.prompt_selection_evaluator import (
    PromptBehaviorExample,
    PromptBehaviorDataset,
    PromptBehaviorEvaluator,
)

__all__ = [
    "PromptSectionModule",
    "PromptBehaviorExample",
    "PromptBehaviorDataset",
    "PromptBehaviorEvaluator",
]
