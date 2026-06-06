"""Tool description evolution for Hermes Agent Self-Evolution (Phase 2).

Optimizes the natural language descriptions in tool schemas so the agent
picks the right tools more reliably. Uses DSPy + GEPA for reflective
evolutionary search.

Modules:
    - tool_description_module.py  -- DSPy wrapper for tool descriptions
    - tool_selection_evaluator.py -- Builds eval datasets + tests selection
    - evolve_tool_descriptions.py  -- Main entry point (CLI)
"""
