# Autoresearch Wrapper for Hermes Agent Self-Evolution

This directory implements the [Karpathy Autoresearch](https://github.com/karpathy/autoresearch) pattern for Phases 2 and 3 of the Hermes Agent self-evolution pipeline.

Instead of a complex programmatic optimization framework, we use an **autonomous agent loop** — an AI agent reads instructions from `program.md`, makes targeted edits to hermes-agent source files, evaluates the results, and commits improvements or reverts failures using git.

## How It Works

```
┌──────────────────────────────────────────────────┐
│  1. READ program.md for goal, rules, constraints │
│  2. RUN evaluate.py to check baseline metric     │
│  3. INSPECT results.json to understand failures  │
│  4. EDIT target files (descriptions / prompts)   │
│  5. RUN evaluate.py again to test changes        │
│  6. COMPARE metrics:                             │
│     ✅ Better → git commit                       │
│     ❌ Worse  → git checkout (revert)            │
│  7. REPEAT until target metric is optimized      │
└──────────────────────────────────────────────────┘
```

## Directory Structure

```
autoresearch/
├── README.md                              # This file
├── generate_datasets.py                   # Generates prompt behavior eval datasets
├── phase2_tool_descriptions/
│   ├── program.md                         # Agent instructions for tool description optimization
│   └── evaluate.py                        # Evaluation harness (outputs accuracy + results.json)
└── phase3_system_prompts/
    ├── program.md                         # Agent instructions for system prompt optimization
    └── evaluate.py                        # Evaluation harness (outputs avg_score + results.json)
```

## Prerequisites

1. **hermes-agent** installed at `~/.hermes/hermes-agent` (or set `HERMES_AGENT_REPO` env var)
2. **OPENCODE_ZEN_API_KEY** set in `~/.hermes/.env`
3. Python dependencies installed: `pip install -e ".[dev]"`

## Phase 2: Tool Description Optimization

**Goal:** Improve tool selection accuracy by evolving the natural language descriptions in tool schemas.

**Target files:** `~/.hermes/hermes-agent/tools/*.py` (description strings only)

**Metric:** Tool selection accuracy on `datasets/tools/devtools_v1.jsonl`

```bash
# Run evaluation
python autoresearch/phase2_tool_descriptions/evaluate.py

# Inspect failures
cat autoresearch/phase2_tool_descriptions/results.json | python -m json.tool
```

Read [phase2_tool_descriptions/program.md](phase2_tool_descriptions/program.md) for the full agent instructions.

## Phase 3: System Prompt Optimization

**Goal:** Improve agent behavioral adherence by evolving system prompt guideline constants.

**Target file:** `~/.hermes/hermes-agent/agent/prompt_builder.py`

**Sections:** `MEMORY_GUIDANCE`, `SESSION_SEARCH_GUIDANCE`, `DEFAULT_AGENT_IDENTITY`, `SKILLS_GUIDANCE`

**Metric:** Average behavior score (LLM-as-judge, 0.0–1.0)

```bash
# Generate datasets (first time only)
python autoresearch/generate_datasets.py

# Run evaluation for a specific section
python autoresearch/phase3_system_prompts/evaluate.py --section MEMORY_GUIDANCE

# Inspect failures
cat autoresearch/phase3_system_prompts/results.json | python -m json.tool
```

Read [phase3_system_prompts/program.md](phase3_system_prompts/program.md) for the full agent instructions.

## Running the Autonomous Loop

Point any agentic coding tool (Claude Code, Cursor, Hermes itself, etc.) at the relevant `program.md` file and let it run. The agent will:

1. Read `program.md` for instructions
2. Run `evaluate.py` to get baseline metrics
3. Inspect `results.json` for failure patterns
4. Edit the target files to improve the metric
5. Re-run `evaluate.py` and commit or revert

Example with Claude Code:
```bash
cd /path/to/hermes-agent-self-evolution
# Point Claude at the program.md and let it iterate
claude "Read autoresearch/phase2_tool_descriptions/program.md and follow the research loop."
```

## Design Principles

- **Simplicity over complexity:** No DSPy/GEPA framework needed — just an agent, a metric, and git.
- **Deterministic evaluation:** Same dataset, same model, reproducible scores.
- **Git-based ratchet:** Only improvements survive — the repo only moves forward.
- **Human reviewable:** Every change is a git commit with a clear metric delta.
