# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Automatically evolve and optimize Hermes Agent's skills, tool descriptions, and system prompts — producing measurably better versions through autonomous evaluation loops.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants.

## How It Works

This project uses two complementary approaches:

### Autoresearch (Recommended — Phase 2 & 3)
The [Karpathy autoresearch](https://github.com/karpathy/autoresearch) pattern: an autonomous AI agent reads instructions, edits target files, evaluates results, and commits improvements using git as a ratchet.

```
Agent reads program.md → runs evaluate.py → inspects results.json
    → edits target files → re-runs evaluate.py
    → metric improved? → git commit : git revert
    → repeat
```

### DSPy + GEPA (Phase 1)
Uses [DSPy](https://github.com/stanfordnlp/dspy) and [GEPA](https://github.com/gepa-ai/gepa) (Genetic-Pareto Prompt Evolution) for reflective evolutionary search on skill files.

## Quick Start

```bash
# Clone
git clone https://github.com/ETHNHUNT/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Run baseline evaluation first
python baseline_eval.py

# Then use the autoresearch loop (see autoresearch/README.md)
```

## Project Structure

```
hermes-agent-self-evolution/
├── autoresearch/                  # Autonomous optimization loops
│   ├── phase2_tool_descriptions/  #   Tool description optimization
│   │   ├── program.md             #   Agent instructions
│   │   └── evaluate.py            #   Evaluation harness
│   ├── phase3_system_prompts/     #   System prompt optimization
│   │   ├── program.md             #   Agent instructions
│   │   └── evaluate.py            #   Evaluation harness
│   ├── generate_datasets.py       #   Generate prompt eval datasets
│   └── README.md                  #   Detailed documentation
├── datasets/                      # Evaluation datasets
│   ├── tools/
│   │   ├── combined_eval.jsonl    #   341 tool selection examples
│   │   └── devtools_v1.jsonl      #   132 handcrafted examples
│   └── prompts/                   #   Prompt behavior scenarios
├── evolution/                     # DSPy/GEPA evolution modules
│   ├── core/                      #   Config, constraints, fitness
│   ├── skills/                    #   Skill file evolution (Phase 1)
│   ├── tools/                     #   Tool description evolution
│   └── prompts/                   #   Prompt section evolution
├── baseline_eval.py               # Standalone baseline evaluation
├── reports/                       # Validation reports
├── tests/                         # Unit tests
├── PLAN.md                        # Full architecture plan
└── pyproject.toml                 # Python packaging
```

## What It Optimizes

| Phase | Target | Approach | Status |
|-------|--------|----------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Validated |
| **Phase 2** | Tool descriptions | Autoresearch | ✅ Implemented |
| **Phase 3** | System prompt sections | Autoresearch | ✅ Implemented |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |

## Guardrails

Every evolved variant must pass:
1. **Size limits** — Tool descriptions ≤500 chars, skills ≤15KB
2. **Metric improvement** — Must score higher than baseline on evaluation dataset
3. **Git ratchet** — Only improvements are committed; failures are reverted
4. **Human review** — All changes go through PR review

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation strategy, constraints, and phased timeline.

## License

MIT
