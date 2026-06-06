# Research Program: System Prompt Optimization (Phase 3)

Optimize the system prompt sections in the `hermes-agent` repository to guide agent behavior (e.g. memory saving, session searching, core identity) more effectively.

---

## Objective
Maximize the average behavior score for a target system prompt section on its evaluation dataset.

## Target Metric
- **primary_metric**: `avg_score` (LLM-as-judge score between 0.0 and 1.0 of the agent's simulated behavior against the rubric).
- **secondary_metric**: `accuracy` (percentage of correct simulated tool selections).
- **constraints**:
  - `max_growth`: Evolved prompt text must not exceed original section size by more than 20%.
  - `regression_tests`: (Optional) Evolved agent must pass all hermes-agent unit tests.

## Workspace Configuration
- **Modifiable Files**: `~/.hermes/hermes-agent/agent/prompt_builder.py` (specifically editing the string constants `MEMORY_GUIDANCE`, `SESSION_SEARCH_GUIDANCE`, `DEFAULT_AGENT_IDENTITY`, and `SKILLS_GUIDANCE`).
- **Read-Only Files**: 
  - Evaluation datasets: `datasets/prompts/<SECTION>_eval.jsonl`
  - Evaluation script: `autoresearch/phase3_system_prompts/evaluate.py`

## How to Evaluate
Run the following command in the `hermes-agent-self-evolution` directory, specifying the target section:
```bash
python autoresearch/phase3_system_prompts/evaluate.py --section MEMORY_GUIDANCE
```
Supported sections: `MEMORY_GUIDANCE`, `SESSION_SEARCH_GUIDANCE`, `DEFAULT_AGENT_IDENTITY`, `SKILLS_GUIDANCE`.

This script will run the evaluation scenarios through the simulator and print the accuracy and average score. It also writes the results and failures details to:
`autoresearch/phase3_system_prompts/results.json`

## Research loop (Step-by-Step)
1. **Establish Baseline**: Run the evaluation command for your target section to check the baseline metrics.
2. **Identify Failures**: Read `autoresearch/phase3_system_prompts/results.json` to analyze which scenarios failed and what the judge's reasoning was.
3. **Formulate Hypothesis**: Refine the system prompt guideline text to explicitly address the failure cases, clarify rules, or specify negative instructions.
4. **Apply Changes**: Edit the target constant in `~/.hermes/hermes-agent/agent/prompt_builder.py`.
5. **Re-evaluate**: Run the evaluation command again.
6. **Apply Ratchet (Commit or Revert)**:
   - **If metrics IMPROVED (higher avg_score or same score with higher accuracy) and size constraints are met**: Commit the changes in the `hermes-agent` repository:
     ```bash
     cd /Users/vipinnandal/.hermes/hermes-agent
     git add agent/prompt_builder.py
     git commit -m "autoresearch: improved system prompt <SECTION> behavior score to X.XX"
     cd -
     ```
   - **If metrics did NOT improve or constraints failed**: Revert the changes to keep the repository clean:
     ```bash
     cd /Users/vipinnandal/.hermes/hermes-agent
     git checkout -- agent/prompt_builder.py
     cd -
     ```
7. **Iterate**: Repeat steps 2-6 to continuously improve.
