# Research Program: Tool Description Optimization (Phase 2)

Optimize the natural language descriptions of tools in the `hermes-agent` repository so that the agent selects the correct tool more reliably for user tasks.

---

## Objective
Maximize tool selection accuracy on the combined evaluation dataset (341 examples from handcrafted + synthetic sources).

## Target Metric
- **primary_metric**: `accuracy` (percentage of correct tool selections).
- **secondary_metric**: `description_length` (keep descriptions as concise as possible; total characters for any modified description must be under 500 characters).

## Workspace Configuration
- **Modifiable Files**: Python tool definition files inside the `hermes-agent` tools directory: `/Users/vipinnandal/.hermes/hermes-agent/tools/` (e.g. `file_tools.py`, `terminal_tool.py`, etc.). Only edit the string literal descriptions inside the tool schemas/constants. Do NOT modify tool schemas, parameters, types, or functional code.
- **Read-Only Files**: 
  - Evaluation dataset: `datasets/tools/combined_eval.jsonl`
  - Evaluation script: `autoresearch/phase2_tool_descriptions/evaluate.py`

## How to Evaluate
Run the following command in the `hermes-agent-self-evolution` directory:
```bash
python autoresearch/phase2_tool_descriptions/evaluate.py
```

This script will run the combined evaluation dataset (341 examples) through the tool selector model and print the overall accuracy. It also writes the results and failures details to:
`autoresearch/phase2_tool_descriptions/results.json`

## Research loop (Step-by-Step)
1. **Establish Baseline**: Run `python autoresearch/phase2_tool_descriptions/evaluate.py` to check the current accuracy.
2. **Identify Failures**: Read `autoresearch/phase2_tool_descriptions/results.json` to analyze which tasks failed, what tool was expected, and which tool was incorrectly selected.
3. **Formulate Hypothesis**: For the most frequent failures/confusion pairs, determine how to clarify the description of the correct tool, or add negative guidelines to the description of the incorrectly selected tool.
4. **Apply Changes**: Edit the relevant tool files in `/Users/vipinnandal/.hermes/hermes-agent/tools/`.
5. **Re-evaluate**: Run `python autoresearch/phase2_tool_descriptions/evaluate.py` to test your changes.
6. **Apply Ratchet (Commit or Revert)**:
   - **If accuracy is HIGHER**: Commit the changes in the `hermes-agent` repository:
     ```bash
     cd /Users/vipinnandal/.hermes/hermes-agent
     git add tools/
     git commit -m "autoresearch: improved tool selection accuracy to X.X%"
     cd -
     ```
   - **If accuracy is LOWER or EQUAL**: Revert the changes to keep the repository clean:
     ```bash
     cd /Users/vipinnandal/.hermes/hermes-agent
     git checkout -- tools/
     cd -
     ```
7. **Iterate**: Repeat steps 2-6 to continuously improve.
