import json
import os
import sys
from pathlib import Path
import click
import dspy
from dotenv import load_dotenv

# Add parent directory to path to import evolution modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.prompts.evolve_prompt_section import extract_prompt_section
from evolution.prompts.prompt_selection_evaluator import PromptBehaviorDataset, PromptBehaviorEvaluator

# ── Load environment ──────────────────────────────────────────────────
env_path = Path.home() / ".hermes" / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
if not api_key:
    print("ERROR: No OPENCODE_ZEN_API_KEY found in environment.")
    sys.exit(1)

# Configure DSPy to use OpenCode Zen globally
zen_base_url = "https://opencode.ai/zen/v1"
dspy.configure(
    lm=dspy.LM(
        model="openai/minimax-m3-free",
        api_key=api_key,
        base_url=zen_base_url,
    )
)

@click.command()
@click.option("--section", required=True, type=click.Choice(["MEMORY_GUIDANCE", "SESSION_SEARCH_GUIDANCE", "DEFAULT_AGENT_IDENTITY", "SKILLS_GUIDANCE"]))
def main(section: str):
    dataset_file = Path(f"datasets/prompts/{section}_eval.jsonl")
    if not dataset_file.exists():
        print(f"ERROR: Dataset not found at {dataset_file}")
        print("Please generate prompt behavior datasets first by running:")
        print("  python autoresearch/generate_datasets.py")
        sys.exit(1)

    # Load dataset
    print(f"Loading prompt behavior dataset: {dataset_file}")
    dataset = PromptBehaviorDataset.load(dataset_file)
    print(f"Loaded {dataset.size} examples.")

    # Extract current prompt section
    hermes_path = get_hermes_agent_path()
    print(f"Extracting prompt section '{section}' from prompt_builder.py at {hermes_path}...")
    current_prompt_text = extract_prompt_section(hermes_path, section)
    print(f"Current length: {len(current_prompt_text)} characters")

    # Run evaluation
    config = EvolutionConfig(
        optimizer_model="openai/minimax-m3-free",
        eval_model="openai/minimax-m3-free",
        judge_model="openai/minimax-m3-free",
    )
    evaluator = PromptBehaviorEvaluator(config)

    print("Running evaluation...")
    results = evaluator.evaluate(current_prompt_text, dataset.examples)

    accuracy = results["accuracy"]
    avg_score = results["avg_score"]
    correct = results["correct"]
    total = results["total"]

    print("\n" + "="*50)
    print(f"Phase 3 Evaluation Complete — Section: {section}")
    print(f"Average Behavior Score: {avg_score:.3f}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("="*50)

    # Save results to local folder
    results_file = Path(__file__).parent / "results.json"
    output_data = {
        "section": section,
        "timestamp": time_str(),
        "avg_score": avg_score,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "failures": results["failures"]
    }
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {results_file}")

def time_str():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    main()
