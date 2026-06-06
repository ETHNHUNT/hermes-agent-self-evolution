import json
import os
import sys
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────
env_path = Path.home() / ".hermes" / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
if not api_key:
    print("ERROR: No OPENCODE_ZEN_API_KEY found in environment.")
    sys.exit(1)

BASE_URL = "https://opencode.ai/zen/v1"
MODEL = "minimax-m3-free"
DATASET_PATH = Path("datasets/tools/devtools_v1.jsonl")

# Add parent directory to path to import evolution modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from evolution.core.config import get_hermes_agent_path
from evolution.tools.evolve_tool_descriptions import extract_tool_descriptions

def call_llm(task: str, tools_json: str, max_retries: int = 3) -> tuple[str, str]:
    prompt = (
        "You are selecting the right tool for a task.\n\n"
        f"Available tools:\n{tools_json}\n\n"
        f"Task: {task}\n\n"
        "Reply with JSON in this exact format:\n"
        '{"selected_tool": "<tool_name>", "reasoning": "<one sentence>"}'
    )

    last_err = ""
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()

                # Strip <think> reasoning
                if "<think>" in content:
                    if "</think>" in content:
                        content = content.split("</think>", 1)[1].strip()
                    else:
                        content = content.replace("<think>", "").strip()

                # Strip markdown fence
                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[-1].strip() == "```":
                        content = "\n".join(lines[1:-1])
                    else:
                        content = "\n".join(lines[1:])

                # Find JSON object
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    content = content[start:end+1]

                parsed = json.loads(content)
                return parsed.get("selected_tool", "").strip(), parsed.get("reasoning", "").strip()
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:100]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
        time.sleep(1)

    return "", f"FAILED: {last_err}"

def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    # Load examples
    examples = []
    with open(DATASET_PATH) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {DATASET_PATH}")

    # Load descriptions
    hermes_path = get_hermes_agent_path()
    print(f"Loading tool descriptions from hermes-agent: {hermes_path}")
    descs = extract_tool_descriptions(hermes_path)
    if not descs:
        print("ERROR: Could not load tool descriptions.")
        sys.exit(1)
    print(f"Loaded {len(descs)} tool descriptions.")

    # Prepare global list
    tools_list = [{"name": n, "description": d} for n, d in descs.items()]
    tools_list.append({
        "name": "NO_TOOL",
        "description": "Use this when the task doesn't require any tool — just respond directly.",
    })
    tools_json = json.dumps(tools_list, indent=2)

    correct = 0
    total = 0
    per_tool = {}
    failures = []
    successes = []

    start_time = time.time()

    for i, ex in enumerate(examples):
        task = ex.get("task_description") or ex.get("task_input")
        expected = ex["correct_tool"]
        if isinstance(expected, list):
            expected = expected[0] if expected else "NO_TOOL"

        # Dynamically build tools list if available on the example
        available = ex.get("tools_available")
        if available:
            task_tools_list = []
            for name in available:
                desc = descs.get(name, f"The {name} tool.")
                task_tools_list.append({"name": name, "description": desc})
            task_tools_list.append({
                "name": "NO_TOOL",
                "description": "Use this when the task doesn't require any tool — just respond directly.",
            })
            current_tools_json = json.dumps(task_tools_list, indent=2)
        else:
            current_tools_json = tools_json

        selected, reasoning = call_llm(task, current_tools_json)

        # Normalize
        if selected.lower() in ("no_tool", "none", "no tool"):
            selected = "NO_TOOL"

        is_correct = (selected == expected)
        total += 1

        if expected not in per_tool:
            per_tool[expected] = {"correct": 0, "total": 0}
        per_tool[expected]["total"] += 1

        res_item = {
            "task": task,
            "expected": expected,
            "selected": selected,
            "reasoning": reasoning,
            "correct": is_correct
        }

        if is_correct:
            correct += 1
            per_tool[expected]["correct"] += 1
            successes.append(res_item)
        else:
            failures.append(res_item)

        if (i + 1) % 5 == 0 or (i + 1) == len(examples):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"Progress: {i+1}/{len(examples)} | Current Accuracy: {correct/total:.1%} | Rate: {rate:.1f} ex/s", flush=True)

    elapsed = time.time() - start_time
    accuracy = correct / total

    print("\n" + "="*50)
    print(f"Phase 2 Evaluation Complete")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Time elapsed: {elapsed:.1f}s")
    print("="*50)

    # Save results to local folder
    results_file = Path(__file__).parent / "results.json"
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_tool": per_tool,
        "failures": failures,
        "successes": successes
    }
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
