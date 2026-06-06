"""Run baseline evaluation of current tool descriptions against the devtools dataset.

Uses OpenCode Zen's minimax-m3-free directly via requests (bypasses DSPy/litellm
which can't route to OpenCode Zen's bare model name).

Usage:
    python3 baseline_eval.py
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Setup ────────────────────────────────────────────────────────────
env_path = Path.home() / ".hermes" / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")

if not api_key:
    print("ERROR: No OPENCODE_ZEN_API_KEY found in env")
    sys.exit(1)

BASE_URL = "https://opencode.ai/zen/v1"
MODEL = "minimax-m3-free"


def call_llm(task: str, tools_json: str, max_retries: int = 3) -> tuple[str, str]:
    """Call OpenCode Zen directly. Returns (selected_tool, reasoning)."""
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
                content = r.json()["choices"][0]["message"]["content"]
                content = content.strip()

                # Strip <think>...</think> reasoning prefix (MiniMax M3 emits this)
                if "<think>" in content:
                    if "</think>" in content:
                        content = content.split("</think>", 1)[1].strip()
                    else:
                        # Incomplete — just strip the tag
                        content = content.replace("<think>", "").strip()

                # Strip markdown fence if present
                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[-1].strip() == "```":
                        content = "\n".join(lines[1:-1])
                    else:
                        content = "\n".join(lines[1:])

                # Find JSON object in content
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    content = content[start : end + 1]

                parsed = json.loads(content)
                return parsed.get("selected_tool", "").strip(), parsed.get("reasoning", "").strip()
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:100]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
        time.sleep(2)

    return "", f"FAILED: {last_err}"


# ── Load data ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from evolution.core.config import get_hermes_agent_path  # noqa: E402
from evolution.tools.evolve_tool_descriptions import extract_tool_descriptions  # noqa: E402

# Support custom dataset path as CLI argument
if len(sys.argv) > 1:
    dataset_path = Path(sys.argv[1])
else:
    dataset_path = Path("datasets/tools/devtools_v1.jsonl")

examples = []
with open(dataset_path) as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples from {dataset_path}")

hermes_path = get_hermes_agent_path()
descs = extract_tool_descriptions(hermes_path)
if not descs:
    from evolution.tools.schema_loader import load_descriptions
    descs = load_descriptions(hermes_path)

print(f"Loaded {len(descs)} tool descriptions")

# ── Evaluate ─────────────────────────────────────────────────────────
correct = 0
total = 0
per_tool = {}
failures = []

# Global fallback tools list
tools_list = [{"name": n, "description": d} for n, d in descs.items()]
tools_list.append(
    {
        "name": "NO_TOOL",
        "description": "Use this when the task doesn't require any tool — just respond directly.",
    }
)
tools_json = json.dumps(tools_list, indent=2)

start_time = time.time()

for i, ex in enumerate(examples):
    task = ex.get("task_description") or ex.get("task_input")
    expected = ex["correct_tool"]
    if isinstance(expected, list):
        expected = expected[0] if expected else "NO_TOOL"

    # Build tools JSON context dynamically for this example if tools_available is present
    available = ex.get("tools_available")
    if available:
        task_tools_list = []
        for name in available:
            desc = descs.get(name, f"The {name} tool.")
            task_tools_list.append({"name": name, "description": desc})
        task_tools_list.append(
            {
                "name": "NO_TOOL",
                "description": "Use this when the task doesn't require any tool — just respond directly.",
            }
        )
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
    if is_correct:
        correct += 1
        per_tool[expected]["correct"] += 1
    else:
        failures.append(
            {
                "task": task[:100],
                "expected": expected,
                "got": selected,
                "reasoning": reasoning[:200],
            }
        )

    if (i + 1) % 5 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta_sec = (len(examples) - i - 1) / rate if rate > 0 else 0
        print(
            f"  Progress: {i+1}/{len(examples)} (acc: {correct/total:.1%}, "
            f"{rate:.1f}/s, ETA: {eta_sec:.0f}s)",
            flush=True,
        )

# ── Results ──────────────────────────────────────────────────────────
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"Baseline Evaluation Results (model: {MODEL})")
print(f"{'='*60}")
print(f"Accuracy: {correct/total:.1%} ({correct}/{total})")
print(f"Time: {elapsed:.1f}s ({(elapsed/total):.1f}s/example)")
print(f"\nPer-Tool:")
for tool, stats in sorted(per_tool.items()):
    acc = stats["correct"] / max(1, stats["total"])
    print(f"  {tool:20s}  {acc:.1%} ({stats['correct']}/{stats['total']})")

print(f"\nTop 20 Failures:")
for f in failures[:20]:
    print(f"  {f['task'][:70]}")
    print(f"    expected={f['expected']} got={f['got']}")

# Save results
output = {
    "model": MODEL,
    "accuracy": correct / max(1, total),
    "correct": correct,
    "total": total,
    "elapsed_sec": elapsed,
    "per_tool": {k: v for k, v in per_tool.items()},
    "failures": failures,
}
out_name = f"baseline_eval_{dataset_path.stem}.json"
out_path = Path("output/tools") / out_name
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
