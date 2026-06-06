import os
import sys
import json
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

# Add parent directory to path to import evolution modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from evolution.core.config import get_hermes_agent_path
from evolution.prompts.evolve_prompt_section import extract_prompt_section
from evolution.prompts.prompt_selection_evaluator import PromptBehaviorExample, PromptBehaviorDataset

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

def generate_scenarios_direct(section_name: str, original_text: str, num_scenarios: int = 20) -> list[dict]:
    prompt = (
        "You are generating test scenarios for evaluating an AI agent's system prompt guidelines.\n\n"
        f"Prompt Section Name: {section_name}\n"
        f"Guideline Text:\n\"\"\"\n{original_text}\n\"\"\"\n\n"
        f"Generate exactly {num_scenarios} diverse, realistic dialogue/task scenarios that test whether the agent adheres to these rules.\n"
        "Guidelines:\n"
        "- Generate some scenarios where the agent SHOULD invoke the tool (positive cases).\n"
        "- Generate some scenarios where the agent SHOULD NOT invoke the tool, or should do something else (negative cases).\n"
        "- Include a specific rubric (expected_behavior_rubric) that describes how the agent should behave and what to check.\n\n"
        "Your response MUST be a JSON array of objects, and nothing else. Do not wrap in markdown code blocks. format:\n"
        "[\n"
        "  {\n"
        '    "conversation_context": "<user request and surrounding context>",\n'
        '    "correct_tool": "<correct tool name or none>",\n'
        '    "rubric": "<detailed rubric to check behavior>",\n'
        '    "difficulty": "medium",\n'
        '    "category": "general"\n'
        "  }\n"
        "]"
    )

    for attempt in range(3):
        try:
            print(f"  Attempt {attempt + 1}/3 to generate JSON scenarios via LLM...")
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.3,
                },
                timeout=120,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()

                # Clean thinking tags
                if "<think>" in content:
                    if "</think>" in content:
                        content = content.split("</think>", 1)[1].strip()
                    else:
                        content = content.replace("<think>", "").strip()

                # Clean code blocks
                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[-1].strip() == "```":
                        content = "\n".join(lines[1:-1])
                    else:
                        content = "\n".join(lines[1:])

                # Extract JSON array
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1 and end > start:
                    content = content[start:end+1]

                # Attempt to parse
                parsed = json.loads(content)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            else:
                print(f"  HTTP error: {r.status_code} - {r.text[:100]}")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
        time.sleep(2)

    raise ValueError(f"Failed to generate valid JSON scenarios for {section_name} after 3 attempts.")

def main():
    sections = ["MEMORY_GUIDANCE", "SESSION_SEARCH_GUIDANCE", "DEFAULT_AGENT_IDENTITY", "SKILLS_GUIDANCE"]
    hermes_path = get_hermes_agent_path()
    
    # Create datasets/prompts directory
    output_dir = Path("datasets/prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting robust generation of prompt behavior datasets...")
    for section in sections:
        output_file = output_dir / f"{section}_eval.jsonl"
        
        # If dataset already exists, skip it (unless it is empty)
        if output_file.exists() and output_file.stat().st_size > 100:
            print(f"\nDataset already exists for {section} at {output_file} — skipping.")
            continue
            
        try:
            print(f"\nProcessing section: {section}")
            original_text = extract_prompt_section(hermes_path, section)
            
            # Generate scenarios
            scenarios = generate_scenarios_direct(section, original_text, num_scenarios=20)
            
            dataset = PromptBehaviorDataset()
            for item in scenarios:
                dataset.add(PromptBehaviorExample(
                    conversation_context=item.get("conversation_context", ""),
                    correct_tool=item.get("correct_tool", "none"),
                    rubric=item.get("rubric", ""),
                    difficulty=item.get("difficulty", "medium"),
                    category=section.lower(),
                    source="synthetic"
                ))
            
            dataset.save(output_file)
            print(f"Successfully generated and saved {dataset.size} scenarios to {output_file}")
        except Exception as e:
            print(f"ERROR generating dataset for {section}: {e}")

if __name__ == "__main__":
    main()
