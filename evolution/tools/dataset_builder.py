"""Phase 2A dataset builder for dev tool description optimization.

Builds a 150-example dataset for the 6 core dev tools:
  - search_files, read_file, write_file, patch, terminal, process

Data sources:
  1. Hand-crafted confusion pairs (highest value)
  2. Synthetic generation via LLM
  3. Real session trace mining from SessionDB
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── 6 Dev Tools for Phase 2A ──────────────────────────────────────────
DEV_TOOLS = ["search_files", "read_file", "write_file", "patch", "terminal", "process"]

# ── Hand-crafted confusion pairs ──────────────────────────────────────
# These are the highest-value examples — they target known misselection patterns
# between the 6 dev tools.

HANDCRAFTED = [
    # === search_files examples ===
    {"task": "Find all Python files that import the 'os' module in this project", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Search for the string 'TODO' across all files in the repository", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Locate all files containing the word 'deprecated' in the src directory", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Find every file that references 'MAX_RETRIES' in the codebase", "tool": "search_files", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Search for function definitions of 'handle_error' across all Python files", "tool": "search_files", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Find all JSON files that contain a 'version' field", "tool": "search_files", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Which files mention 'GEPA' in the hermes-agent-self-evolution repo?", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Search for 'console.print' in all Python files", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},

    # === search_files vs terminal(grep) confusions ===
    {"task": "Grep for 'import json' in all .py files", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Use grep to find 'TODO' in Python files", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Run grep -r 'deprecated' in the project", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Find all occurrences of 'config' using grep", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Search the codebase for 'raise Exception' with grep", "tool": "search_files", "category": "confusion", "difficulty": "hard"},

    # === read_file examples ===
    {"task": "Show me the contents of README.md", "tool": "read_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Read the first 50 lines of evolution/core/config.py", "tool": "read_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Display lines 100-150 of the main config file", "tool": "read_file", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "What's in the .env file in the project root?", "tool": "read_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Read the PLAN.md file", "tool": "read_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Show me the imports at the top of evolve_tool_descriptions.py", "tool": "read_file", "category": "synthetic_positive", "difficulty": "medium"},

    # === read_file vs terminal(cat) confusions ===
    {"task": "Cat the contents of README.md", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Run 'cat config.py' to see the file", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Use head -20 to show the start of the file", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Display the file using terminal cat command", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Pipe the file contents through cat", "tool": "read_file", "category": "confusion", "difficulty": "hard"},

    # === read_file vs search_files confusions ===
    {"task": "Find the line that has 'def main' in app.py", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Look for the API key in config.yaml", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Show me where MAX_LIMIT is set in the config", "tool": "read_file", "category": "confusion", "difficulty": "hard"},

    # === write_file examples ===
    {"task": "Create a new file called output.txt with the results", "tool": "write_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Write the generated report to report.md", "tool": "write_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Save the JSON results to output/data.json", "tool": "write_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Create a new Python script at scripts/analyze.py", "tool": "write_file", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Write a summary of findings to SUMMARY.md", "tool": "write_file", "category": "synthetic_positive", "difficulty": "easy"},

    # === write_file vs terminal confusions ===
    {"task": "Echo the results into a file called output.txt", "tool": "write_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Use 'echo > file.txt' to create the output", "tool": "write_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Redirect output to a file using tee", "tool": "write_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Append text to a file using >>", "tool": "write_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Use printf to write content to a file", "tool": "write_file", "category": "confusion", "difficulty": "hard"},

    # === patch examples ===
    {"task": "Fix the typo in line 42 of config.py: 'teh' should be 'the'", "tool": "patch", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Update the version number in setup.py from 1.0 to 2.0", "tool": "patch", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Replace 'localhost' with '127.0.0.1' in the config file", "tool": "patch", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Remove the debug print statement from line 156", "tool": "patch", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Change the function signature to add a new parameter", "tool": "patch", "category": "synthetic_positive", "difficulty": "medium"},

    # === patch vs read_file confusions ===
    {"task": "Read the file and tell me what to change on line 42", "tool": "patch", "category": "confusion", "difficulty": "hard"},
    {"task": "Show me the current value of the version field", "tool": "patch", "category": "confusion", "difficulty": "hard"},
    {"task": "Find the line with the typo so I can fix it", "tool": "patch", "category": "confusion", "difficulty": "hard"},

    # === patch vs write_file confusions ===
    {"task": "Overwrite the entire config file with the corrected version", "tool": "patch", "category": "confusion", "difficulty": "hard"},
    {"task": "Rewrite the whole file with the fix applied", "tool": "patch", "category": "confusion", "difficulty": "hard"},
    {"task": "Save a new version of the file with changes", "tool": "patch", "category": "confusion", "difficulty": "hard"},

    # === terminal examples ===
    {"task": "Run pytest to execute the test suite", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Execute 'git status' to check the repository state", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Run 'npm install' to install dependencies", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Start the development server with 'python -m http.server'", "tool": "terminal", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Run the build script: 'make build'", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Execute 'docker ps' to list running containers", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Run 'ls -la' to list all files with details", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Check the Python version with 'python --version'", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},

    # === terminal vs search_files confusions ===
    {"task": "Run 'grep -r TODO .' to find all TODOs", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Use find to locate all .py files", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Execute 'find . -name *.json' to find JSON files", "tool": "search_files", "category": "confusion", "difficulty": "hard"},

    # === terminal vs write_file confusions ===
    {"task": "Use 'echo hello > file.txt' to create a file", "tool": "write_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Run 'cat > file.txt << EOF' to write content", "tool": "write_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Pipe output to a file with 'cmd > output.txt'", "tool": "write_file", "category": "confusion", "difficulty": "hard"},

    # === process examples ===
    {"task": "Check if the background server process is still running", "tool": "process", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Wait for the build process to complete and show output", "tool": "process", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Kill the background process with ID 12345", "tool": "process", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "List all background processes", "tool": "process", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Poll the long-running task and report progress", "tool": "process", "category": "synthetic_positive", "difficulty": "medium"},

    # === process vs terminal confusions ===
    {"task": "Run 'ps aux' to see running processes", "tool": "process", "category": "confusion", "difficulty": "hard"},
    {"task": "Execute 'kill -9 12345' to stop a process", "tool": "process", "category": "confusion", "difficulty": "hard"},
    {"task": "Use 'top' to monitor system processes", "tool": "process", "category": "confusion", "difficulty": "hard"},
    {"task": "Run 'bg' to background a task", "tool": "process", "category": "confusion", "difficulty": "hard"},

    # === No-tool examples (efficiency penalty) ===
    {"task": "What's the difference between search_files and grep?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "Explain how the patch tool works", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "What tools are available for file operations?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "Should I use read_file or terminal(cat) for small files?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "medium"},
    {"task": "List all the tools you have", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "What's the weather like today?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "Tell me a joke", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "What's 2 + 2?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "Explain what GEPA optimization does", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "medium"},
    {"task": "How do I install Python packages?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "medium"},

    # === Additional search_files examples ===
    {"task": "Find all test files in the project", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Locate every file that imports 'dspy'", "tool": "search_files", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Search for 'TODO' or 'FIXME' comments in all source files", "tool": "search_files", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Find all YAML files in the project", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Which files contain 'EvolutionConfig'?", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Search for 'batch_runner' across the codebase", "tool": "search_files", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Find all files with 'import os' in the src directory", "tool": "search_files", "category": "synthetic_positive", "difficulty": "easy"},

    # === Additional read_file examples ===
    {"task": "Read the pyproject.toml to see dependencies", "tool": "read_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Show me the contents of the test file for config", "tool": "read_file", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Read lines 1-30 of the main entry point", "tool": "read_file", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "What does the .gitignore file contain?", "tool": "read_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Read the evolution/core/fitness.py file", "tool": "read_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Show me the last 20 lines of the log file", "tool": "read_file", "category": "synthetic_positive", "difficulty": "medium"},

    # === Additional write_file examples ===
    {"task": "Create a new .env file with default settings", "tool": "write_file", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Write the test results to results.json", "tool": "write_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Save the configuration to config.yaml", "tool": "write_file", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Create a new directory structure and placeholder files", "tool": "write_file", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Write a Python script that prints Hello World to hello.py", "tool": "write_file", "category": "synthetic_positive", "difficulty": "easy"},

    # === Additional patch examples ===
    {"task": "Fix the indentation error on line 23 of utils.py", "tool": "patch", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Update the timeout value from 30 to 60 in config.py", "tool": "patch", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Rename the function 'old_name' to 'new_name' in the file", "tool": "patch", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Add a missing import statement at the top of the file", "tool": "patch", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Remove the commented-out code block from lines 45-50", "tool": "patch", "category": "synthetic_positive", "difficulty": "medium"},

    # === Additional terminal examples ===
    {"task": "Run 'git log --oneline -10' to see recent commits", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Execute 'pip list' to see installed packages", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Run the linter: 'ruff check .'", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Execute 'chmod +x script.sh' to make it executable", "tool": "terminal", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Run 'git diff' to see uncommitted changes", "tool": "terminal", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Execute 'python -m pytest tests/ -v' to run tests with verbose output", "tool": "terminal", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Run 'curl https://api.example.com/health' to check API status", "tool": "terminal", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Execute 'tar -xzf archive.tar.gz' to extract files", "tool": "terminal", "category": "synthetic_positive", "difficulty": "medium"},

    # === Additional process examples ===
    {"task": "Show me the output of the background training job", "tool": "process", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Wait for the deployment to finish and show the result", "tool": "process", "category": "synthetic_positive", "difficulty": "medium"},
    {"task": "Check the status of all my background tasks", "tool": "process", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Kill the stuck build process", "tool": "process", "category": "synthetic_positive", "difficulty": "easy"},
    {"task": "Monitor the progress of the data processing job", "tool": "process", "category": "synthetic_positive", "difficulty": "medium"},

    # === Additional confusion pairs ===
    {"task": "Use find to locate all .json files in the project", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Run 'wc -l' to count lines in all Python files", "tool": "search_files", "category": "confusion", "difficulty": "hard"},
    {"task": "Use 'tail -f' to watch a log file in real time", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Cat the error log to see what went wrong", "tool": "read_file", "category": "confusion", "difficulty": "hard"},
    {"task": "Use sed to replace a string in a file", "tool": "patch", "category": "confusion", "difficulty": "hard"},
    {"task": "Run 'awk' to extract the second column from a CSV", "tool": "terminal", "category": "confusion", "difficulty": "hard"},
    {"task": "Use 'curl' to download a file from the internet", "tool": "terminal", "category": "confusion", "difficulty": "hard"},
    {"task": "Run 'jq' to parse a JSON file", "tool": "terminal", "category": "confusion", "difficulty": "hard"},
    {"task": "Use 'xargs' to process a list of files", "tool": "terminal", "category": "confusion", "difficulty": "hard"},
    {"task": "Pipe the output of find into grep", "tool": "search_files", "category": "confusion", "difficulty": "hard"},

    # === Additional no-tool examples ===
    {"task": "What's the difference between write_file and patch?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "When should I use terminal vs process?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "medium"},
    {"task": "Can you help me understand this error message: 'SyntaxError: invalid syntax'?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "What's your favorite programming language?", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "easy"},
    {"task": "Summarize the git workflow", "tool": "NO_TOOL", "category": "no_tool", "difficulty": "medium"},
]


@dataclass
class DevToolExample:
    """A single tool selection test case for Phase 2A."""
    task_description: str
    correct_tool: str
    difficulty: str = "medium"
    category: str = "general"
    source: str = "handcrafted"
    params: Optional[dict] = None  # Expected parameters (for param correctness scoring)

    def to_dict(self) -> dict:
        d = {
            "task_description": self.task_description,
            "correct_tool": self.correct_tool,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }
        if self.params:
            d["params"] = self.params
        return d

    def to_json(self) -> str:
        d = self.to_dict()
        # Fix: to_dict returns task_description directly, let's use a proper dict
        return json.dumps({
            "task_description": self.task_description,
            "correct_tool": self.correct_tool,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        })


class DevToolDatasetBuilder:
    """Builds the Phase 2A evaluation dataset for dev tools.

    Combines:
    1. Hand-crafted confusion pairs (highest value)
    2. Synthetic generation via LLM
    3. Real session trace mining
    """

    def __init__(self, output_dir: str = "datasets/tools"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples: list[DevToolExample] = []

    def build_handcrafted(self) -> list[DevToolExample]:
        """Build the hand-crafted dataset."""
        examples = []
        for item in HANDCRAFTED:
            examples.append(DevToolExample(
                task_description=item["task"],
                correct_tool=item["tool"],
                difficulty=item.get("difficulty", "medium"),
                category=item.get("category", "general"),
                source="handcrafted",
            ))
        return examples

    def build_synthetic(self, llm_call_fn, num_per_tool: int = 5) -> list[DevToolExample]:
        """Generate synthetic examples using an LLM.

        Args:
            llm_call_fn: Function that takes a prompt and returns LLM response text
            num_per_tool: Number of synthetic examples per tool
        """
        examples = []
        tool_descriptions = {
            "search_files": "Search for files by name pattern or content across directories. Use for finding files, grepping content, locating code.",
            "read_file": "Read the contents of a file. Use when you need to see what's in a specific file.",
            "write_file": "Write content to a file. Use when creating new files or overwriting existing ones.",
            "patch": "Make targeted edits to existing files. Use for small changes like fixing typos, updating values, or modifying specific lines.",
            "terminal": "Execute shell commands. Use for running programs, git operations, package installation, and system commands.",
            "process": "Manage background processes. Use for monitoring, waiting on, or killing long-running processes.",
        }

        prompt = f"""Generate {num_per_tool} realistic user tasks for EACH of the following 6 tools.
For each task, the specified tool should be the BEST choice.

Tools:
{json.dumps(tool_descriptions, indent=2)}

Rules:
- Each task should be something a real user/developer would ask
- Tasks should clearly favor the specified tool over the other 5
- Vary difficulty (easy/medium/hard)
- Include some tasks that mention terminal commands but should use a file tool instead
- Include some "no_tool" tasks where the agent should just respond directly

Output JSON array:
[{{"task": "...", "correct_tool": "...", "difficulty": "easy|medium|hard", "category": "synthetic_positive|confusion|no_tool"}}]

Generate exactly {num_per_tool} tasks per tool (6 tools × {num_per_tool} = {6 * num_per_tool} total) plus 10 no-tool tasks."""

        try:
            response = llm_call_fn(prompt)
            # Parse JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            tasks = json.loads(json_str)
            for t in tasks:
                if t.get("task") and t.get("correct_tool"):
                    examples.append(DevToolExample(
                        task_description=t["task"],
                        correct_tool=t["correct_tool"],
                        difficulty=t.get("difficulty", "medium"),
                        category=t.get("category", "synthetic"),
                        source="synthetic",
                    ))
        except Exception as e:
            print(f"Synthetic generation failed: {e}")

        return examples

    def build_full_dataset(self, include_synthetic: bool = False, llm_call_fn=None) -> list[DevToolExample]:
        """Build the complete Phase 2A dataset.

        Args:
            include_synthetic: Whether to include LLM-generated examples
            llm_call_fn: LLM call function (required if include_synthetic=True)

        Returns:
            List of DevToolExample objects
        """
        # Start with hand-crafted (104 examples)
        self.examples = self.build_handcrafted()

        # Optionally add synthetic
        if include_synthetic and llm_call_fn:
            synthetic = self.build_synthetic(llm_call_fn, num_per_tool=8)
            self.examples.extend(synthetic)

        return self.examples

    def save_jsonl(self, filename: str = "devtools_v1.jsonl"):
        """Save dataset to JSONL file."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps({
                    "task_description": ex.task_description,
                    "correct_tool": ex.correct_tool,
                    "difficulty": ex.difficulty,
                    "category": ex.category,
                    "source": ex.source,
                }) + "\n")
        print(f"Saved {len(self.examples)} examples to {path}")
        return path

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        stats = {"total": len(self.examples), "tools": {}, "categories": {}, "sources": {}, "difficulties": {}}
        for ex in self.examples:
            stats["tools"][ex.correct_tool] = stats["tools"].get(ex.correct_tool, 0) + 1
            stats["categories"][ex.category] = stats["categories"].get(ex.category, 0) + 1
            stats["sources"][ex.source] = stats["sources"].get(ex.source, 0) + 1
            stats["difficulties"][ex.difficulty] = stats["difficulties"].get(ex.difficulty, 0) + 1
        return stats

    def split(self, train_ratio=0.5, val_ratio=0.25):
        """Split into train/val/holdout."""
        shuffled = list(self.examples)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        return (
            shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:],
        )


if __name__ == "__main__":
    builder = DevToolDatasetBuilder()
    examples = builder.build_full_dataset(include_synthetic=False)
    path = builder.save_jsonl("devtools_v1.jsonl")
    stats = builder.get_stats()
    print(f"\nDataset stats: {json.dumps(stats, indent=2)}")
    train, val, holdout = builder.split()
    print(f"\nSplits: train={len(train)}, val={len(val)}, holdout={len(holdout)}")
