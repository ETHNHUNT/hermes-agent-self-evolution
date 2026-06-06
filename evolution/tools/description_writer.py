"""Description writer for Phase 2A.

Writes evolved tool descriptions to a git branch in the hermes-agent repo.
Follows the deployment pattern: changes go to a branch, then PR for review.
Never hot-swaps into active sessions.
"""

import subprocess
from pathlib import Path
from datetime import datetime


def write_evolved_descriptions(
    hermes_agent_path: Path,
    evolved: dict[str, str],
    branch_name: str = None,
) -> str:
    """Write evolved descriptions to a new git branch.

    Args:
        hermes_agent_path: Path to hermes-agent repo
        evolved: Dict of tool_name -> new description
        branch_name: Git branch name (auto-generated if None)

    Returns:
        Branch name created
    """
    if branch_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"evolve/tool-descriptions-{ts}"

    # Create branch
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=hermes_agent_path,
        capture_output=True,
    )

    # TODO: Patch the actual description constants in the source files
    # For now, write a summary file
    summary_path = hermes_agent_path / "EVOLVED_DESCRIPTIONS.md"
    lines = ["# Evolved Tool Descriptions\n", f"Generated: {datetime.now().isoformat()}\n\n"]
    for tool_name, desc in sorted(evolved.items()):
        lines.append(f"## {tool_name}\n")
        lines.append(f"({len(desc)} chars)\n\n")
        lines.append(f"{desc}\n\n")

    summary_path.write_text("".join(lines))

    # Commit
    subprocess.run(["git", "add", "."], cwd=hermes_agent_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"evolve: update tool descriptions ({branch_name})"],
        cwd=hermes_agent_path,
        capture_output=True,
    )

    return branch_name
