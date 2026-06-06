"""Schema loader for Phase 2A.

Reads current tool descriptions from hermes-agent source code.
Tools are registered via registry.register() and their descriptions
are in the schema dicts within tools/*.py files.
"""

import re
import ast
from pathlib import Path


# The 6 dev tools we're optimizing in Phase 2A
DEV_TOOLS = ["search_files", "read_file", "write_file", "patch", "terminal", "process"]

# Known description constants in hermes-agent
DESCRIPTION_CONSTANTS = {
    "terminal": "TERMINAL_TOOL_DESCRIPTION",
}


def load_tool_descriptions(hermes_agent_path: Path) -> dict:
    """Load current tool descriptions from hermes-agent source.

    Args:
        hermes_agent_path: Path to hermes-agent repo

    Returns:
        Dict mapping tool_name -> description text
    """
    descriptions = {}
    tools_dir = hermes_agent_path / "hermes_agent" / "tools"

    if not tools_dir.exists():
        # Try alternate structure
        tools_dir = hermes_agent_path / "tools"

    if not tools_dir.exists():
        raise FileNotFoundError(f"Cannot find tools directory in {hermes_agent_path}")

    # Scan all Python files in tools directory
    for py_file in sorted(tools_dir.rglob("*.py")):
        content = py_file.read_text()

        # Look for description constants
        for tool_name in DEV_TOOLS:
            if tool_name in descriptions:
                continue

            # Pattern 1: DESCRIPTION = "..." or DESCRIPTION = '...'
            const_name = DESCRIPTION_CONSTANTS.get(tool_name, f"{tool_name.upper()}_DESCRIPTION")
            pattern = rf'{const_name}\s*=\s*["\']([^"\']+)["\']'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                descriptions[tool_name] = match.group(1).strip()
                continue

            # Pattern 2: "description": "..." in schema dicts
            pattern = rf'"description"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern, content)
            for m in matches:
                # Check if this file is about this tool
                if tool_name.replace("_", "") in py_file.stem.lower() or \
                   tool_name.lower() in content.lower()[:500]:
                    if tool_name not in descriptions:
                        descriptions[tool_name] = m.strip()
                    break

    return descriptions


def get_default_descriptions() -> dict:
    """Return the current default descriptions for the 6 dev tools.

    These are the actual descriptions from hermes-agent as of the
    current version. Used as fallback if source parsing fails.
    """
    return {
        "search_files": (
            "Search for files by name pattern or content across directories. "
            "Use for finding files by glob pattern, searching file contents with regex, "
            "or locating code. Preferred over terminal(grep/find) for file operations. "
            "Returns matching file paths and line numbers."
        ),
        "read_file": (
            "Read the contents of a file with optional line range. "
            "Use when you need to see what's in a specific file. "
            "Supports offset and limit for large files. "
            "Preferred over terminal(cat/head/tail) for reading files."
        ),
        "write_file": (
            "Write content to a file, creating it if it doesn't exist. "
            "Use for creating new files or overwriting existing ones. "
            "Creates parent directories automatically. "
            "For small edits to existing files, prefer patch."
        ),
        "patch": (
            "Make targeted edits to existing files using find-and-replace. "
            "Use for small changes like fixing typos, updating values, or modifying "
            "specific lines. More precise than write_file for small edits. "
            "Cannot create new files — use write_file for that."
        ),
        "terminal": (
            "Execute shell commands on the local system. "
            "Use for running programs, git operations, package installation, "
            "system commands, and anything that needs a shell. "
            "For file operations, prefer the dedicated file tools."
        ),
        "process": (
            "Manage background processes started with terminal. "
            "Use for monitoring, waiting on, or killing long-running processes. "
            "Can poll, log, wait, or terminate background tasks. "
            "Not for starting new commands — use terminal for that."
        ),
    }


def load_descriptions(hermes_agent_path: Path = None) -> dict:
    """Load tool descriptions, falling back to defaults if needed."""
    defaults = get_default_descriptions()

    if hermes_agent_path:
        try:
            loaded = load_tool_descriptions(Path(hermes_agent_path))
            # Merge: use loaded where available, default for rest
            merged = dict(defaults)
            merged.update(loaded)
            return merged
        except FileNotFoundError:
            pass

    return defaults


if __name__ == "__main__":
    import json
    from evolution.core.config import get_hermes_agent_path

    try:
        path = get_hermes_agent_path()
        print(f"Loading from: {path}")
        descs = load_descriptions(path)
    except Exception as e:
        print(f"Using defaults (error: {e})")
        descs = get_default_descriptions()

    for name, desc in descs.items():
        print(f"\n{name}: {len(desc)} chars")
        print(f"  {desc[:120]}...")
