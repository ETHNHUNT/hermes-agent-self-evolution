"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path
    hermes_agent_path: Path = field(default_factory=lambda: get_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Model for GEPA reflections
    eval_model: str = "openai/gpt-4.1-mini"  # Model for LLM-as-judge scoring
    judge_model: str = "openai/gpt-4.1"  # Model for dataset generation

    # Constraints
    max_skill_size: int = 15_000  # 15KB default
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )


def setup_dspy_lm(model_name: str):
    """Configures DSPy to use the specified model.

    Supports:
      - Gemini models (e.g. gemini-3.5-flash, antigravity-preview-05-2026) via gemini/ or google/ prefix
      - OpenRouter models (e.g. openrouter/anthropic/claude-3-opus) via openrouter/ prefix
      - OpenCode Zen models (e.g. minimax-m3-free) via openai/ prefix or by default
    """
    import os
    import dspy
    from pathlib import Path
    from dotenv import load_dotenv

    # Load keys
    load_dotenv(Path.home() / ".hermes" / ".env")

    if model_name.startswith("gemini/") or model_name.startswith("google/"):
        # Normalize to gemini/ for LiteLLM
        model_id = model_name
        if model_name.startswith("google/"):
            model_id = "gemini/" + model_name[7:]

        # Ensure GEMINI_API_KEY is populated from GOOGLE_API_KEY if needed
        if not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

        lm = dspy.LM(model_id)
        
    elif model_name.startswith("openrouter/"):
        # Extract the model identifier
        model_id = model_name[11:]
        # LiteLLM routes to openrouter via openrouter/ prefix or base_url config
        lm = dspy.LM(
            model=f"openrouter/{model_id}",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
        
    else:
        # Default to OpenCode Zen
        zen_api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
        zen_base_url = "https://opencode.ai/zen/v1"

        model_id = model_name
        if not (model_name.startswith("openai/") or "/" in model_name):
            model_id = f"openai/{model_name}"

        lm = dspy.LM(
            model=model_id,
            api_key=zen_api_key,
            base_url=zen_base_url,
        )

    dspy.configure(lm=lm)
    return lm

