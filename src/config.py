from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider
import os
from dotenv import load_dotenv


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    # Load .env file if it exists
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    # Create directories
    state_dir = root / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "profiles").mkdir(exist_ok=True)

    # Read config knobs with sensible defaults for local Qwen model
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    # Main Model Configuration
    provider = os.getenv("LLM_PROVIDER", "custom")
    model_name = os.getenv("LLM_MODEL", "qwen2.5-1.5b-instruct")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")

    model_config = ProviderConfig(
        provider=normalize_provider(provider),
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )

    # Judge Model Configuration
    judge_provider = os.getenv("JUDGE_PROVIDER", provider)
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)
    judge_base_url = os.getenv("JUDGE_BASE_URL", base_url)

    judge_config = ProviderConfig(
        provider=normalize_provider(judge_provider),
        model_name=judge_model_name,
        temperature=judge_temperature,
        api_key=judge_api_key,
        base_url=judge_base_url
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_config
    )

