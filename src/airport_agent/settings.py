"""Settings and config loading.

API keys come only from the environment (or a local .env file); they are never logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    oag_api_key: SecretStr | None = Field(default=None, alias="OAG_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")
    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")       # anthropic (default) | openrouter
    openrouter_model: str = Field(default="anthropic/claude-opus-5.5", alias="OPENROUTER_MODEL")  # OpenRouter option; ID not re-verified

    claude_model: str = "claude-haiku-5-5"      # the configured model (README, ARCHITECTURE §7, DESIGN §4)
    cache_dir_override: Path | None = Field(default=None, alias="AIRPORT_AGENT_CACHE_DIR")  # used by e2e tests
    data_dir: Path = PROJECT_ROOT / "data"
    config_dir: Path = PROJECT_ROOT / "config"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def cache_dir(self) -> Path:
        return self.cache_dir_override or self.data_dir / "cache"

    def require_oag_key(self) -> str:
        if self.oag_api_key is None:
            raise RuntimeError("OAG_API_KEY is not set. Put it in the environment or in .env (never in code).")
        return self.oag_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_config(name: str) -> dict[str, Any]:
    """Load config/<name>.yaml."""
    path = get_settings().config_dir / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
