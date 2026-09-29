from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    api_host: str = "127.0.0.1"
    api_port: int = 8787

    db_url: str = "postgresql+psycopg://aegis@127.0.0.1:5432/aegis3"
    redis_url: str = "redis://127.0.0.1:6379/0"

    artifacts_dir: Path = Field(default_factory=lambda: Path.home() / ".local/share/aegis3/artifacts")
    runs_dir: Path = Field(default_factory=lambda: Path.home() / ".local/share/aegis3/runs")
    policy_path: Path = Field(default_factory=lambda: Path.home() / ".config/aegis3/policy.json")

    default_egress: str = "deny"
    hypo_backend: str = "rule"

    keychain_service: str = "aegis3"


settings = Settings()
