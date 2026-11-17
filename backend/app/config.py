"""Application settings (environment variables, see .env.example). None are AI-service API keys."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
APP_VERSION = "1.0.0"
DISCLAIMER = "Research & educational prototype — not a medical device. Not for clinical use."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    database_url: str = f"sqlite:///{ROOT / 'backend' / 'medivision.db'}"
    storage_dir: Path = ROOT / "backend" / "storage"
    artifacts_dir: Path = ROOT / "artifacts"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    model_profile: str = "auto"  # auto | starter | full
    inference_backend: str = "onnx"  # onnx | torch
    mc_dropout_samples: int = 8
    pseudonym_salt: str = "medivision-demo-salt"
    log_level: str = "INFO"
    copilot_local_llm: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    @field_validator("storage_dir", "artifacts_dir", mode="before")
    @classmethod
    def _abs(cls, v: str | Path) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (ROOT / p).resolve()

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    return s
