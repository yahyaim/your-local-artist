"""Application settings, read from environment variables or the repo-level `.env` file."""

from functools import cached_property
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Root for temp uploads and confirmed projects (env: STORAGE_DIR).
    storage_dir: Path = REPO_ROOT / "storage"
    # Unconfirmed uploads untouched for this long are deleted (env: TEMP_TTL_MINUTES).
    temp_ttl_minutes: float = 60.0

    @cached_property
    def temp_dir(self) -> Path:
        return self.storage_dir / "temp"

    @cached_property
    def projects_dir(self) -> Path:
        return self.storage_dir / "projects"
