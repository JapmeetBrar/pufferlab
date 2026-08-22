"""Server-side settings. Secrets never cross the API boundary."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pufferlab_environment: str = "development"
    pufferlab_data_dir: Path = Path("data")
    pufferlab_fixture_dir: Path = Path("fixtures/tiny-corpus")
    pufferlab_search_namespace: str | None = None
    pufferlab_cors_origins: str = "http://localhost:5173"
    turbopuffer_api_key: SecretStr | None = None
    turbopuffer_region: str = "gcp-us-central1"

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip() for origin in self.pufferlab_cors_origins.split(",") if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
