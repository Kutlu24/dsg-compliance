from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    anthropic_api_key: str | None = None  # for the later chat/generation layer, not embeddings
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma")
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"  # free, local, strong German support

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), extra="ignore"
    )


def data_dir() -> Path:
    return PROJECT_ROOT / "data"


def config_dir() -> Path:
    return PROJECT_ROOT / "config"


settings = Settings()
