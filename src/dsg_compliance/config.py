from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    anthropic_api_key: str | None = None  # not used yet - swap in once Anthropic credit is available
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma")
    # Hosted (not local) so the Render free-tier instance (512MB RAM) never
    # has to load a torch model - a local sentence-transformers model was
    # tried first and reliably OOM-killed the process on every /ask request.
    embedding_model: str = "gemini-embedding-001"

    # Chat/answer-generation layer: GLM (z.ai), free tier - chosen over Gemini
    # because Gemini's shared daily quota was already exhausted by other
    # projects in this workspace when this was built (see fuzzy-logic
    # project notes). Swap chat_provider back to "anthropic" once credit
    # is available, or to "gemini" once its quota resets.
    chat_provider: str = "glm"  # "glm" | "gemini" | "anthropic"
    glm_api_key: str | None = None
    glm_model: str = "glm-4.5-flash"
    glm_base_url: str = "https://api.z.ai/api/paas/v4/"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), extra="ignore"
    )


def data_dir() -> Path:
    return PROJECT_ROOT / "data"


def config_dir() -> Path:
    return PROJECT_ROOT / "config"


settings = Settings()
