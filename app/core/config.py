from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENV: str = "dev"

    INTERNAL_AUTH_MODE: Literal["off", "token", "assertion"] = "token"
    MASTER_INTERNAL_TOKEN: str | None = None
    MASTER_ASSERTION_PUBLIC_KEY: str | None = None

    PUBLIC_PATHS: str = "/health"
    AGENT_ID: str = "epoz"

    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    EMBED_MODEL: str
    OLLAMA_MODEL: str
    OLLAMA_HOST: str = "http://localhost:11434"

    EMBED_DEVICE: str = "cpu"
    EMBED_LOCAL_FILES_ONLY: bool = False

    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_ENABLED: bool = True
    RERANK_ALPHA: float = 0.5

    RETRIEVAL_THRESHOLD: float = 0.35
    QUERY_REWRITE: bool = True

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6336
    QDRANT_COLLECTION: str = "rag_docs_hybrid"

    MINERU_API_URL: str
    MINERU_BACKEND: str = "pipeline"
    MINERU_LANG: str = "cyrillic"
    MINERU_TIMEOUT_SECONDS: int = 600

    MAX_ATTACHED_FILES: int = 5

    CONTEXT_WINDOW: int = 32768
    RESERVE_OUTPUT_TOKENS: int = 4096
    CONTEXT_SAFETY_TOKENS: int = 128
    HISTORY_MIN_TOKENS: int = 2048
    TOKENIZER_REPO: str | None = None
    DOCUMENT_OVERFLOW: str = "truncate"

    HISTORY_LIMIT: int = 10

    @property
    def is_prod(self) -> bool:
        return self.ENV.strip().lower() in {"prod", "production"}

    @property
    def public_paths(self) -> tuple[str, ...]:
        return tuple(
            p.strip() for p in self.PUBLIC_PATHS.split(",") if p.strip()
        )

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @field_validator("EMBED_DEVICE", mode="before")
    @classmethod
    def _normalize_embed_device(cls, value):
        """Пусто/None → cpu.

        Старый README предлагал ``EMBED_DEVICE=`` (пусто = авто-выбор CUDA).
        Такие .env не должны снова включать GPU после перехода на cpu по
        умолчанию.
        """
        if value is None:
            return "cpu"
        if isinstance(value, str) and not value.strip():
            return "cpu"
        return value

    @model_validator(mode="after")
    def _check_internal_auth(self) -> "Settings":
        """Fail-closed: агент не стартует при неполной конфигурации auth."""
        if self.INTERNAL_AUTH_MODE == "token":
            token = self.MASTER_INTERNAL_TOKEN or ""
            if not token:
                raise ValueError(
                    "INTERNAL_AUTH_MODE=token, но MASTER_INTERNAL_TOKEN пуст"
                )
            if self.is_prod and len(token) < 32:
                raise ValueError(
                    "MASTER_INTERNAL_TOKEN короче 32 символов при ENV=prod"
                )
        elif self.INTERNAL_AUTH_MODE == "assertion":
            if not self.MASTER_ASSERTION_PUBLIC_KEY:
                raise ValueError(
                    "INTERNAL_AUTH_MODE=assertion, но "
                    "MASTER_ASSERTION_PUBLIC_KEY пуст"
                )
        if self.is_prod and self.INTERNAL_AUTH_MODE == "off":
            raise ValueError("INTERNAL_AUTH_MODE=off запрещён при ENV=prod")
        return self

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
