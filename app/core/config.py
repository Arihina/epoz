from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    EMBED_MODEL: str
    OLLAMA_MODEL: str

    EMBED_DEVICE: str | None = None
    EMBED_LOCAL_FILES_ONLY: bool = True

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6336

    HISTORY_LIMIT: int = 10

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
