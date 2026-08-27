from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./backend/data/impossible_market.db"
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file="backend/.env",
        env_prefix="IMPOSSIBLE_MARKET_",
        extra="ignore",
    )

    @cached_property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
