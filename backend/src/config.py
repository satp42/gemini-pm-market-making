"""Application configuration loaded from environment variables via pydantic-settings."""

from __future__ import annotations

from functools import cached_property

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    """Gemini API connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(default="", description="Gemini API key")
    api_secret: str = Field(default="", description="Gemini API secret")
    env: str = Field(default="sandbox", description="sandbox or live")

    @cached_property
    def base_url(self) -> str:
        if self.env == "live":
            return "https://api.gemini.com"
        return "https://api.sandbox.gemini.com"


class ASSettings(BaseSettings):
    """Avellaneda-Stoikov model parameters."""

    model_config = SettingsConfigDict(env_prefix="AS_")

    gamma: float = Field(default=0.1, description="Risk aversion parameter")
    k: float = Field(default=1.5, description="Order arrival intensity")
    sigma_default: float = Field(
        default=0.01, description="Default volatility when insufficient data"
    )
    variance_window: int = Field(default=100, description="Number of trades for rolling variance")


class BotSettings(BaseSettings):
    """Bot loop behaviour settings."""

    model_config = SettingsConfigDict(env_prefix="")

    bot_cycle_seconds: int = Field(default=10, alias="BOT_CYCLE_SECONDS")
    scanner_cycle_seconds: int = Field(default=300, alias="SCANNER_CYCLE_SECONDS")
    min_spread: float = Field(default=0.03, alias="MIN_SPREAD")
    min_time_to_expiry_hours: int = Field(default=1, alias="MIN_TIME_TO_EXPIRY_HOURS")
    excluded_symbols: list[str] = Field(default_factory=list, alias="EXCLUDED_SYMBOLS")


class RiskSettings(BaseSettings):
    """Risk management limits."""

    model_config = SettingsConfigDict(env_prefix="")

    max_inventory_per_symbol: int = Field(default=200, alias="MAX_INVENTORY_PER_SYMBOL")
    max_total_exposure: int = Field(default=1000, alias="MAX_TOTAL_EXPOSURE")
    risk_widen_threshold: float = Field(default=0.8, alias="RISK_WIDEN_THRESHOLD")


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="")

    url: str = Field(
        default="postgresql+asyncpg://gemini:gemini_dev@localhost:5432/gemini_mm",
        alias="DATABASE_URL",
    )


class AppSettings(BaseSettings):
    """General application settings."""

    model_config = SettingsConfigDict(env_prefix="")

    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")
    data_retention_days: int = Field(default=7, alias="DATA_RETENTION_DAYS")


class Settings(BaseSettings):
    """Root settings object that composes all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    avellaneda_stoikov: ASSettings = Field(default_factory=ASSettings)
    bot: BotSettings = Field(default_factory=BotSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    app: AppSettings = Field(default_factory=AppSettings)


def get_settings() -> Settings:
    """Create and return a Settings instance (reads .env on first call)."""
    return Settings()
