"""Configuration settings for the Telco Support Agent UI."""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # App settings
    environment: str = Field(default="development")
    host: str = Field(default="0.0.0.0")  # noqa: S104
    # Use DATABRICKS_APP_PORT if available, otherwise default to 8000
    port: int = Field(
        default_factory=lambda: int(os.getenv("DATABRICKS_APP_PORT", "8000"))
    )

    # Databricks settings - use environment variables when available
    databricks_host: str = Field(
        default_factory=lambda: os.getenv(
            "DATABRICKS_HOST", "https://db-ml-models-prod-us-west.cloud.databricks.com"
        )
    )
    databricks_token: str = Field(default="")
    databricks_endpoint_name: str = Field(default="telco-customer-support-agent")

    # Request settings
    request_timeout: int = Field(default=300)
    max_retries: int = Field(default=3)

    def get_cors_origins(self) -> list[str]:
        """Get CORS origins from environment or defaults."""
        cors_str = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        )
        return [origin.strip() for origin in cors_str.split(",") if origin.strip()]

    def get_demo_customer_ids(self) -> list[str]:
        """Get demo customer IDs from environment or defaults."""
        customers_str = os.getenv(
            "DEMO_CUSTOMER_IDS",
            "CUS-10001,CUS-10002,CUS-10006,CUS-10023,CUS-10048,CUS-10172,CUS-11081,CUS-10619",
        )
        return [
            customer_id.strip()
            for customer_id in customers_str.split(",")
            if customer_id.strip()
        ]

    @property
    def cors_origins(self) -> list[str]:
        """Get CORS origins."""
        return self.get_cors_origins()

    @property
    def demo_customer_ids(self) -> list[str]:
        """Get demo customer IDs."""
        return self.get_demo_customer_ids()

    @property
    def databricks_endpoint(self) -> str:
        """Get the full Databricks endpoint URL."""
        return f"{self.databricks_host}/serving-endpoints/{self.databricks_endpoint_name}/invocations"

    @property
    def databricks_headers(self) -> dict:
        """Get headers for Databricks API requests."""
        if not self.databricks_token:
            raise ValueError("DATABRICKS_TOKEN is required but not set")
        return {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json",
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
