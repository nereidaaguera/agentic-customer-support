"""Configuration settings for the Telco Support Agent UI."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", env="HOST")  # noqa: S104
    port: int = Field(default=8000, env="PORT")

    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"], env="CORS_ORIGINS"
    )

    # Databricks settings
    databricks_host: str = Field(
        default="https://db-ml-models-prod-us-west.cloud.databricks.com",
        env="DATABRICKS_HOST",
    )
    databricks_token: str = Field(..., env="DATABRICKS_TOKEN")
    databricks_endpoint_name: str = Field(
        default="telco-customer-support-agent", env="DATABRICKS_ENDPOINT_NAME"
    )

    # Request settings
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    max_retries: int = Field(default=3, env="MAX_RETRIES")

    # Default customer IDs for demo/testing
    demo_customer_ids: list[str] = Field(
        default=[
            "CUS-10001",
            "CUS-10002",
            "CUS-10006",
            "CUS-10023",
            "CUS-10048",
            "CUS-10172",
            "CUS-11081",
            "CUS-10619",
        ],
        env="DEMO_CUSTOMER_IDS",
    )

    @property
    def databricks_endpoint(self) -> str:
        """Get the full Databricks endpoint URL."""
        return f"{self.databricks_host}/serving-endpoints/{self.databricks_endpoint_name}/invocations"

    @property
    def databricks_headers(self) -> dict:
        """Get headers for Databricks API requests."""
        return {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json",
        }

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
