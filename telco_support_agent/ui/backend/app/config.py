"""Config settings for the Telco Support Agent UI."""

import logging
import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # App settings
    environment: str = Field(
        default_factory=lambda: os.getenv("ENV", "dev")
    )
    host: str = Field(default="0.0.0.0")  # noqa: S104
    port: int = Field(
        default_factory=lambda: int(os.getenv("DATABRICKS_APP_PORT", "8000"))
    )

    # Databricks settings
    databricks_host: str = Field(
        default_factory=lambda: os.getenv(
            "DATABRICKS_HOST", "https://db-ml-models-prod-us-west.cloud.databricks.com"
        )
    )
    databricks_token: str = Field(
        default_factory=lambda: os.getenv("DATABRICKS_TOKEN", "")
    )
    databricks_client_id: str = Field(
        default_factory=lambda: os.getenv("DATABRICKS_CLIENT_ID", "")
    )
    databricks_client_secret: str = Field(
        default_factory=lambda: os.getenv("DATABRICKS_CLIENT_SECRET", "")
    )
    databricks_endpoint_name: str = Field(
        default_factory=lambda: os.getenv(
            "DATABRICKS_ENDPOINT_NAME", "dev-telco-customer-support-agent"
        )
    )

    # MLflow settings
    mlflow_experiment_path_override: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_PATH", "")
    )

    # Request settings
    request_timeout: int = Field(default=300)
    max_retries: int = Field(default=3)

    def __init__(self, **kwargs):
        """Initialize settings with logging."""
        super().__init__(**kwargs)

        if self.databricks_host and not self.databricks_host.startswith(
            ("http://", "https://")
        ):
            self.databricks_host = f"https://{self.databricks_host}"
            logger.info(
                f"Added https:// protocol to Databricks host: {self.databricks_host}"
            )

        # Configure MLflow environment variables
        if self.has_auth:
            os.environ["DATABRICKS_HOST"] = self.databricks_host
            if self.databricks_token:
                os.environ["DATABRICKS_TOKEN"] = self.databricks_token
            # Note: OAuth credentials are handled by the service layer

        logger.info("Initialized settings:")
        logger.info(f"  Environment: {self.environment}")
        logger.info(f"  Port: {self.port}")
        logger.info(f"  Databricks Host: {self.databricks_host}")
        logger.info(f"  Endpoint Name: {self.databricks_endpoint_name}")
        logger.info(f"  Full Endpoint URL: {self.databricks_endpoint}")
        logger.info(f"  Auth Method: {self.auth_method}")
        logger.info(f"  MLflow Experiment Path: {self.mlflow_experiment_path}")

        if self.auth_method == "none":
            logger.warning(
                "No Databricks authentication configured - chat functionality will be disabled"
            )

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
    def auth_method(self) -> str:
        """Get the authentication method being used."""
        if self.databricks_client_id and self.databricks_client_secret:
            return "oauth"
        elif self.databricks_token:
            return "token"
        else:
            return "none"

    @property
    def has_auth(self) -> bool:
        """Check if any authentication is configured."""
        return self.auth_method != "none"

    @property
    def databricks_headers(self) -> dict:
        """Get headers for Databricks API requests."""
        headers = {"Content-Type": "application/json"}

        if self.databricks_client_secret:
            headers["Authorization"] = f"Bearer {self.databricks_client_secret}"
        elif self.databricks_token:
            headers["Authorization"] = f"Bearer {self.databricks_token}"
        else:
            pass

        return headers

    @property
    def mlflow_experiment_path(self) -> str:
        """Get the MLflow experiment path based on environment and endpoint."""
        if self.mlflow_experiment_path_override:
            return self.mlflow_experiment_path_override

        # get environment and base name from endpoint name
        if self.databricks_endpoint_name.startswith("dev-"):
            env = "dev"
        elif self.databricks_endpoint_name.startswith("staging-"):
            env = "staging"
        elif self.databricks_endpoint_name.startswith("prod-"):
            env = "prod"
        else:
            if self.environment in ("production", "prod"):
                env = "prod"
            elif self.environment == "staging":
                env = "staging"
            else:
                env = "dev"

        experiment_path = f"/Shared/telco_support_agent/{env}/{env}_telco_support_agent"

        return experiment_path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
