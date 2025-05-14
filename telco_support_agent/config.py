"""Configuration management for the telco support agent."""

import os

# Environment detection
ENV = os.environ.get("TELCO_SUPPORT_AGENT_ENV", "dev")

# Unity Catalog configuration
UNITY_CATALOG_CONFIG = {
    "dev": {
        "catalog": "telco_customer_support_dev",
        "schema": "bronze",
    },
    "prod": {
        "catalog": "telco_customer_support_prod",
        "schema": "bronze",
    },
}


def get_uc_config(env: str = ENV) -> dict[str, str]:
    """Get Unity Catalog configuration for the specified environment.

    Args:
        env: Environment name (dev, prod)

    Returns:
        Dictionary with catalog and schema configuration
    """
    return UNITY_CATALOG_CONFIG.get(env, UNITY_CATALOG_CONFIG["dev"])
