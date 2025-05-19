"""Configuration management for the telco support agent."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from mlflow.artifacts import download_artifacts

from telco_support_agent import PROJECT_ROOT
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

ENV = os.environ.get("TELCO_SUPPORT_AGENT_ENV", "dev")

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


class ConfigManager:
    """Manager for loading and accessing agent configurations."""

    _instance = None
    _configs = {}

    def __new__(cls):
        """Implement as singleton to cache configurations."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the config manager if not already initialized."""
        if getattr(self, "_initialized", False):
            return

        self._configs = {}
        self._project_root = PROJECT_ROOT
        self._initialized = True

    def _find_config_file(self, agent_type: str) -> Optional[Path]:
        """Find the configuration file for the given agent type.

        Search multiple locations:
        1. Development paths (project directory)
        2. Model serving paths (/model/artifacts)
        3. MLflow run artifacts

        Args:
            agent_type: Type of agent (e.g., 'supervisor', 'account')

        Returns:
            Path to the config file, or None if not found
        """
        filename = f"{agent_type}.yaml"

        # 1. try development paths first
        search_paths = [
            self._project_root / "configs" / "agents" / filename,
            Path("/Workspace/Repos/")
            / f"*/telco-support-agent/configs/agents/{filename}",
            Path.cwd() / "configs" / "agents" / filename,
        ]

        for path in search_paths:
            if isinstance(path, Path) and path.exists():
                logger.info(f"Found config file for {agent_type} at {path}")
                return path
            elif isinstance(path, str) and Path(path).exists():
                logger.info(f"Found config file for {agent_type} at {path}")
                return Path(path)

        # 2. try model serving paths
        model_artifact_paths = [
            Path("/model/artifacts/configs/agents") / filename,
            Path("/model/artifacts") / filename,
        ]

        for path in model_artifact_paths:
            if path.exists():
                logger.info(f"Found config file for {agent_type} at {path}")
                return path

        # 3. try using MLflow artifact APIs
        try:
            artifact_path = f"configs/agents/{filename}"
            local_path = download_artifacts(artifact_path=artifact_path)
            if local_path:
                logger.info(
                    f"Downloaded config file for {agent_type} from MLflow artifacts"
                )
                return Path(local_path)
        except Exception as e:
            logger.debug(f"Could not download config from MLflow artifacts: {e}")

        logger.warning(f"Could not find config file for {agent_type}")
        return None

    def get_config(self, agent_type: str) -> dict[str, Any]:
        """Get the configuration for the given agent type.

        Args:
            agent_type: Type of agent (e.g., 'supervisor', 'account')

        Returns:
            Configuration dictionary for the agent

        Raises:
            FileNotFoundError: If no configuration file can be found
            ValueError: If the configuration is invalid
        """
        if agent_type in self._configs:
            return self._configs[agent_type]

        config_path = self._find_config_file(agent_type)
        if config_path is None:
            raise FileNotFoundError(
                f"No configuration file found for agent type: {agent_type}"
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self._configs[agent_type] = config
                return config
        except Exception as e:
            raise ValueError(
                f"Error loading configuration for {agent_type}: {str(e)}"
            ) from e

    def get_all_agent_types(self) -> list[str]:
        """Get a list of all available agent types based on config files."""
        agent_types = []

        config_dir = self._project_root / "configs" / "agents"
        if config_dir.exists() and config_dir.is_dir():
            for file in config_dir.glob("*.yaml"):
                agent_type = file.stem
                if agent_type not in agent_types:
                    agent_types.append(agent_type)

        model_config_dir = Path("/model/artifacts/configs/agents")
        if model_config_dir.exists() and model_config_dir.is_dir():
            for file in model_config_dir.glob("*.yaml"):
                agent_type = file.stem
                if agent_type not in agent_types:
                    agent_types.append(agent_type)

        return agent_types


config_manager = ConfigManager()
