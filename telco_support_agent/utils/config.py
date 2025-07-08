"""Configuration management for the telco support agent."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from mlflow.artifacts import download_artifacts
from pydantic import BaseModel

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

class MCPServer(BaseModel):
    """Model for MCP server configuration."""

    server_url: str
    app_name: Optional[str]


class UCConfig(BaseModel):
    data: dict[str, Any]
    agent: dict[str, Any]

    def get_uc_function_name(self, function_name: str) -> str:
        """Returns name of uc function based on config."""
        agent_catalog = self.agent["catalog"]
        agent_schema = self.agent["schema"]
        return f"{agent_catalog}.{agent_schema}.{function_name}"

    def get_uc_index_name(self, index_name: str) -> str:
        """Returns name of uc index based on config."""
        catalog = self.data["catalog"]
        schema = self.data["schema"]
        return f"{catalog}.{schema}.{index_name}"

    def get_uc_table_name(self, table_name: str) -> str:
        """Returns name of uc table based on config."""
        data_catalog = self.data["catalog"]
        data_schema = self.data["schema"]
        return f"{data_catalog}.{data_schema}.{table_name}"

    def get_uc_model_name(self) -> str:
        """Returns name of uc model based on config."""
        return (
            f"{self.agent['catalog']}.{self.agent['schema']}.{self.agent['model_name']}"
        )


ENV = os.environ.get("TELCO_SUPPORT_AGENT_ENV", "dev")
UC_CONFIG_FILE = "uc_config.yaml"


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

    def __init__(self, environment: Optional[str] = ENV):
        """Initialize the config manager if not already initialized."""
        if getattr(self, "_initialized", False):
            return

        self._configs = {}
        self.uc_config = None
        self._project_root = self._find_project_root()
        self._environment = environment
        self._initialized = True

    def get_uc_config(self):
        """Initialize uc config if not already initialized."""
        if not self.uc_config:
            uc_config_file_path = self._find_config_file(UC_CONFIG_FILE)
            if uc_config_file_path is None:
                raise FileNotFoundError(
                    f"No configuration file found: {UC_CONFIG_FILE}"
                )

            try:
                with open(uc_config_file_path) as f:
                    config = yaml.safe_load(f)
                    self.uc_config = UCConfig(**config[self._environment])
            except Exception as e:
                raise ValueError(
                    f"Error loading configuration {UC_CONFIG_FILE}: {str(e)}"
                ) from e

        return self.uc_config

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current_dir = Path(__file__).resolve().parent

        search_dir = current_dir
        for _ in range(10):
            if (search_dir / "telco_support_agent").exists() and (
                search_dir / "pyproject.toml"
            ).exists():
                return search_dir
            parent_dir = search_dir.parent
            if parent_dir == search_dir:  # reached root directory
                break
            search_dir = parent_dir

        return current_dir.parent

    def _find_config_file(
        self, filename: str, parent_folder: Optional[str] = None
    ) -> Optional[Path]:
        """Find the configuration file path.

        Search multiple locations:
        1. Development paths (project directory)
        2. Model serving paths (/model/artifacts)
        3. MLflow run artifacts

        Args:
            filename: Agent or UC config file name. (e.g., 'supervisor.yaml', 'account.yaml')
            parent_folder: Name of the folder inside the configs folder that contains the configuration file.

        Returns:
            Path to the config file, or None if not found
        """
        # 1. try development paths first
        search_paths = [
            Path(
                self._project_root,
                "configs",
                parent_folder if parent_folder else "",
                filename,
            ),
            Path(
                "/Workspace/Repos/",
                "*/telco-support-agent/configs",
                parent_folder if parent_folder else "",
                filename,
            ),
            Path(
                Path.cwd(), "configs", parent_folder if parent_folder else "", filename
            ),
        ]

        for path in search_paths:
            if isinstance(path, Path) and path.exists():
                logger.info(f"Found config file {filename} at {path}")
                return path
            elif isinstance(path, str) and Path(path).exists():
                logger.info(f"Found config file {filename} at {path}")
                return Path(path)

        # 2. try model serving paths
        model_artifact_paths = [
            Path("/model/artifacts/configs", parent_folder if parent_folder else "")
            / filename,
            Path("/model/artifacts") / filename,
        ]

        for path in model_artifact_paths:
            if path.exists():
                logger.info(f"Found config file {filename} at {path}")
                return path

        # 3. try using MLflow artifact APIs
        try:
            artifact_path = (
                f"configs/agents/{filename}" if parent_folder else f"configs/{filename}"
            )
            local_path = download_artifacts(artifact_path=artifact_path)
            if local_path:
                logger.info(f"Downloaded config file {filename} from MLflow artifacts")
                return Path(local_path)
        except Exception as e:
            logger.debug(f"Could not download config from MLflow artifacts: {e}")

        logger.warning(f"Could not find config file {filename}")
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

        config_path = self._find_config_file(f"{agent_type}.yaml", "agents")
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
