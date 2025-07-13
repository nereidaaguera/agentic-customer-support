"""Telco Agents."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Model for LLM configuration from YAML."""

    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)


class UCConfig(BaseModel):
    """Unity Catalog configuration with separation of data reading vs agent artifacts."""

    # For reading data (customer info, billing, etc.) - always prod
    data_catalog: str = "telco_customer_support_prod"
    data_schema: str = "gold"

    # For agent artifacts (functions, models) - environment specific
    agent_catalog: str = "telco_customer_support_prod"  # Default to prod for testing
    agent_schema: str = "agent"
    model_name: str = "telco_customer_support_agent"

    def get_uc_function_name(self, function_name: str) -> str:
        """Returns full UC function name (uses agent catalog)."""
        return f"{self.agent_catalog}.{self.agent_schema}.{function_name}"

    def get_uc_table_name(self, table_name: str) -> str:
        """Returns full UC table name for data reading (uses data catalog)."""
        return f"{self.data_catalog}.{self.data_schema}.{table_name}"

    def get_uc_index_name(self, index_name: str) -> str:
        """Returns full UC vector search index name (uses data catalog)."""
        return f"{self.data_catalog}.{self.data_schema}.{index_name}"

    def get_uc_model_name(self) -> str:
        """Returns full UC model name (uses agent catalog)."""
        return f"{self.agent_catalog}.{self.agent_schema}.{self.model_name}"

    # Backward compatibility
    @property
    def catalog(self) -> str:
        """Backward compatibility - returns agent catalog."""
        return self.agent_catalog


class AgentConfig(BaseModel):
    """Model for agent configuration from YAML."""

    name: str
    description: str
    llm: LLMConfig
    system_prompt: str
    uc_functions: list[str] = Field(default_factory=list)
    uc_config: UCConfig

    @classmethod
    def load_from_file(cls, agent_type: str, uc_config: UCConfig) -> "AgentConfig":
        """Load agent config from YAML file."""
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        # Log environment for debugging
        logger.info(f"Loading config for agent type: {agent_type}")
        logger.info(f"Environment variables:")
        for key in ["MLFLOW_RUN_ID", "MLFLOW_TRACKING_URI", "MLFLOW_MODEL_DIR", "DATABRICKS_HOST"]:
            logger.info(f"  {key}: {os.environ.get(key, 'Not set')}")
        
        # First, try to load from MLflow artifacts if we're in a model serving context
        try:
            from mlflow.artifacts import download_artifacts
            logger.info(f"Attempting to load {agent_type} config from MLflow artifacts...")
            
            # Check if we're in an MLflow context
            import mlflow
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            logger.info(f"MLflow active run ID: {run_id}")
            
            # Try to download from MLflow artifacts
            artifact_path = f"configs/agents/{agent_type}.yaml"
            logger.info(f"Trying to download artifact: {artifact_path}")
            local_path = download_artifacts(artifact_path=artifact_path)
            
            if local_path:
                logger.info(f"Downloaded artifact to: {local_path}")
                if Path(local_path).exists():
                    logger.info(f"Confirmed file exists at: {local_path}")
                    with open(local_path) as f:
                        config_dict = yaml.safe_load(f)
                        logger.info(f"Successfully loaded {agent_type} config from MLflow artifacts")
                        return cls(**config_dict, uc_config=uc_config)
                else:
                    logger.warning(f"Downloaded path doesn't exist: {local_path}")
            else:
                logger.warning("download_artifacts returned None")
                
        except ImportError as e:
            logger.warning(f"MLflow not available: {e}")
        except Exception as e:
            logger.warning(f"Could not load from MLflow artifacts: {type(e).__name__}: {e}")

        # Find the agent config file
        config_paths = [
            Path.cwd() / "configs" / "agents" / f"{agent_type}.yaml",
            Path(__file__).parent.parent.parent
            / "configs"
            / "agents"
            / f"{agent_type}.yaml",
            Path("/Workspace/Files") / "configs" / "agents" / f"{agent_type}.yaml",
            # MLflow model serving paths - artifacts in the model directory
            Path("/model/artifacts") / "configs" / "agents" / f"{agent_type}.yaml",
            Path("/model") / "configs" / "agents" / f"{agent_type}.yaml",
            Path("artifacts") / "configs" / "agents" / f"{agent_type}.yaml",
            # Also check parent directories in case we're in a subdirectory
            Path("../artifacts") / "configs" / "agents" / f"{agent_type}.yaml",
            Path("../../artifacts") / "configs" / "agents" / f"{agent_type}.yaml",
        ]

        # If MLFLOW_MODEL_DIR env var is set, also check there
        mlflow_model_dir = os.environ.get("MLFLOW_MODEL_DIR")
        if mlflow_model_dir:
            config_paths.append(
                Path(mlflow_model_dir)
                / "artifacts"
                / "configs"
                / "agents"
                / f"{agent_type}.yaml"
            )
            config_paths.append(
                Path(mlflow_model_dir) / "configs" / "agents" / f"{agent_type}.yaml"
            )

        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                logger.info(f"Found {agent_type} config at: {path}")
                logger.info(f"Running from directory: {os.getcwd()}")
                break

        if not config_path:
            # Log debugging info
            logger.error(f"Config file not found for {agent_type}. Searched paths:")
            for path in config_paths:
                logger.error(f"  - {path} (exists: {path.exists()})")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"__file__ location: {__file__}")

            raise FileNotFoundError(f"Agent config file not found for {agent_type}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded {agent_type} config from file system: {config_path}")
        return cls(**config_dict, uc_config=uc_config)
