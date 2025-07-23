"""Core configuration schemas for the telco support agent."""

from pathlib import Path
from typing import Any, Optional

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


class MCPServer(BaseModel):
    """MCP Server configuration."""

    server_url: str
    app_name: Optional[str] = None


class AgentConfig(BaseModel):
    """Model for agent configuration from YAML."""

    name: str
    description: str
    llm: LLMConfig
    system_prompt: str
    uc_functions: list[str] = Field(default_factory=list)
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    uc_config: UCConfig

    @classmethod
    def load_from_file(cls, agent_type: str, uc_config: UCConfig) -> "AgentConfig":
        """Load agent config from YAML file."""
        import logging

        logger = logging.getLogger(__name__)

        # path checking - different between model serving and development
        config_paths = [
            # Model Serving: artifacts under /model/artifacts/
            Path("/model/artifacts") / f"{agent_type}.yaml",
            # Development/notebook setting: configs in project structure
            Path(__file__).parent.parent.parent
            / "configs"
            / "agents"
            / f"{agent_type}.yaml",
            # Current working directory (for local development)
            Path.cwd() / "configs" / "agents" / f"{agent_type}.yaml",
        ]

        for path in config_paths:
            if path.exists():
                logger.info(f"Found {agent_type} config at: {path}")
                with open(path) as f:
                    config_dict = yaml.safe_load(f)
                    return cls(**config_dict, uc_config=uc_config)

        logger.error(f"Config file not found for {agent_type}. Searched paths:")
        for path in config_paths:
            logger.error(f"  - {path} (exists: {path.exists()})")

        raise FileNotFoundError(f"Agent config file not found for {agent_type}")
