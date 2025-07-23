"""Config models for each notebook."""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from telco_support_agent.config.schemas import UCConfig


class RunEvalsConfig(BaseModel):
    # Environment
    env: str

    # Unity Catalog
    uc_catalog: str
    agent_schema: str
    model_name: str
    model_version: Optional[int] = None  # if not set, use latest

    # MLflow
    experiment_name: str

    def to_uc_config(self) -> "UCConfig":
        """Convert to UCConfig for Unity Catalog operations."""
        from telco_support_agent.config.schemas import UCConfig

        return UCConfig(
            agent_catalog=self.uc_catalog,
            agent_schema=self.agent_schema,
            data_schema="gold",
            model_name=self.model_name,
        )


class LogRegisterConfig(BaseModel):
    """Config for log_register_agent notebook."""

    # Environment
    env: str

    # Unity Catalog
    uc_catalog: str
    agent_schema: str  # UC schema for agent models/functions
    data_schema: str  # UC schema for data tables
    model_name: str

    # MLflow
    experiment_name: str

    # Model metadata
    agent_name: str = "telco_customer_support_agent"
    agent_description: str = "Multi-agent system for telco customer support"

    # Model signature example
    input_example: dict[str, Any] = Field(
        default_factory=lambda: {
            "input": [
                {
                    "role": "user",
                    "content": "What was the customer's data usage last month?",
                }
            ],
            "custom_inputs": {"customer": "CUS-10001"},
        }
    )

    # Feature flags
    disable_tools: list[str] = Field(default_factory=list)
    git_commit: Optional[str] = None

    def to_uc_config(self) -> "UCConfig":
        """Convert to UCConfig for Unity Catalog operations."""
        from telco_support_agent.config.schemas import UCConfig

        return UCConfig(
            agent_catalog=self.uc_catalog,
            agent_schema=self.agent_schema,
            data_schema=self.data_schema,
            model_name=self.model_name,
        )

    @property
    def full_model_name(self) -> str:
        """Get the full Unity Catalog model name."""
        return self.to_uc_config().get_uc_model_name()


class DeployAgentConfig(BaseModel):
    """Config for deploy_agent notebook."""

    # Environment
    env: str
    git_commit: Optional[str] = None

    # Unity Catalog
    uc_catalog: str
    agent_schema: str
    model_name: str
    model_version: Optional[int] = None  # if not set, use latest

    # Model Serving
    endpoint_name: str
    scale_to_zero_enabled: bool = False
    workload_size: str = "Small"
    wait_for_ready: bool = True

    # Cleanup settings
    cleanup_old_versions: bool = True
    keep_previous_count: int = 1

    # Monitoring
    monitoring_enabled: bool = True
    monitoring_replace_existing: bool = True
    monitoring_fail_on_error: bool = False

    # Permissions
    permissions: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {"users": ["telco-customer-support"], "permission_level": "CAN_MANAGE"},
            {"users": ["users"], "permission_level": "CAN_QUERY"},
        ]
    )

    # Instructions for review app
    instructions: str = """Telco Customer Support Agent

This agent helps with telecom customer support queries including:
- Account information and subscription details
- Billing inquiries and payment information
- Technical support and troubleshooting
- Product information and plan comparisons

Please test various query types and provide feedback on response quality."""

    def to_uc_config(self) -> "UCConfig":
        """Convert to UCConfig for Unity Catalog operations."""
        from telco_support_agent.config.schemas import UCConfig

        return UCConfig(
            agent_catalog=self.uc_catalog,
            agent_schema=self.agent_schema,
            data_schema="gold",  # data schema not used in deployment
            model_name=self.model_name,
        )

    @property
    def full_model_name(self) -> str:
        """Get the full Unity Catalog model name."""
        return self.to_uc_config().get_uc_model_name()
