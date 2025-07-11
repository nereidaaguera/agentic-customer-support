"""Configuration models for each notebook - clear and direct."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class LogRegisterConfig(BaseModel):
    """Configuration for log_register_agent notebook."""
    
    # Environment
    env: str
    
    # Unity Catalog
    uc_catalog: str
    agent_schema: str  # UC schema for agent models/functions
    data_schema: str   # UC schema for data tables
    model_name: str
    
    # MLflow
    experiment_name: str
    
    # Model metadata
    agent_name: str = "telco_customer_support_agent"
    agent_description: str = "Multi-agent system for telco customer support"
    
    # Model signature example
    input_example: Dict[str, Any] = Field(default_factory=lambda: {
        "input": [{"role": "user", "content": "What was the customer's data usage last month?"}],
        "custom_inputs": {"customer": "CUS-10001"}
    })
    
    # Feature flags
    disable_tools: List[str] = Field(default_factory=list)
    git_commit: Optional[str] = None
    
    @property
    def full_model_name(self) -> str:
        """Get the full Unity Catalog model name."""
        return f"{self.uc_catalog}.{self.agent_schema}.{self.model_name}"
    
    def get_table_name(self, table: str) -> str:
        """Get full table name in data schema."""
        return f"{self.uc_catalog}.{self.data_schema}.{table}"
    
    def get_function_name(self, function: str) -> str:
        """Get full function name in agent schema."""
        return f"{self.uc_catalog}.{self.agent_schema}.{function}"


class DeployAgentConfig(BaseModel):
    """Configuration for deploy_agent notebook."""
    
    # Environment
    env: str
    
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
    monitoring_replace_existing: bool = False
    monitoring_fail_on_error: bool = False
    
    # Permissions
    permissions: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {"users": ["telco-customer-support"], "permission_level": "CAN_MANAGE"},
        {"users": ["users"], "permission_level": "CAN_QUERY"}
    ])
    
    # Instructions for review app
    instructions: str = """Telco Customer Support Agent

This agent helps with telecom customer support queries including:
- Account information and subscription details
- Billing inquiries and payment information
- Technical support and troubleshooting
- Product information and plan comparisons

Please test various query types and provide feedback on response quality."""
    
    @property
    def full_model_name(self) -> str:
        """Get the full Unity Catalog model name."""
        return f"{self.uc_catalog}.{self.agent_schema}.{self.model_name}"