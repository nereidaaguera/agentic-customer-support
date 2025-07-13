"""Billing agent for handling billing-related queries."""

from typing import Optional

from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.config import UCConfig
from telco_support_agent.tools.registry import get_toolkit_for_domain
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class BillingAgent(BaseAgent):
    """Billing agent to manage customer billing queries.

    This agent answers questions related to customer billing information,
    payments, charges, billing cycles, and usage details.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        config_dir: Optional[str] = None,
        disable_tools: Optional[list[str]] = None,
        uc_config: Optional[UCConfig] = None,
    ) -> None:
        """Initialize the billing agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
            disable_tools: Optional list of tool names to disable
            uc_config: Optional UC configuration for Unity Catalog resources
        """
        # get toolkit for billing domain (custom UC functions)
        billing_toolkit = get_toolkit_for_domain(
            "billing",
            uc_config
            or UCConfig(
                agent_catalog="telco_customer_support_prod", data_schema="gold"
            ),
        )

        # add system.ai.python_exec for date calculations and billing analysis
        try:
            python_exec_toolkit = UCFunctionToolkit(
                function_names=["system.ai.python_exec"]
            )
            all_tools = billing_toolkit.tools + python_exec_toolkit.tools
            logger.info("Added system.ai.python_exec tool for billing calculations")
        except Exception as e:
            logger.warning(
                f"Could not add python_exec tool: {e}. Continuing with UC functions only."
            )
            all_tools = billing_toolkit.tools

        logger.info(f"Billing agent initialized with {len(all_tools)} tools")

        super().__init__(
            agent_type="billing",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=all_tools,
            inject_tool_args=["customer"],
            disable_tools=disable_tools,
            uc_config=uc_config,
        )
