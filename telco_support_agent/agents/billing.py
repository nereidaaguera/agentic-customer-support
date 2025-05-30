"""Billing agent for handling billing-related queries."""

from typing import Optional

from telco_support_agent.agents.base_agent import BaseAgent
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
    ) -> None:
        """Initialize the billing agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
            disable_tools: Optional list of tool names to disable
        """
        # get toolkit for billing domain
        toolkit = get_toolkit_for_domain("billing")

        logger.info(f"Billing agent initialized with {len(toolkit.tools)} tools")

        super().__init__(
            agent_type="billing",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=toolkit.tools,
            inject_tool_args=["customer"],
            disable_tools=disable_tools,
        )
