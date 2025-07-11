"""Account agent for handling account-related queries."""

from typing import Optional

from telco_support_agent.agents import UCConfig
from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools.registry import get_toolkit_for_domain
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class AccountAgent(BaseAgent):
    """Account agent to manage user's account queries.

    This agent answers questions related to user's account information,
    subscriptions, and account management.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        config_dir: Optional[str] = None,
        disable_tools: Optional[list[str]] = None,
        uc_config: Optional[UCConfig] = None,
    ) -> None:
        """Initialize the account agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
            disable_tools: Optional list of tool names to disable
            uc_config: Optional UC configuration for Unity Catalog resources
        """
        # get toolkit for account domain
        toolkit = get_toolkit_for_domain(
            "account",
            uc_config
            or UCConfig(catalog="telco_customer_support_prod", data_schema="gold"),
        )

        logger.info(f"Account agent initialized with {len(toolkit.tools)} tools")

        super().__init__(
            agent_type="account",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=toolkit.tools,
            inject_tool_args=["customer"],
            disable_tools=disable_tools,
            uc_config=uc_config,
        )
