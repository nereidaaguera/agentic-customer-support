"""Account agent for handling account-related queries."""

import logging
from typing import Optional

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class AccountAgent(BaseAgent):
    """Account agent to manage user's account queries.

    This agent answers questions related to user's account information,
    subscriptions, and account management.
    """

    def __init__(
        self, llm_endpoint: Optional[str] = None, config_dir: Optional[str] = None
    ) -> None:
        """Initialize the account agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
        """
        # Initialize tools from registry
        tools = [tool.get_tool_info() for tool in ToolRegistry.get_tools("account")]

        if not tools:
            logger.warning(
                "No tools registered for account agent. Make sure tools are initialized."
            )
        else:
            logger.info(f"Account agent initialized with {len(tools)} tools")

        super().__init__(
            agent_type="account",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=tools,
        )
