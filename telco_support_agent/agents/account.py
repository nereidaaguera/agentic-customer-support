"""Account agent for handling account-related queries."""

from typing import Optional

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
        self, llm_endpoint: Optional[str] = None, config_dir: Optional[str] = None
    ) -> None:
        """Initialize the account agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
        """
        # get toolkit for account domain
        toolkit = get_toolkit_for_domain("account")

        logger.info(f"Account agent initialized with {len(toolkit.tools)} tools")

        super().__init__(
            agent_type="account",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=toolkit.tools,
            inject_tool_args={"customer": "context"},
        )
