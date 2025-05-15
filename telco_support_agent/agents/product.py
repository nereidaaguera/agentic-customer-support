"""Product agent for handling products and plans related queries."""

import logging
from typing import Optional

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class ProductAgent(BaseAgent):
    """Product agent to manage products and plans queries.

    This agent answers questions related to products and plans information.
    """

    def __init__(
        self, llm_endpoint: Optional[str] = None, config_dir: Optional[str] = None
    ) -> None:
        """Initialize the product agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
        """
        # Initialize tools from registry
        tools = [tool.get_tool_info() for tool in ToolRegistry.get_tools("product")]

        if not tools:
            logger.warning(
                "No tools registered for product agent. Make sure tools are initialized."
            )
        else:
            logger.info(f"Product agent initialized with {len(tools)} tools")

        super().__init__(
            agent_type="product",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=tools,
        )
