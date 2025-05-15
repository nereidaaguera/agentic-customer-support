"""Product info tools."""

import logging

from telco_support_agent.tools.base import FunctionType, ToolRegistry, UCTool
from telco_support_agent.tools.queries.product import PLANS_INFO_QUERY

logger = logging.getLogger(__name__)


class PlansInfoTool(UCTool):
    """Tool for retrieving plans information."""

    def __init__(self, env: str = None):
        """Initialize account info tool.

        Args:
            env: Optional environment override
        """
        super().__init__(
            function_name="plans_info_tool",
            function_type=FunctionType.SQL,
            description=(
                "Provides detailed information about the plans available."
                "Use this tool to address queries about plans and reasoning about plan comparisons."
            ),
            env=env,
        )

    def create_function_value(self) -> str:
        """Create the SQL function for account info."""
        return f"""
        CREATE OR REPLACE FUNCTION {self.uc_name}()
        RETURNS TABLE
        COMMENT '{self.description}'
        RETURN {PLANS_INFO_QUERY.format(catalog=self.catalog, schema=self.schema)}
        """


# Register tools with the registry
# NOTE: some of these tools may be used for multiple agents
def register_product_tools():
    """Register all product tools with the tool registry."""
    plans_info_tool = PlansInfoTool()

    # Register for product agent
    ToolRegistry.register_tool("product", plans_info_tool)
