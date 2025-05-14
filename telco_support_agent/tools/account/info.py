"""Account info tools."""

import time
from io import StringIO

import pandas as pd

from telco_support_agent.tools.base import FunctionType, ToolRegistry, UCTool
from telco_support_agent.tools.queries.account import (
    ACCOUNT_INFO_QUERY,
    SUBSCRIPTIONS_INFO_QUERY,
)
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class AccountInfoTool(UCTool):
    """Tool for retrieving customer account information."""

    def __init__(self, env: str = None):
        """Initialize account info tool.

        Args:
            env: Optional environment override
        """
        super().__init__(
            function_name="account_info_tool",
            function_type=FunctionType.SQL,
            description=(
                "Provides detailed information about a specific customer, including their "
                "registration date, status, and contact preferences. "
                "Use this tool to address queries about customer account details."
            ),
            env=env,
        )

    def create_function_value(self) -> str:
        """Create the SQL function for account info."""
        return f"""
        CREATE OR REPLACE FUNCTION {self.uc_name}(customer STRING COMMENT 'ID of the customer whose info to look up.')
        RETURNS TABLE
        COMMENT '{self.description}'
        RETURN {ACCOUNT_INFO_QUERY.format(catalog=self.catalog, schema=self.schema)}
        """

    def exec_fn(self, **kwargs) -> str:
        """Execute the account info query with error handling and retries.

        Args:
            **kwargs: Keyword arguments passed to the function, should include:
                customer: Customer ID to look up

        Returns:
            Formatted markdown table with customer information
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.client.execute_function(self.uc_name, parameters=kwargs)
                df = pd.read_csv(StringIO(result.value))
                if df.empty:
                    return (
                        f"No customer found with ID {kwargs.get('customer', 'unknown')}"
                    )
                return df.to_markdown(index=False)
            except Exception as e:
                logger.warning(
                    f"Account info tool execution failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Account info tool execution failed after {max_retries} attempts: {e}"
                    )
                    return f"Error retrieving customer information: {str(e)}"
                time.sleep(1 * (2**attempt))  # exponential backoff


class SubscriptionsInfoTool(UCTool):
    """Tool for retrieving customer subscription information."""

    def __init__(self, env: str = None):
        """Initialize the subscriptions info tool.

        Args:
            env: Optional environment override
        """
        super().__init__(
            function_name="subscriptions_info_tool",
            function_type=FunctionType.SQL,
            description=(
                "Provides comprehensive details on the subscriptions purchased by the customer, "
                "along with information about their corresponding plans. "
                "Use this tool to address queries about subscriptions and the plans linked to them."
            ),
            env=env,
        )

    def create_function_value(self) -> str:
        """Create the SQL function for subscriptions info."""
        return f"""
        CREATE OR REPLACE FUNCTION {self.uc_name}(customer STRING COMMENT 'ID of the customer whose subscriptions to look up.')
        RETURNS TABLE
        COMMENT '{self.description}'
        RETURN {SUBSCRIPTIONS_INFO_QUERY.format(catalog=self.catalog, schema=self.schema)}
        """

    def exec_fn(self, **kwargs) -> str:
        """Execute the subscriptions info query with error handling and retries.

        Args:
            **kwargs: Keyword arguments passed to the function, should include:
                customer: Customer ID to look up

        Returns:
            Formatted markdown table with subscription information
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.client.execute_function(self.uc_name, parameters=kwargs)
                df = pd.read_csv(StringIO(result.value))
                if df.empty:
                    return f"No subscriptions found for customer ID {kwargs.get('customer', 'unknown')}"
                return df.to_markdown(index=False)
            except Exception as e:
                logger.warning(
                    f"Subscriptions info tool execution failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Subscriptions info tool execution failed after {max_retries} attempts: {e}"
                    )
                    return f"Error retrieving subscription information: {str(e)}"
                time.sleep(1 * (2**attempt))  # Exponential backoff


# Register tools with the registry
# NOTE: some of these tools may be used for multiple agents
def register_account_tools():
    """Register all account tools with the tool registry."""
    account_info_tool = AccountInfoTool()
    subscriptions_info_tool = SubscriptionsInfoTool()

    # Register for account agent
    ToolRegistry.register_tool("account", account_info_tool)
    ToolRegistry.register_tool("account", subscriptions_info_tool)

    # NOTE: tools may also be useful for billing agent
    ToolRegistry.register_tool("billing", subscriptions_info_tool)
