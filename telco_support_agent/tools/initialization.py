"""Tool initialization and registration."""

import logging

logger = logging.getLogger(__name__)


def initialize_tools():
    """Initialize and register all tools."""
    logger.info("Initializing agent tools...")

    # import and register account tools
    from telco_support_agent.tools.account import register_account_tools

    register_account_tools()

    # TODO: add similar imports for other tool domains as they're implemented
    # from telco_support_agent.tools.billing import register_billing_tools
    # register_billing_tools()

    logger.info("Tool initialization complete")
