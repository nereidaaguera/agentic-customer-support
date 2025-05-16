"""Tools package for the telco support agent."""

import importlib

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

TOOL_MODULES = [
    # Account domain tools
    "telco_support_agent.tools.account.functions",
    # Supervisor domain tools
    "telco_support_agent.tools.supervisor.functions",
    # TODO: add when implemented
    # "telco_support_agent.tools.billing.functions",
    # "telco_support_agent.tools.tech_support.functions",
    # "telco_support_agent.tools.product.functions",
]


def initialize_tools() -> None:
    """Initialize and register all tools."""
    logger.info("Initializing tools...")

    # import all tool modules to register as UC functions
    for module_name in TOOL_MODULES:
        try:
            importlib.import_module(module_name)
            logger.info(f"Initialized tool module: {module_name}")
        except Exception as e:
            logger.error(f"Error initializing tool module {module_name}: {str(e)}")

    logger.info("Tools initialization complete")
