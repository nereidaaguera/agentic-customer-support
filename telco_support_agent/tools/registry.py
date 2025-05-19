"""Utilities for registering and accessing UC functions."""

from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from telco_support_agent.agents.types import AgentType
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

# default catalog and schema
DEFAULT_CATALOG = "telco_customer_support_dev"
DEFAULT_SCHEMA = "agent"


def get_toolkit_for_domain(domain: str) -> UCFunctionToolkit:
    """Get a toolkit for a specific domain.

    Args:
        domain: Domain to get the toolkit for (e.g., "account", "billing")

    Returns:
        UCFunctionToolkit: Toolkit containing the appropriate functions
    """
    try:
        # validate against known agent types
        AgentType.from_string(domain)
    except ValueError:
        logger.warning(f"Requested toolkit for non-standard domain: {domain}")

    # map domains to function names
    function_map = {
        AgentType.ACCOUNT.value: [
            f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.get_customer_info",
            f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.get_customer_subscriptions",
        ],
        AgentType.BILLING.value: [
            # TODO: add billing functions
        ],
        AgentType.TECH_SUPPORT.value: [
            # TODO: add tech support functions
        ],
        AgentType.PRODUCT.value: [
            # TODO: add product functions
        ],
    }

    function_names = function_map.get(domain, [])

    return UCFunctionToolkit(function_names=function_names)
