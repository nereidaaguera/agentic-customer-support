"""Utilities for registering and accessing UC functions."""

from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

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
    # map domains to function names
    function_map = {
        "account": [
            f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.get_customer_info",
            f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.get_customer_subscriptions",
        ],
        "billing": [
            # TODO: add billing functions
        ],
        "tech_support": [
            # TODO: add tech support functions
        ],
        "product": [
            # TODO: add product functions
        ],
        "supervisor": [
            f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.route_to_specialized_agent"
        ],
    }

    function_names = function_map.get(domain, [])

    return UCFunctionToolkit(function_names=function_names)
