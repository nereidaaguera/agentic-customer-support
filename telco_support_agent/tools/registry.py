"""Utilities for registering and accessing UC functions."""

from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from telco_support_agent.agents.types import AgentType
from telco_support_agent.config import UCConfig
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

# mapping of domains to function names
DOMAIN_FUNCTION_MAP = {
    AgentType.ACCOUNT.value: [
        "get_customer_info",
        "customer_subscriptions",
    ],
    AgentType.BILLING.value: [
        "get_billing_info",
        "get_usage_info",
    ],
    AgentType.PRODUCT.value: [
        "get_plans_info",
        "get_devices_info",
        "get_promotions_info",
        "get_customer_devices",
    ],
}


def get_toolkit_for_domain(domain: str, uc_config: UCConfig) -> UCFunctionToolkit:
    """Get a toolkit for a specific domain.

    Args:
        domain: Domain to get the toolkit for (e.g., "account", "billing")
        uc_config: Unity Catalog configuration

    Returns:
        UCFunctionToolkit: Toolkit containing the appropriate functions
    """
    try:
        # validate against known agent types
        AgentType.from_string(domain)
    except ValueError:
        logger.warning(f"Requested toolkit for non-standard domain: {domain}")

    function_names = get_functions_for_domain(domain, uc_config)
    return UCFunctionToolkit(function_names=function_names)


def get_functions_for_domain(domain: str, uc_config: UCConfig) -> list[str]:
    """Get function names for a specific domain.

    Args:
        domain: Domain to get function names for (e.g., "account", "billing")
        uc_config: Unity Catalog configuration

    Returns:
        List of function names for the domain
    """
    function_names = [
        uc_config.get_uc_function_name(function_name)
        for function_name in DOMAIN_FUNCTION_MAP.get(domain, [])
    ]
    return function_names


def check_function_exists(function_name: str) -> bool:
    """Check if a UC function exists.

    Args:
        function_name: Fully qualified function name (catalog.schema.function)

    Returns:
        True if function exists, False otherwise
    """
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()

    try:
        client.functions.get(function_name)
        logger.info(f"Function {function_name} exists")
        return True
    except Exception as e:
        logger.info(f"Function {function_name} does not exist: {str(e)}")
        return False


def _register_domain_functions(domain: str, uc_config: UCConfig) -> dict[str, bool]:
    """Register UC functions for a specific domain.

    Args:
        domain: Domain to register functions for
        uc_config: Unity Catalog configuration

    Returns:
        Dict mapping function names to registration status
    """
    import importlib

    # Domain to module mapping
    domain_modules = {
        AgentType.ACCOUNT.value: "telco_support_agent.tools.account.functions",
        AgentType.BILLING.value: "telco_support_agent.tools.billing.functions",
        AgentType.PRODUCT.value: "telco_support_agent.tools.product.functions",
    }

    results = {}

    if domain not in domain_modules:
        logger.warning(f"No function module found for domain: {domain}")
        return dict.fromkeys(get_functions_for_domain(domain, uc_config), False)

    # import module to trigger registration
    try:
        logger.info(f"Importing functions for domain: {domain}")
        importlib.import_module(domain_modules[domain])

        for func_name in get_functions_for_domain(domain, uc_config):
            results[func_name] = check_function_exists(func_name)

        return results
    except Exception as e:
        logger.error(f"Error registering functions for domain {domain}: {e}")
        return dict.fromkeys(get_functions_for_domain(domain, uc_config), False)
