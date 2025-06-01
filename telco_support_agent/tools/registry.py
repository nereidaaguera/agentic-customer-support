"""Utilities for registering and accessing UC functions."""

from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from telco_support_agent.agents.types import AgentType
from telco_support_agent.utils.config import config_manager
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

# mapping of domains to function names
DOMAIN_FUNCTION_MAP = {
    AgentType.ACCOUNT.value: [
        "get_customer_info",
        "get_customer_subscriptions",
    ],
    AgentType.BILLING.value: [
        "get_billing_info",
        "get_usage_info",
    ],
    AgentType.TECH_SUPPORT.value: [
        # TODO: add tech support functions
    ],
    AgentType.PRODUCT.value: [
        "get_plans_info",
        "get_devices_info",
        "get_promotions_info",
        "get_customer_devices",
    ],
}


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

    function_names = get_functions_for_domain(domain)
    return UCFunctionToolkit(function_names=function_names)


def get_functions_for_domain(domain: str) -> list[str]:
    """Get function names for a specific domain.

    Args:
        domain: Domain to get function names for (e.g., "account", "billing")

    Returns:
        List of function names for the domain
    """
    uc_config = config_manager.get_uc_config()
    function_names = [
        f"{uc_config.agent['catalog']}.{uc_config.agent['schema']}.{function_name}"
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


def _register_domain_functions(domain: str) -> dict[str, bool]:
    """Register UC functions for a specific domain.

    Args:
        domain: Domain to register functions for

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
        return dict.fromkeys(get_functions_for_domain(domain), False)

    # import module to trigger registration
    try:
        importlib.import_module(domain_modules[domain])

        for func_name in get_functions_for_domain(domain):
            results[func_name] = check_function_exists(func_name)

        return results
    except Exception as e:
        logger.error(f"Error registering functions for domain {domain}: {e}")
        return dict.fromkeys(get_functions_for_domain(domain), False)
