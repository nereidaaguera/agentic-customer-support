"""Utilities for registering and accessing UC functions."""

import importlib

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


def get_functions_for_domain(domain: str) -> list[str]:
    """Get function names for a specific domain.

    Args:
        domain: Domain to get function names for (e.g., "account", "billing")

    Returns:
        List of function names for the domain
    """
    try:
        toolkit = get_toolkit_for_domain(domain)
        return [t["function"]["name"].replace("__", ".") for t in toolkit.tools]
    except Exception as e:
        logger.error(f"Error getting functions for domain {domain}: {e}")
        return []


def _get_registration_functions(domain: str) -> dict[str, callable]:
    """Get registration functions for a domain.

    Args:
        domain: Domain to get registration functions for

    Returns:
        Dict mapping function names to registration functions
    """
    registration_map = {}

    try:
        if domain == AgentType.ACCOUNT.value:
            module = importlib.import_module(
                "telco_support_agent.tools.account.functions"
            )

            registration_map = {
                f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.get_customer_info": module.register_customer_info,
                f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.get_customer_subscriptions": module.register_customer_subscriptions,
            }
        elif domain == AgentType.BILLING.value:
            # TODO: Add when billing functions are implemented
            logger.warning(f"No registration functions available for domain: {domain}")
        elif domain == AgentType.TECH_SUPPORT.value:
            # TODO: Add when tech support functions are implemented
            logger.warning(f"No registration functions available for domain: {domain}")
        elif domain == AgentType.PRODUCT.value:
            # TODO: Add when product functions are implemented
            logger.warning(f"No registration functions available for domain: {domain}")
        else:
            logger.warning(f"Unknown domain: {domain}")

    except ImportError as e:
        logger.error(f"Error importing registration module for domain {domain}: {e}")

    return registration_map


def register_required_functions(required_functions: list[str]) -> dict[str, bool]:
    """Register specific UC functions.

    Args:
        required_functions: List of function names to register

    Returns:
        Dict mapping function names to registration success
    """
    results = {}

    domain_functions = {}
    for func_name in required_functions:
        domain = None
        if "customer_info" in func_name or "customer_subscriptions" in func_name:
            domain = AgentType.ACCOUNT.value
        elif "billing" in func_name:
            domain = AgentType.BILLING.value
        elif "tech_support" in func_name:
            domain = AgentType.TECH_SUPPORT.value
        elif "product" in func_name:
            domain = AgentType.PRODUCT.value

        if domain:
            if domain not in domain_functions:
                domain_functions[domain] = []
            domain_functions[domain].append(func_name)
        else:
            logger.warning(f"Could not determine domain for function: {func_name}")
            results[func_name] = False

    for domain, funcs in domain_functions.items():
        registration_map = _get_registration_functions(domain)

        for func_name in funcs:
            if func_name in registration_map:
                try:
                    logger.info(f"Registering {func_name}...")
                    registration_map[func_name]()
                    results[func_name] = True
                    logger.info(f"Successfully registered {func_name}")
                except Exception as e:
                    logger.error(f"Error registering {func_name}: {e}")
                    results[func_name] = False
            else:
                logger.warning(f"No registration function found for {func_name}")
                results[func_name] = False

    success_count = sum(1 for success in results.values() if success)
    logger.info(
        f"Registration complete. Successfully registered {success_count}/{len(required_functions)} functions"
    )

    return results


def register_functions_for_domain(domain: str) -> dict[str, bool]:
    """Register all UC functions for a specific domain.

    Args:
        domain: Domain to register functions for

    Returns:
        Dict mapping function names to registration success
    """
    function_names = []
    try:
        function_names = get_functions_for_domain(domain)
        if not function_names:
            logger.warning(f"No functions defined for domain: {domain}")
            return {}
    except Exception as e:
        logger.error(f"Error getting functions for domain {domain}: {e}")
        return {}

    logger.info(f"Registering {len(function_names)} functions for domain {domain}")
    return register_required_functions(function_names)


def register_functions_for_agent_config(agent_config: dict) -> dict[str, bool]:
    """Register UC functions required by an agent config.

    Args:
        agent_config: Agent configuration dictionary

    Returns:
        Dict mapping function names to registration success
    """
    if not agent_config.get("uc_functions"):
        logger.info("No UC functions required by this agent config")
        return {}

    required_functions = agent_config["uc_functions"]
    logger.info(f"Registering {len(required_functions)} functions required by agent")
    return register_required_functions(required_functions)
