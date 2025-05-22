"""Tools package for the telco support agent."""

from typing import Optional

from telco_support_agent.tools.registry import (
    DOMAIN_FUNCTION_MAP,
    _register_domain_functions,
)
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


def initialize_tools(
    domains: Optional[list[str]] = None, agent_config: Optional[dict] = None
) -> dict[str, dict[str, bool]]:
    """Initialize UC functions and tools.

    This can be called with either:
    - A list of domains to initialize
    - An agent config to extract function requirements from
    - Neither, which initializes all implemented domains

    Args:
        domains: Optional list of domain names (account, billing, etc.)
        agent_config: Optional agent configuration dictionary

    Returns:
        Dictionary mapping domains to function registration status
    """
    logger.info("Initializing tools...")

    results = {}

    # Case 1: Initialize from agent config
    if agent_config is not None:
        if not agent_config.get("uc_functions"):
            logger.info("No UC functions specified in agent config")
            return {}

        # Determine which domains we need based on function names
        required_functions = agent_config["uc_functions"]
        required_domains = set()

        for func_name in required_functions:
            for domain, functions in DOMAIN_FUNCTION_MAP.items():
                if func_name in functions:
                    required_domains.add(domain)

        # Register functions for each required domain
        for domain in required_domains:
            results[domain] = _register_domain_functions(domain)

    # Case 2: Initialize specific domains
    elif domains is not None:
        for domain in domains:
            results[domain] = _register_domain_functions(domain)

    # Case 3: Initialize all domains with implemented functions
    else:
        for domain, functions in DOMAIN_FUNCTION_MAP.items():
            if functions:  # only initialize domains with defined functions
                results[domain] = _register_domain_functions(domain)

    total_domains = len(results)
    successful_domains = sum(
        1 for _, functions in results.items() if all(functions.values())
    )

    logger.info(
        f"Tool initialization complete: {successful_domains}/{total_domains} domains fully initialized"
    )

    return results
