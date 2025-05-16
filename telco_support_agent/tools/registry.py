"""Utilities for registering UC functions."""

from collections.abc import Callable

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

# default catalog and schema
DEFAULT_CATALOG = "telco_customer_support_dev"
DEFAULT_SCHEMA = "agent"

# global client
_uc_client = None

# registry of UC functions by domain
_domain_registry: dict[str, list[str]] = {
    "account": [],
    "billing": [],
    "tech_support": [],
    "product": [],
    "supervisor": [],
}


def get_uc_client() -> DatabricksFunctionClient:
    """Get or create a UC function client.

    Returns:
        DatabricksFunctionClient: The UC function client
    """
    global _uc_client
    if _uc_client is None:
        _uc_client = DatabricksFunctionClient()
    return _uc_client


def uc_function(
    domain: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    replace: bool = True,
) -> Callable:
    """Decorator to register a function as a UC function.

    Args:
        domain: Domain the function belongs to (account, billing, etc.)
        catalog: Catalog name
        schema: Schema name
        replace: Whether to replace an existing function

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        # register the function with UC
        client = get_uc_client()
        function_name = f"{catalog}.{schema}.{func.__name__}"

        try:
            client.create_python_function(
                func=func, catalog=catalog, schema=schema, replace=replace
            )
            logger.info(f"Registered UC function {function_name}")

            # add to domain registry
            if domain in _domain_registry:
                if function_name not in _domain_registry[domain]:
                    _domain_registry[domain].append(function_name)
            else:
                _domain_registry[domain] = [function_name]

        except Exception as e:
            logger.error(f"Failed to register UC function {func.__name__}: {str(e)}")

        return func

    return decorator


def get_toolkit_for_domain(domain: str) -> UCFunctionToolkit:
    """Get a toolkit for a specific domain.

    Args:
        domain: Domain to get the toolkit for

    Returns:
        UCFunctionToolkit for the specified domain
    """
    function_names = _domain_registry.get(domain, [])

    if not function_names:
        logger.warning(f"No UC functions registered for domain: {domain}")
        return UCFunctionToolkit(function_names=[])

    try:
        logger.info(
            f"Creating toolkit for domain {domain} with {len(function_names)} functions"
        )
        return UCFunctionToolkit(function_names=function_names)
    except Exception as e:
        logger.error(f"Error creating toolkit for domain {domain}: {str(e)}")
        return UCFunctionToolkit(function_names=[])


def register_sql_function(
    sql_function_body: str, domain: str, replace: bool = True
) -> None:
    """Register SQL function with Unity Catalog.

    Args:
        sql_function_body: SQL function body (CREATE OR REPLACE FUNCTION statement)
        domain: Domain the function belongs to
        replace: Whether to replace an existing function
    """
    client = get_uc_client()

    try:
        # register function
        client.create_function(sql_function_body=sql_function_body)

        # Extract function name from SQL body
        import re

        name_match = re.search(r"FUNCTION\s+([^\s(]+)", sql_function_body)
        if name_match:
            function_name = name_match.group(1)
            logger.info(f"Registered SQL function {function_name}")

            # add to domain registry
            if domain in _domain_registry:
                if function_name not in _domain_registry[domain]:
                    _domain_registry[domain].append(function_name)
            else:
                _domain_registry[domain] = [function_name]
        else:
            logger.warning("Could not extract function name from SQL body")

    except Exception as e:
        logger.error(f"Failed to register SQL function: {str(e)}")
