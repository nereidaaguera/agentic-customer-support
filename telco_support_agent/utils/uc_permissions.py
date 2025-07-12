"""Utility functions for managing UC permissions."""

from databricks.sdk import WorkspaceClient

from telco_support_agent.agents import UCConfig
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


def grant_function_permissions(
    function_name: str, uc_config: UCConfig, client: WorkspaceClient = None
) -> bool:
    """Grant permissions on a UC function to configured groups.

    Args:
        function_name: Fully qualified function name (catalog.schema.function)
        uc_config: UC configuration containing permission settings
        client: Optional WorkspaceClient instance

    Returns:
        True if permissions were granted successfully, False otherwise
    """
    if client is None:
        client = WorkspaceClient()

    # check if permissions are configured
    permissions = getattr(uc_config, "permissions", None)
    if not permissions:
        logger.warning(f"No permissions configured for function {function_name}")
        return True

    groups = permissions.get("groups", [])
    privileges = permissions.get("privileges", ["EXECUTE"])

    if not groups:
        logger.warning(
            f"No groups configured for permissions on function {function_name}"
        )
        return True

    success = True
    for group in groups:
        for privilege in privileges:
            try:
                grant_sql = (
                    f"GRANT {privilege} ON FUNCTION {function_name} TO `{group}`"
                )
                logger.info(f"Granting permissions: {grant_sql}")

                client.statement_execution.execute_statement(
                    warehouse_id=_get_warehouse_id(client),
                    statement=grant_sql,
                    wait_timeout="30s",
                )

                logger.info(
                    f"Successfully granted {privilege} on {function_name} to {group}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to grant {privilege} on {function_name} to {group}: {str(e)}"
                )
                success = False

    return success


def _get_warehouse_id(client: WorkspaceClient) -> str:
    """Get the first available warehouse ID for SQL execution.

    Args:
        client: WorkspaceClient instance

    Returns:
        Warehouse ID string

    Raises:
        RuntimeError: If no warehouses are available
    """
    try:
        warehouses = list(client.warehouses.list())
        if not warehouses:
            raise RuntimeError("No SQL warehouses available")

        # use first available warehouse
        warehouse = warehouses[0]
        logger.info(f"Using warehouse: {warehouse.name} ({warehouse.id})")
        return warehouse.id

    except Exception as e:
        logger.error(f"Error getting warehouse ID: {str(e)}")
        raise RuntimeError(f"Failed to get warehouse ID: {str(e)}") from e
