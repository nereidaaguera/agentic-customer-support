"""Functions for deploying Telco Support Agent models."""

import time
from typing import Any, Optional, Union

from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointStateConfigUpdate,
    EndpointStateReady,
)
from mlflow.entities.model_registry import ModelVersion

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentDeploymentError(Exception):
    """Raised when agent deployment fails."""

    pass


def deploy_agent(
    uc_model_name: str,
    model_version: Union[str, int, ModelVersion],
    deployment_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    scale_to_zero_enabled: bool = False,
    environment_vars: Optional[dict[str, str]] = None,
    workload_size: str = "Small",
    wait_for_ready: bool = True,
    permissions: Optional[dict] = None,
    instructions: Optional[str] = None,
    budget_policy_id: Optional[str] = None,
) -> Any:
    """Deploy a registered agent model to a Model Serving endpoint.

    Args:
        uc_model_name: Fully qualified name in Unity Catalog (catalog.schema.model)
        model_version: Version of the model to deploy
        deployment_name: Optional name for the deployment
        tags: Optional tags to attach to the deployment
        scale_to_zero_enabled: Whether to enable scale-to-zero for idle endpoints
        environment_vars: Optional environment variables for the deployment
        workload_size: Size of the workload ("Small", "Medium", "Large")
        wait_for_ready: Whether to wait for deployment to be ready
        permissions: Optional permissions configuration dict with 'users' and 'permission_level'
        instructions: Optional review instructions for the Review App
        budget_policy_id: Optional budget policy ID

    Returns:
        Deployment object with information about the deployed agent

    Raises:
        AgentDeploymentError: If deployment fails
    """
    # get version number if ModelVersion object
    if isinstance(model_version, ModelVersion):
        model_version = model_version.version

    logger.info(f"Deploying agent model: {uc_model_name} version {model_version}")
    logger.info(f"Workload size: {workload_size}")
    logger.info(f"Scale to zero: {scale_to_zero_enabled}")

    # set default tags if none provided
    if tags is None:
        tags = {"source": "telco_support_agent"}

    # validate workload size
    valid_sizes = ["Small", "Medium", "Large"]
    if workload_size not in valid_sizes:
        raise ValueError(
            f"Invalid workload size: {workload_size}. Must be one of: {valid_sizes}"
        )

    try:
        # deploy agent
        logger.info("Starting deployment...")
        deployment = agents.deploy(
            model_name=uc_model_name,
            model_version=model_version,
            endpoint_name=deployment_name,
            tags=tags,
            scale_to_zero=scale_to_zero_enabled,
            environment_vars=environment_vars,
            workload_size=workload_size,
            budget_policy_id=budget_policy_id,
        )

        logger.info(
            f"Agent deployment started. Endpoint name: {deployment.endpoint_name}"
        )

        # set review instructions if provided
        if instructions:
            logger.info("Setting review instructions...")
            agents.set_review_instructions(uc_model_name, instructions)

        # wait for endpoint to be ready if requested
        if wait_for_ready:
            logger.info("Waiting for endpoint to be ready...")
            _wait_for_endpoint_ready(deployment.endpoint_name)
            logger.info("Endpoint is ready")

        # set permissions if provided
        if permissions:
            logger.info("Setting agent permissions...")
            _set_permissions(uc_model_name, permissions)
            logger.info("Permissions set successfully")

        logger.info(f"Successfully deployed agent at: {deployment.endpoint_name}")
        logger.info(f"Query endpoint: {deployment.query_endpoint}")

        return deployment

    except Exception as e:
        error_msg = f"Failed to deploy agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise AgentDeploymentError(error_msg) from e


def _wait_for_endpoint_ready(endpoint_name: str) -> None:
    """Wait for endpoint to be ready."""
    logger.info(f"Starting to wait for endpoint {endpoint_name}")
    w = WorkspaceClient()
    iteration = 0

    while True:
        endpoint_state = w.serving_endpoints.get(endpoint_name).state
        ready_state = endpoint_state.ready
        config_state = endpoint_state.config_update

        if (
            ready_state == EndpointStateReady.NOT_READY
            or config_state == EndpointStateConfigUpdate.IN_PROGRESS
        ):
            iteration += 1
            if iteration % 4 == 0:
                logger.info(
                    f"Still waiting for endpoint... (Ready state: {ready_state}, Config state: {config_state})"
                )
            time.sleep(30)
        else:
            break

    logger.info(f"Endpoint {endpoint_name} is ready")


def _set_permissions(model_name: str, permissions: dict) -> None:
    """Set agent permissions.

    Args:
        model_name: Full UC model name
        permissions: Permission configuration dictionary with structure:
            {
                "users": List[str],  # List of user/group names
                "permission_level": str  # One of CAN_VIEW, CAN_RUN, CAN_MANAGE
            }
    """
    logger.info(f"Setting permissions for {model_name}")
    logger.debug(f"Permission config: {permissions}")

    try:
        users = permissions.get("users")
        permission_level = permissions.get("permission_level")

        if not users or not permission_level:
            raise ValueError(
                "Both 'users' and 'permission_level' must be specified in permissions config"
            )

        logger.info(
            f"Setting {permission_level} permission for users: {', '.join(users)}"
        )

        agents.set_permissions(
            model_name=model_name,
            users=users,
            permission_level=getattr(agents.PermissionLevel, permission_level),
        )
        logger.info("Permissions set successfully")

    except ValueError as e:
        error_msg = f"Invalid permission configuration: {str(e)}"
        logger.error(error_msg)
        raise AgentDeploymentError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to set permissions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise AgentDeploymentError(error_msg) from e
