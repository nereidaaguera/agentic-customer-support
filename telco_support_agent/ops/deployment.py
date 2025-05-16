"""Functions for deploying Telco Support Agent models."""

from typing import Optional, Union

from databricks import agents
from mlflow.entities.model import ModelVersion

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


def deploy_agent(
    uc_model_name: str,
    model_version: Union[str, int, ModelVersion],
    deployment_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    scale_to_zero_enabled: bool = False,
    environment_vars: Optional[dict[str, str]] = None,
) -> agents.Deployment:
    """Deploy a registered agent model to a Model Serving endpoint.

    Args:
        uc_model_name: Fully qualified name in Unity Catalog (catalog.schema.model)
        model_version: Version of the model to deploy
        deployment_name: Optional name for the deployment
        tags: Optional tags to attach to the deployment
        scale_to_zero_enabled: Whether to enable scale-to-zero for idle endpoints
        environment_vars: Optional environment variables for the deployment

    Returns:
        Deployment object with information about the deployed agent
    """
    # get version number if ModelVersion object
    if isinstance(model_version, ModelVersion):
        model_version = model_version.version

    logger.info(f"Deploying agent model: {uc_model_name} version {model_version}")

    # set default tags if none provided
    if tags is None:
        tags = {"source": "telco_support_agent"}

    deployment = agents.deploy(
        model_name=uc_model_name,
        model_version=model_version,
        endpoint_name=deployment_name,
        tags=tags,
        scale_to_zero_enabled=scale_to_zero_enabled,
        environment_vars=environment_vars,
    )

    logger.info(f"Successfully deployed agent at: {deployment.endpoint_name}")
    logger.info(f"Query endpoint: {deployment.query_endpoint}")

    return deployment
