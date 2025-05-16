"""Functions for managing models in Unity Catalog."""

from typing import Optional

import mlflow
from mlflow.entities.model import ModelVersion

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

mlflow.set_registry_uri("databricks-uc")


def register_agent_to_uc(
    model_uri: str,
    uc_model_name: str,
) -> mlflow.entities.model.ModelVersion:
    """Register an agent model to Unity Catalog.

    Args:
        model_uri: URI of the MLflow model to register
        uc_model_name: Fully qualified name in Unity Catalog (catalog.schema.model)

    Returns:
        ModelVersion object containing details of the registered model
    """
    logger.info(f"Registering agent model to Unity Catalog: {uc_model_name}")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=uc_model_name,
    )

    logger.info(
        f"Successfully registered model: {uc_model_name} version {model_version.version}"
    )
    return model_version


def list_model_versions(
    uc_model_name: str,
    max_results: Optional[int] = None,
) -> list[ModelVersion]:
    """List versions of a model in Unity Catalog.

    Args:
        uc_model_name: Fully qualified name in Unity Catalog (catalog.schema.model)
        max_results: Optional maximum number of results to return

    Returns:
        List of ModelVersion objects
    """
    logger.info(f"Listing versions for model: {uc_model_name}")
    versions = mlflow.search_model_versions(f"name='{uc_model_name}'")

    if max_results is not None:
        versions = versions[:max_results]

    logger.info(f"Found {len(versions)} version(s) for model {uc_model_name}")
    return versions


def get_latest_model_version(
    uc_model_name: str,
) -> Optional[ModelVersion]:
    """Get the latest version of a model in Unity Catalog.

    Args:
        uc_model_name: Fully qualified name in Unity Catalog (catalog.schema.model)

    Returns:
        Latest ModelVersion object or None if no versions exist
    """
    versions = list_model_versions(uc_model_name)

    if not versions:
        logger.warning(f"No versions found for model: {uc_model_name}")
        return None

    # Sort by version number (descending)
    versions.sort(key=lambda v: int(v.version), reverse=True)

    latest = versions[0]
    logger.info(f"Latest version for {uc_model_name} is {latest.version}")
    return latest
