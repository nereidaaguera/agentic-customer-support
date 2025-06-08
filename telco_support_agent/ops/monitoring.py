"""Monitoring utils for deployed agents."""

from databricks.agents.monitoring import (
    AssessmentsSuiteConfig,
    create_external_monitor,
    delete_external_monitor,
    get_external_monitor,
)

from telco_support_agent.utils.config import UCConfig
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentMonitoringError(Exception):
    """Raised when agent monitoring operations fail."""

    pass


def create_agent_monitor(
    uc_config: UCConfig,
    experiment_name: str,
    replace_existing: bool = True,
) -> any:
    """Create an external monitor for the deployed agent with minimal configuration.

    Args:
        uc_config: Unity Catalog configuration
        experiment_name: MLflow experiment name
        replace_existing: Whether to replace existing monitor

    Returns:
        Created external monitor

    Raises:
        AgentMonitoringError: If monitor creation fails
    """
    try:
        # Check if monitor already exists
        if replace_existing:
            try:
                get_external_monitor(experiment_name=experiment_name)
                logger.info(f"Found existing monitor for experiment: {experiment_name}")
                logger.info("Deleting existing monitor for replacement...")
                delete_external_monitor(experiment_name=experiment_name)
            except ValueError:
                # No existing monitor found, which is fine
                logger.info(
                    f"No existing monitor found for experiment: {experiment_name}"
                )

        # Create monitor with empty assessments for now
        logger.info(f"Creating external monitor for experiment: {experiment_name}")

        # Simple config with no assessments initially
        assessments_config = AssessmentsSuiteConfig(
            sample=0.1,  # Sample 10% of traces
            paused=False,  # Start monitoring immediately
            assessments=[],  # Empty assessments array for first deployment
        )

        monitor = create_external_monitor(
            catalog_name=uc_config.data["catalog"],
            schema_name=uc_config.data["schema"],
            assessments_config=assessments_config,
            experiment_name=experiment_name,
        )

        logger.info("Successfully created external monitor with empty assessments")
        return monitor

    except Exception as e:
        error_msg = f"Failed to create agent monitor: {str(e)}"
        logger.error(error_msg)
        raise AgentMonitoringError(error_msg) from e


def delete_agent_monitor(experiment_name: str) -> None:
    """Delete external monitor for an experiment.

    Args:
        experiment_name: MLflow experiment name

    Raises:
        AgentMonitoringError: If monitor deletion fails
    """
    try:
        logger.info(f"Deleting external monitor for experiment: {experiment_name}")
        delete_external_monitor(experiment_name=experiment_name)
        logger.info("Successfully deleted external monitor")

    except Exception as e:
        error_msg = f"Failed to delete agent monitor: {str(e)}"
        logger.error(error_msg)
        raise AgentMonitoringError(error_msg) from e
