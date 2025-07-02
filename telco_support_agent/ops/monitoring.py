"""Monitoring utils for deployed agents."""

from typing import Optional

from databricks.agents.monitoring import (
    AssessmentsSuiteConfig,
    CustomMetric,
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
    sample: float = None,
    custom_metrics: Optional[list] = None,
) -> any:
    """Create an external monitor for the deployed agent with custom metrics.

    Args:
        uc_config: Unity Catalog configuration
        experiment_name: MLflow experiment name
        replace_existing: Whether to replace existing monitor
        sample: Sampling rate for traces (0.0 < rate <= 1.0)
        custom_metrics: List of custom metric functions to use

    Returns:
        Created external monitor

    Raises:
        AgentMonitoringError: If monitor creation fails
    """
    try:
        if replace_existing:
            try:
                get_external_monitor(experiment_name=experiment_name)
                logger.info(f"Found existing monitor for experiment: {experiment_name}")
                logger.info("Deleting existing monitor for replacement...")
                delete_external_monitor(experiment_name=experiment_name)
            except ValueError:
                logger.info(
                    f"No existing monitor found for experiment: {experiment_name}"
                )

        # create monitor with custom metrics
        logger.info(f"Creating external monitor for experiment: {experiment_name}")
        logger.info(f"Using agent catalog: {uc_config.agent['catalog']}")
        logger.info(f"Using agent schema: {uc_config.agent['schema']}")

        # prepare assessments list
        assessments = []
        if custom_metrics:
            for metric_func in custom_metrics:
                metric_name = getattr(metric_func, "__name__", "unknown_metric")
                logger.info(f"Adding custom metric: {metric_name}")
                assessments.append(CustomMetric(metric_func))

        logger.info(f"Monitor will use {len(assessments)} custom assessments")

        assessments_config = AssessmentsSuiteConfig(
            sample=sample,
            paused=False,
            assessments=assessments,
        )

        monitor = create_external_monitor(
            catalog_name=uc_config.agent["catalog"],
            schema_name=uc_config.agent["schema"],
            assessments_config=assessments_config,
            experiment_name=experiment_name,
        )

        logger.info(
            "Successfully created external monitor with custom telco assessments"
        )
        logger.info(
            f"Monitor will create tables in: {uc_config.agent['catalog']}.{uc_config.agent['schema']}"
        )
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
