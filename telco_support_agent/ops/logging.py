"""Log Agent models to MLflow."""

from typing import Optional, Union

import mlflow
import yaml
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
)
from mlflow.pyfunc import ResponsesAgent

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


def log_agent(
    agent: ResponsesAgent,
    artifact_path: str = "agent",
    resources: Optional[list] = None,
    conda_env: Optional[Union[dict, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    input_example: Optional[dict] = None,
    extra_pip_requirements: Optional[list[str]] = None,
) -> mlflow.models.model.ModelInfo:
    """Log any ResponsesAgent as an MLflow model.

    Args:
        agent: The ResponsesAgent to log
        artifact_path: Path within the MLflow run artifacts to store the model
        resources: Databricks resources needed for automatic authentication
        conda_env: Either a dictionary representation of a Conda environment or
                  the path to a Conda environment yaml file
        pip_requirements: List of pip requirements for the model environment
        input_example: Optional input example for MLflow signature inference
        extra_pip_requirements: Additional pip requirements to add

    Returns:
        ModelInfo object containing details of the logged model
    """
    logger.info(f"Logging agent to MLflow with path: {artifact_path}")

    artifacts = {}

    try:
        from telco_support_agent import PROJECT_ROOT

        # add all agent config files as artifacts
        config_dir = PROJECT_ROOT / "configs" / "agents"
        if config_dir.exists() and config_dir.is_dir():
            for config_file in config_dir.glob("*.yaml"):
                artifact_path = f"configs/agents/{config_file.name}"
                artifacts[artifact_path] = str(config_file)
                logger.info(f"Adding config artifact: {artifact_path}")
    except Exception as e:
        logger.warning(f"Error collecting config artifacts: {e}")

    if resources is None:
        resources = []

        # LLM endpoints
        if hasattr(agent, "llm_endpoint") and agent.llm_endpoint:
            resources.append(
                DatabricksServingEndpoint(endpoint_name=agent.llm_endpoint)
            )

        # Registered functions
        if hasattr(agent, "get_tool_specs"):
            for tool in agent.get_tool_specs():
                if "function" in tool:
                    function_name = tool["function"]["name"].replace("__", ".")
                    resources.append(DatabricksFunction(function_name=function_name))

        logger.info(f"Automatically detected {len(resources)} resources")

    if input_example is None:
        input_example = {
            "input": [{"role": "user", "content": "Hello, how can you help me today?"}]
        }

    with mlflow.start_run():
        # log configs as individual dictionaries
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                with open(config_file) as f:
                    config_dict = yaml.safe_load(f)
                    mlflow.log_dict(config_dict, f"configs/agents/{config_file.name}")

        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=agent,
            input_example=input_example,
            resources=resources,
            conda_env=conda_env,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            artifacts=artifacts,
        )

        logger.info(f"Successfully logged model: {model_info.model_uri}")
        return model_info
