"""Log Agent models to MLflow using Models from Code approach."""

import inspect
from pathlib import Path
from typing import Optional

import mlflow
import yaml
from mlflow.models.model import ModelInfo
from mlflow.models.resources import Resource

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


def log_agent(
    agent_class: type,
    name: str = "agent",
    resources: Optional[list[Resource]] = None,
    pip_requirements: Optional[list[str]] = None,
    input_example: Optional[dict] = None,
    extra_pip_requirements: Optional[list[str]] = None,
) -> ModelInfo:
    """Log agent using MLflow Models from Code approach.

    https://mlflow.org/docs/latest/model/models-from-code

    Use Models from Code approach to log the agent,
    looking up the source file of the agent's class and passing that
    to MLflow. This requires the agent module to call `set_model()`
    at the module level.

    Args:
        agent_class: The agent class to log (e.g., SupervisorAgent)
        name: Name for the model in MLflow (replaces deprecated artifact_path)
        resources: Databricks resources needed for automatic authentication
        pip_requirements: List of pip requirements
        input_example: Optional input example for MLflow signature inference
        extra_pip_requirements: Additional pip requirements

    Returns:
        ModelInfo object containing details of the logged model
    """
    # get path to the agent's module file
    module_path = inspect.getfile(agent_class)
    logger.info(f"Using agent file: {module_path}")

    # get package root directory
    try:
        from telco_support_agent import PROJECT_ROOT

        package_root = PROJECT_ROOT
        logger.info(f"Using package root: {package_root}")
    except ImportError:
        current_path = Path(module_path).parent
        package_root = None

        for _ in range(10):
            if (current_path / "telco_support_agent").exists():
                package_root = current_path
                break
            parent = current_path.parent
            if parent == current_path:
                break
            current_path = parent

        if package_root is None:
            raise ValueError(
                "Could not find telco_support_agent package root"
            ) from None

        logger.info(f"Using fallback package root: {package_root}")

    package_root_str = str(package_root)
    logger.info(f"Package root path: {package_root_str}")

    # collect config files as artifacts
    artifacts = {}
    try:
        config_dir = package_root / "configs" / "agents"
        logger.debug(f"Looking for config files in: {config_dir}")

        if config_dir.exists() and config_dir.is_dir():
            for config_file in config_dir.glob("*.yaml"):
                artifact_key = f"configs/agents/{config_file.name}"
                artifacts[artifact_key] = str(config_file)
                logger.info(f"Adding config artifact: {artifact_key}")
        else:
            logger.warning(
                f"Config directory doesn't exist or isn't a directory: {config_dir}"
            )
    except Exception as e:
        logger.warning(f"Error collecting config artifacts: {e}", exc_info=True)

    if resources is None:
        agent_instance = agent_class()
        resources = []

        # LLM endpoints
        if hasattr(agent_instance, "llm_endpoint") and agent_instance.llm_endpoint:
            from mlflow.models.resources import DatabricksServingEndpoint

            resources.append(
                DatabricksServingEndpoint(endpoint_name=agent_instance.llm_endpoint)
            )

        # registered functions
        if hasattr(agent_instance, "get_tool_specs"):
            from mlflow.models.resources import DatabricksFunction

            for tool in agent_instance.get_tool_specs():
                if "function" in tool:
                    function_name = tool["function"]["name"].replace("__", ".")
                    resources.append(DatabricksFunction(function_name=function_name))

        logger.info(f"Automatically detected {len(resources)} resources")

    # default input example if none provided
    if input_example is None:
        input_example = {
            "input": [{"role": "user", "content": "Hello, how can you help me today?"}]
        }

    with mlflow.start_run():
        try:
            config_dir = package_root / "configs" / "agents"
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    with open(config_file) as f:
                        config_dict = yaml.safe_load(f)
                        mlflow.log_dict(
                            config_dict, f"configs/agents/{config_file.name}"
                        )
        except Exception as e:
            logger.warning(f"Error logging config dictionaries: {e}")

        model_info = mlflow.pyfunc.log_model(
            python_model=module_path,  # path to the module file
            code_paths=[package_root_str],  # include entire package for imports
            name=name,
            input_example=input_example,
            resources=resources,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            artifacts=artifacts,
        )

        logger.info(f"Successfully logged model: {model_info.model_uri}")
        return model_info
