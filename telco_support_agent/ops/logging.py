"""Log Agent models to MLflow using Models from Code approach."""

import inspect
from typing import Optional

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.models.resources import Resource

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


def collect_dependent_files() -> list[str]:
    """Collect all Python files that the agent depends on.

    Args:
        agent_class: The agent class to analyze

    Returns:
        List of file paths that should be included as code_paths
    """
    from telco_support_agent import PROJECT_ROOT

    # core files that supervisor.py depends on
    dependent_files = [
        "telco_support_agent/__init__.py",
        "telco_support_agent/agents/__init__.py",
        "telco_support_agent/agents/types.py",
        "telco_support_agent/agents/base_agent.py",
        "telco_support_agent/agents/account.py",
        "telco_support_agent/agents/config.py",
        "telco_support_agent/utils/__init__.py",
        "telco_support_agent/utils/logging_utils.py",
        "telco_support_agent/tools/__init__.py",
        "telco_support_agent/tools/registry.py",
        "telco_support_agent/config.py",
    ]

    code_paths = []
    for file_path in dependent_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            code_paths.append(str(full_path))
            logger.info(f"Adding code dependency: {file_path}")
        else:
            logger.warning(f"Dependency file not found: {full_path}")

    return code_paths


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

    # collect all dependent files
    code_paths = collect_dependent_files()

    # collect config files as artifacts
    artifacts = {}
    try:
        from telco_support_agent import PROJECT_ROOT

        config_dir = PROJECT_ROOT / "configs" / "agents"
        if config_dir.exists() and config_dir.is_dir():
            for config_file in config_dir.glob("*.yaml"):
                artifact_key = f"configs/agents/{config_file.name}"
                artifacts[artifact_key] = str(config_file)
                logger.info(f"Adding config artifact: {artifact_key}")
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

        # Registered functions
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
        model_info = mlflow.pyfunc.log_model(
            python_model=module_path,  # path to the module file
            name=name,
            input_example=input_example,
            resources=resources,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            code_paths=code_paths,  # include all dependent files
            artifacts=artifacts,
        )

        logger.info(f"Successfully logged model: {model_info.model_uri}")
        return model_info
