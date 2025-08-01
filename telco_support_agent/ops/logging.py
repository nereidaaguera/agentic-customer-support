"""Log Agent models to MLflow."""

import inspect
import os
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import yaml
from mlflow.models.model import ModelInfo
from mlflow.models.resources import (
    DatabricksApp,
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    Resource,
)
from mlflow.types.responses import ResponsesAgentRequest

from telco_support_agent import PACKAGE_DIR, PROJECT_ROOT
from telco_support_agent.config import UCConfig
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


def log_agent(
    agent_class: type,
    name: str = "agent",
    input_example: Optional[dict] = None,
    resources: Optional[list[Resource]] = None,
    environment: str = "prod",
    disable_tools: Optional[list[str]] = None,
    uc_config: Optional[UCConfig] = None,
    config_dir: Optional[Path] = None,
) -> ModelInfo:
    """Log agent using MLflow Models from Code approach.

    Args:
        agent_class: The agent class to log (e.g., SupervisorAgent)
        name: Name for the model in MLflow
        input_example: Optional input example for MLflow signature inference
        resources: Optional list of resources (if None, will auto-detect)
        environment: Environment for resource detection (dev, prod)
        disable_tools: Optional list of tool names to disable. Can be simple names or full UC function names.
        uc_config: Unity Catalog configuration object
        config_dir: Optional directory containing agent config files (defaults to PROJECT_ROOT/configs/agents)

    Returns:
        ModelInfo object containing details of the logged model
    """
    module_path = inspect.getfile(agent_class)

    logger.info(f"Using agent file: {module_path}")
    logger.info(f"Using package directory: {PACKAGE_DIR}")
    logger.info(f"Using project root: {PROJECT_ROOT}")

    # default config dir if not provided
    if config_dir is None:
        config_dir = PROJECT_ROOT / "configs" / "agents"
    logger.info(f"Using config directory: {config_dir}")

    # default UC config if not provided
    if uc_config is None:
        uc_config = UCConfig(
            agent_catalog=f"telco_customer_support_{environment}",
            agent_schema="agent",
            data_schema="gold",
            model_name="telco_customer_support_agent",
        )

    # save uc_config in config directory.
    uc_config_file_path = Path(config_dir, "uc_config.yaml")
    with open(uc_config_file_path, "w") as file:
        yaml.dump(uc_config.model_dump(), file, default_flow_style=False)

    # config artifacts
    artifacts = _collect_config_artifacts(config_dir)

    # auto-detect resources if not provided
    if resources is None:
        resources = _get_supervisor_resources(
            uc_config.agent_catalog,
            uc_config.agent_schema,
            uc_config.data_schema,
            config_dir,
        )

    if input_example is None:
        input_example = {
            "input": [{"role": "user", "content": "Hello, how can you help me today?"}],
            "custom_inputs": {"customer": "CUS-10001"},
        }

    logger.info(f"Using input example: {input_example}")
    _validate_agent_with_custom_inputs(agent_class, input_example)

    extra_pip_requirements = _get_requirements()

    with mlflow.start_run():
        _log_config_dicts(uc_config, config_dir)

        if disable_tools:
            import json

            temp_dir = tempfile.gettempdir()
            disable_tools_path = os.path.join(temp_dir, "disable_tools.json")
            with open(disable_tools_path, "w") as f:
                json.dump({"disable_tools": disable_tools}, f)
            artifacts["disable_tools.json"] = disable_tools_path
            logger.info(f"Adding disable_tools artifact: {disable_tools}")

        model_info = mlflow.pyfunc.log_model(
            python_model=module_path,
            code_paths=[str(PACKAGE_DIR)],
            name=name,
            input_example=input_example,
            resources=resources,
            extra_pip_requirements=extra_pip_requirements,
            artifacts=artifacts,
        )

        logger.info(f"Successfully logged model: {model_info.model_uri}")
        return model_info


def _validate_agent_with_custom_inputs(agent_class: type, input_example: dict) -> None:
    """Validate that agent works properly with custom inputs.

    Args:
        agent_class: The agent class to validate
        input_example: Input example to test with

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating agent with custom inputs...")

    try:
        agent = agent_class()

        request = ResponsesAgentRequest(**input_example)
        response = agent.predict(request)

        if not response or not response.output:
            raise ValueError("Agent returned empty response")

        logger.info("Agent validation with custom inputs successful")

    except Exception as e:
        logger.error(f"Agent validation failed: {str(e)}")
        raise ValueError(f"Agent validation failed: {str(e)}") from e


def _collect_config_artifacts(config_dir: Path) -> dict[str, str]:
    """Collect configuration files as artifacts."""
    artifacts = {}

    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            artifact_key = f"configs/agents/{config_file.name}"
            artifacts[artifact_key] = str(config_file)
            logger.info(f"Adding config artifact: {artifact_key}")

    # topics.yaml file
    topics_config_path = config_dir / "topics.yaml"
    if topics_config_path.exists():
        topics_config_key = "configs/agents/topics.yaml"
        artifacts[topics_config_key] = str(topics_config_path)
        logger.info(f"Adding topics config file as artifact: {topics_config_key}")
    else:
        logger.warning(
            "topics.yaml not found - topic classification may not work in deployment"
        )

    # uc_config.yaml file.
    uc_config_path = config_dir / "uc_config.yaml"
    if uc_config_path.exists():
        uc_config_key = "configs/agents/uc_config.yaml"
        artifacts[uc_config_key] = str(uc_config_path)
        logger.info(f"Adding uc config file as artifact: {uc_config_key}")
    else:
        logger.warning(
            "uc_config.yaml not found - UC Config may not work in deployment"
        )

    return artifacts


def _get_supervisor_resources(
    uc_catalog: str, agent_schema: str, data_schema: str, config_dir: Path
) -> list[Resource]:
    """Get all resources needed by the supervisor agent."""
    import yaml

    from telco_support_agent.tools.registry import DOMAIN_FUNCTION_MAP

    resources = []

    # Get configs for all available agent types by scanning files
    agent_configs = {}

    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            agent_type = config_file.stem
            try:
                with open(config_file) as f:
                    agent_configs[agent_type] = yaml.safe_load(f)
                    logger.info(f"Loaded config for {agent_type} agent")
            except Exception as e:
                logger.warning(f"Could not load {agent_type} config: {e}")

    # LLM endpoints
    llm_endpoints = set()
    for _, config in agent_configs.items():
        if "llm" in config and "endpoint" in config["llm"]:
            endpoint = config["llm"]["endpoint"]
            if endpoint not in llm_endpoints:
                resources.append(DatabricksServingEndpoint(endpoint_name=endpoint))
                llm_endpoints.add(endpoint)
                logger.info(f"Added LLM endpoint: {endpoint}")

    # UC functions based on domain function map
    uc_functions = set()
    for functions in DOMAIN_FUNCTION_MAP.values():
        for func_name in functions:
            uc_func_name = f"{uc_catalog}.{agent_schema}.{func_name}"
            if uc_func_name not in uc_functions:
                resources.append(DatabricksFunction(function_name=uc_func_name))
                uc_functions.add(uc_func_name)
                logger.info(f"Added UC function: {uc_func_name}")

    # Handle custom MCP server app dependencies
    databricks_app_dependencies = set()
    for _, config in agent_configs.items():
        for mcp_server_spec in config.get("mcp_servers", []):
            if app_name := mcp_server_spec.get("app_name"):
                resources.append(DatabricksApp(app_name=app_name))
                databricks_app_dependencies.add(app_name)
                logger.info(f"Added Databricks app dependency: {app_name}")

    # System functions used by agents
    system_functions = ["system.ai.python_exec"]
    for sys_func in system_functions:
        resources.append(DatabricksFunction(function_name=sys_func))
        logger.info(f"Added system function: {sys_func}")

    # Vector Search indexes - always use prod catalog for data reading
    data_catalog = "telco_customer_support_prod"
    vector_indexes = [
        f"{data_catalog}.{data_schema}.knowledge_base_index",
        f"{data_catalog}.{data_schema}.support_tickets_index",
    ]

    for index_name in vector_indexes:
        resources.append(DatabricksVectorSearchIndex(index_name=index_name))
        logger.info(f"Added vector search index: {index_name}")

    logger.info(f"Total resources: {len(resources)}")
    return resources


def _get_requirements() -> list[str]:
    """Get pip requirements from requirements.txt."""
    requirements_path = PROJECT_ROOT / "requirements.txt"

    if not requirements_path.exists():
        logger.warning("requirements.txt not found")
        return []

    try:
        with requirements_path.open("r") as f:
            requirements = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        logger.info(f"Found {len(requirements)} requirements in requirements.txt")
        return requirements
    except Exception as e:
        logger.warning(f"Error reading requirements.txt: {e}")
        return []


def _log_config_dicts(uc_config: UCConfig, config_dir: Path) -> None:
    """Log configuration files as MLflow dictionaries."""
    # Log UC config
    uc_config_data = uc_config.model_dump()
    mlflow.log_dict(uc_config_data, "uc_config.yaml")
    logger.info(f"Logged UC config: {uc_config_data}")

    # Log agent config files
    if not config_dir.exists():
        return

    for config_file in config_dir.glob("*.yaml"):
        try:
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
                mlflow.log_dict(config_dict, f"configs/agents/{config_file.name}")
        except Exception as e:
            logger.warning(f"Error logging config {config_file.name}: {e}")
