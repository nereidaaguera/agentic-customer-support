"""Config utilities for loading agent configs."""

from pathlib import Path
from typing import Any

import yaml


def load_agent_config(agent_type: str) -> dict[str, Any]:
    """Load agent configuration from YAML file.

    Args:
        agent_type: Type of agent to load config for (e.g., "supervisor", "account")

    Returns:
        Dict containing agent configuration
    """
    config_path = (
        Path(__file__).parent.parent / "configs" / "agents" / f"{agent_type}.yaml"
    )

    if not config_path.exists():
        raise ValueError(f"No configuration found for agent type: {agent_type}")

    with open(config_path) as f:
        return yaml.safe_load(f)
