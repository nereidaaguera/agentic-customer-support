"""Tool implementations for agent use."""

from collections.abc import Callable
from typing import Any, Optional

from pydantic import BaseModel

from telco_support_agent.tools.base import (
    FunctionType,
    PythonTool,
    Tool,
    ToolRegistry,
    UCTool,
)


class ToolParameter(BaseModel):
    """Model for tool parameter configuration."""

    type: str
    description: str
    enum: Optional[list[str]] = None


class ToolConfig(BaseModel):
    """Model for tool configuration from YAML."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]


class ToolInfo:
    """Tool information container with execution function."""

    def __init__(self, name: str, spec: dict[str, Any], exec_fn: Callable):
        """Initialize tool info.

        Args:
            name: Tool name
            spec: Tool specification for LLM
            exec_fn: Function to execute when tool is called
        """
        self.name = name
        self.spec = spec
        self.exec_fn = exec_fn
