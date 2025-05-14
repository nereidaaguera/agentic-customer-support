"""Base classes and interfaces for all agent tools."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Union

from telco_support_agent.config import get_uc_config
from telco_support_agent.tools.tool_info import ToolInfo

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Base tool interface for all agent tools."""

    @abstractmethod
    def get_tool_info(self) -> ToolInfo:
        """Return tool info for agent consumption."""
        pass

    @abstractmethod
    def exec_fn(self, **kwargs: dict[str, Any]) -> Any:
        """Execute the tool with the given parameters."""
        pass


class FunctionType(Enum):
    """Enum for function types."""

    SQL = 1
    PYTHON = 2


class UCTool(Tool):
    """Tool implementation for Unity Catalog functions."""

    def __init__(
        self,
        function_name: str,
        function_type: FunctionType,
        description: str,
        env: str = None,
    ) -> None:
        """Initialize UC Tool.

        Args:
            function_name: Function name
            function_type: Type of function (SQL or PYTHON)
            description: Tool description
            env: Environment name (dev, prod)
        """
        from unitycatalog.ai.core.databricks import DatabricksFunctionClient

        uc_config = get_uc_config(env)
        self.client = DatabricksFunctionClient()
        self.catalog = uc_config["catalog"]
        self.schema = uc_config["schema"]
        self.function_name = function_name
        self.description = description
        self.uc_name = f"{self.catalog}.{self.schema}.{self.function_name}"
        self.function_type = function_type
        self.spec = self.create_function(self.create_function_value())
        # Remove strict parameter from function. it does not work with claude.
        if "strict" in self.spec.get("function", {}):
            self.spec["function"].pop("strict")

    def create_function(self, function_value: Union[str, Callable]) -> dict[str, Any]:
        """Create tool function in unity catalog.

        Args:
            function_value: SQL or Python function to create

        Returns:
            Function specification
        """
        try:
            from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

            if self.function_type is FunctionType.SQL:
                self.client.create_function(sql_function_body=function_value)
            else:
                self.client.create_python_function(
                    func=function_value, catalog=self.catalog, schema=self.schema
                )
            toolkit = UCFunctionToolkit(function_names=[self.uc_name])
            return toolkit.tools[0]
        except Exception as e:
            logger.error(f"Failed to create function in Unity Catalog: {e}")
            raise

    @abstractmethod
    def create_function_value(self) -> Union[str, Callable]:
        """Create function value for the tool."""
        pass

    def exec_fn(self, **kwargs: dict[str, Any]) -> Any:
        """Execute the function in unity catalog.

        Args:
            **kwargs: Parameters to pass to the function

        Returns:
            Function execution result
        """
        try:
            output = self.client.execute_function(self.uc_name, parameters=kwargs)
            return output.value
        except Exception as e:
            logger.error(f"Failed to execute function {self.uc_name}: {e}")
            return f"Error executing tool: {str(e)}"

    def get_tool_info(self) -> ToolInfo:
        """Return tool info to an agent inherent from BaseAgent."""
        return ToolInfo(
            name=self.uc_name.replace(".", "__"),
            spec=self.spec,
            exec_fn=self.exec_fn,
        )


class PythonTool(Tool):
    """Tool implementation for Python functions."""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize Python Tool.

        Args:
            name: Tool name
            description: Tool description
            function: Python function to execute
            parameters: Parameter specifications for the function
        """
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
        self.spec = self._create_tool_spec()

    def _create_tool_spec(self) -> dict[str, Any]:
        """Create tool specification for LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }

    def exec_fn(self, **kwargs: dict[str, Any]) -> Any:
        """Execute the Python function.

        Args:
            **kwargs: Parameters to pass to the function

        Returns:
            Function execution result
        """
        try:
            return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Failed to execute Python function {self.name}: {e}")
            return f"Error executing tool: {str(e)}"

    def get_tool_info(self) -> ToolInfo:
        """Return tool info for agent consumption."""
        return ToolInfo(name=self.name, spec=self.spec, exec_fn=self.exec_fn)


class ToolRegistry:
    """Registry for tool discovery and management."""

    _tools = {}

    @classmethod
    def register_tool(cls, agent_type: str, tool: Tool) -> None:
        """Register a tool for a specific agent type.

        Args:
            agent_type: Type of agent (e.g., "account", "billing")
            tool: Tool instance to register
        """
        if agent_type not in cls._tools:
            cls._tools[agent_type] = []
        cls._tools[agent_type].append(tool)

    @classmethod
    def get_tools(cls, agent_type: str) -> list[Tool]:
        """Get all tools for a specific agent type.

        Args:
            agent_type: Type of agent (e.g., "account", "billing")

        Returns:
            List of registered tools for the agent type
        """
        return cls._tools.get(agent_type, [])

    @classmethod
    def register_tool_for_multiple_agents(
        cls, agent_types: list[str], tool: Tool
    ) -> None:
        """Register a tool for multiple agent types.

        Args:
            agent_types: List of agent types
            tool: Tool instance to register
        """
        for agent_type in agent_types:
            cls.register_tool(agent_type, tool)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the registry (primarily for testing)."""
        cls._tools = {}
