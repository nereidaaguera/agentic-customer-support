"""Tool parameter injection utilities."""

from collections import defaultdict
from typing import Any

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ToolParameterInjector:
    """Handle injection of runtime params into tool calls."""

    def __init__(self, inject_params: list[str]):
        """Initialize the parameter injector.

        Args:
            inject_params: List of parameter names to inject at runtime
        """
        self.inject_params = inject_params
        self.tools_with_injected_params: dict[str, list[str]] = defaultdict(list)

    def prepare_tool_spec_for_llm(self, tool_spec: dict[str, Any]) -> dict[str, Any]:
        """Remove injected parameters from tool spec for LLM consumption.

        Args:
            tool_spec: Original tool specification

        Returns:
            Tool specification with injected parameters removed
        """
        if "function" not in tool_spec:
            return tool_spec

        # deep copy - avoid modifying original
        cleaned_spec = {
            "type": "function",
            "function": {
                "name": tool_spec["function"]["name"],
                "description": tool_spec["function"].get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        if "parameters" in tool_spec["function"]:
            if "properties" in tool_spec["function"]["parameters"]:
                cleaned_spec["function"]["parameters"]["properties"] = tool_spec[
                    "function"
                ]["parameters"]["properties"].copy()

            if "required" in tool_spec["function"]["parameters"]:
                cleaned_spec["function"]["parameters"]["required"] = tool_spec[
                    "function"
                ]["parameters"]["required"].copy()

        func_name = cleaned_spec["function"]["name"]
        parameters = cleaned_spec["function"]["parameters"]["properties"]
        required_params = cleaned_spec["function"]["parameters"]["required"]

        for param in self.inject_params:
            if param in parameters:
                parameters.pop(param)
                self.tools_with_injected_params[func_name].append(param)

            if param in required_params:
                required_params.remove(param)

        return cleaned_spec

    def inject_parameters(
        self, function_name: str, args: dict[str, Any], custom_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Inject runtime params into function arguments.

        Args:
            function_name: Name of the function being called
            args: Original function arguments
            custom_inputs: Custom inputs containing values to inject

        Returns:
            Function arguments with injected parameters
        """
        if function_name not in self.tools_with_injected_params:
            return args

        enhanced_args = args.copy()
        for param in self.tools_with_injected_params[function_name]:
            if param in custom_inputs:
                enhanced_args[param] = custom_inputs[param]
            else:
                logger.warning(
                    f"Missing custom input '{param}' for function {function_name}"
                )

        return enhanced_args
