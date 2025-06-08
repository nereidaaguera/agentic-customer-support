"""Base agent class for telco support agents."""

import abc
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

import backoff
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from telco_support_agent.agents import AgentConfig
from telco_support_agent.agents.utils.exceptions import (
    AgentConfigurationError,
    MissingCustomInputError,
    ToolExecutionError,
)
from telco_support_agent.agents.utils.message_formatting import (
    prepare_messages_for_llm,
)
from telco_support_agent.agents.utils.tool_injection import ToolParameterInjector
from telco_support_agent.agents.utils.trace_utils import (
    create_request_structure,
    create_response_structure,
    patch_trace_info,
    update_trace_preview,
)
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

patch_trace_info()


class BaseAgent(ResponsesAgent, abc.ABC):
    """Base agent class all agents inherit from."""

    _config_cache: dict[str, AgentConfig] = {}

    def __init__(
        self,
        agent_type: str,
        llm_endpoint: Optional[str] = None,
        tools: Optional[list[dict]] = None,  # UC function tools
        vector_search_tools: Optional[
            dict[str, Any]
        ] = None,  # Map of tool name -> VectorSearchRetrieverTool (not UC function)
        system_prompt: Optional[str] = None,
        config_dir: Optional[Path | str] = None,
        inject_tool_args: Optional[list[str]] = None,
        disable_tools: Optional[list[str]] = None,
    ):
        """Initialize base agent from config file.

        Args:
            agent_type: Type of agent (used for config loading)
            llm_endpoint: Optional override for LLM endpoint
            tools: Optional list of UC function tools
            vector_search_tools: Optional dict mapping tool names to VectorSearchRetrieverTool objects
            system_prompt: Optional override for system prompt
            config_dir: Optional directory for config files
            inject_tool_args: Optional list of additional tool arguments to be injected into tool calls from custom_inputs.
            disable_tools: Optional list of tool names to disable. Can be simple names or full UC function names.
        """
        # load config file
        self.agent_type = agent_type
        self.config = self._load_config(agent_type, config_dir)
        if disable_tools is None:
            disable_tools = self._load_disable_tools_from_artifact()
        self.disable_tools = disable_tools or []

        self.llm_endpoint = llm_endpoint or self.config.llm.endpoint
        self.llm_params = self.config.llm.params
        self.workspace_client = WorkspaceClient()
        self.model_serving_client = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )

        # init UC function client for tool execution
        self.uc_client = DatabricksFunctionClient()

        # system prompt
        self.system_prompt = system_prompt or self.config.system_prompt

        # set up tools
        raw_tools = tools or self._load_tools_from_config()
        self.tools = self._filter_disabled_tools(raw_tools)

        # vector search
        self.vector_search_tools = self._filter_disabled_vector_tools(
            vector_search_tools or {}
        )

        # tool parameter injector
        self.parameter_injector = ToolParameterInjector(inject_tool_args or [])
        self.llm_tool_specs = self._prepare_llm_tool_specs()

        logger.info(f"Initialized {agent_type} agent with {len(self.tools)} tools")
        if self.disable_tools:
            logger.info(f"Disabled tools: {self.disable_tools}")

    @classmethod
    def _load_config(
        cls, agent_type: str, config_dir: Optional[Union[str, Path]] = None
    ) -> AgentConfig:
        """Load agent configuration from YAML file.

        Args:
            agent_type: Type of agent to load config for
            config_dir: Optional directory for config files

        Returns:
            Validated agent configuration
        """
        # use cached config if available
        if agent_type in cls._config_cache:
            return cls._config_cache[agent_type]

        try:
            from telco_support_agent.utils.config import config_manager

            config_dict = config_manager.get_config(agent_type)
            uc_config = config_manager.get_uc_config()
            config = AgentConfig(**config_dict, uc_config=uc_config)
            cls._config_cache[agent_type] = config
            return config

        except (FileNotFoundError, ValueError) as e:
            raise AgentConfigurationError(agent_type, str(e)) from e

    def _load_tools_from_config(self) -> list[dict]:
        """Load UC function tools based on the agent's configuration."""
        try:
            catalog = self.config.uc_config.agent["catalog"]
            schema = self.config.uc_config.agent["schema"]

            function_names = [
                f"{catalog}.{schema}.{function_name}"
                for function_name in getattr(self.config, "uc_functions", [])
            ]

            if not function_names:
                logger.warning(
                    f"No UC functions configured for agent type: {self.agent_type}"
                )
                return []

            toolkit = UCFunctionToolkit(function_names=function_names)
            return toolkit.tools

        except Exception as e:
            logger.error(f"Error loading UC function tools: {str(e)}")
            return []

    def _load_disable_tools_from_artifact(self) -> list[str]:
        """Load disable_tools from artifact if available."""
        logger.info("Attempting to load disable_tools from artifact...")

        try:
            import json

            from mlflow.artifacts import download_artifacts

            logger.info("Downloading disable_tools.json artifact...")
            artifact_path = download_artifacts(artifact_path="disable_tools.json")
            logger.info(f"Download result: {artifact_path}")

            if artifact_path:
                artifact_path_obj = Path(artifact_path)
                logger.info(f"Checking if artifact path exists: {artifact_path_obj}")
                logger.info(f"Path exists: {artifact_path_obj.exists()}")

                if artifact_path_obj.exists():
                    logger.info(f"Reading disable_tools from: {artifact_path}")
                    with open(artifact_path) as f:
                        data = json.load(f)
                        logger.info(f"Loaded JSON data: {data}")
                        disable_tools = data.get("disable_tools", [])
                        logger.info(f"Extracted disable_tools: {disable_tools}")
                        return disable_tools
                else:
                    logger.warning(f"Artifact path does not exist: {artifact_path}")
            else:
                logger.warning("download_artifacts returned None/empty path")

        except Exception as e:
            logger.debug(f"Could not load disable_tools artifact: {e}")

        logger.info("Falling back to empty disable_tools list")
        return []

    def _filter_disabled_tools(self, tools: list[dict]) -> list[dict]:
        """Filter disabled tools from the tools list."""
        if not self.disable_tools:
            return tools

        filtered_tools = []
        disabled_count = 0

        for tool in tools:
            tool_name = self._extract_tool_name(tool)
            if tool_name and self._is_tool_disabled(tool_name):
                disabled_count += 1
                logger.info(f"Disabling tool: {tool_name}")
                continue
            filtered_tools.append(tool)

        if disabled_count > 0:
            logger.info(
                f"Filtered out {disabled_count} disabled tools for {self.agent_type} agent"
            )

        return filtered_tools

    def _filter_disabled_vector_tools(
        self, vector_tools: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter disabled vector search tools."""
        if not self.disable_tools:
            return vector_tools

        filtered_tools = {}
        for tool_name, tool_obj in vector_tools.items():
            if not self._is_tool_disabled(tool_name):
                filtered_tools[tool_name] = tool_obj
            else:
                logger.info(f"Disabling vector search tool: {tool_name}")

        return filtered_tools

    def _extract_tool_name(self, tool: dict) -> Optional[str]:
        """Extract tool name from tool specification."""
        if "function" in tool and "name" in tool["function"]:
            return tool["function"]["name"]
        elif "name" in tool:
            return tool["name"]
        elif hasattr(tool, "tool_name"):
            return tool.tool_name
        return None

    def _is_tool_disabled(self, tool_name: str) -> bool:
        """Check if a tool is disabled."""
        if "." in tool_name:
            simple_name = tool_name.split(".")[-1]
        elif "__" in tool_name:
            simple_name = tool_name.split("__")[-1]
        else:
            simple_name = tool_name

        return tool_name in self.disable_tools or simple_name in self.disable_tools

    def _prepare_llm_tool_specs(self) -> list[dict[str, Any]]:
        """Prepare tool specifications for LLM by removing injected parameters.

        Returns:
            List of tool specifications formatted for LLM consumption
        """
        return [
            self.parameter_injector.prepare_tool_spec_for_llm(tool)
            for tool in self.tools
        ]

    def get_tool_specs(self) -> list[dict]:
        """Return tool specifications in the format LLM expects."""
        return self.llm_tool_specs

    def validate_request(self, request: ResponsesAgentRequest) -> None:
        """Validate that request contains required custom inputs.

        Args:
            request: The incoming request to validate

        Raises:
            MissingCustomInputError: If required custom inputs are missing
        """
        if not self.parameter_injector.inject_params:
            return

        missing_inputs = []
        custom_inputs = request.custom_inputs or {}

        for param in self.parameter_injector.inject_params:
            if param not in custom_inputs:
                missing_inputs.append(param)

        if missing_inputs:
            raise MissingCustomInputError(
                missing_inputs, self.parameter_injector.inject_params
            )

    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute tool (UC function or vector search)."""
        trace_tool_name = tool_name.replace("__", ".").split(".")[-1]

        with mlflow.start_span(
            name=f"tool_{trace_tool_name}", span_type=SpanType.TOOL
        ) as span:
            span.set_inputs({"tool_name": tool_name, "args": args})
            span.set_attributes(
                {
                    "tool_type": "uc_function"
                    if "__" in tool_name
                    else "vector_search",
                    "agent_type": self.agent_type,
                }
            )

            try:
                if tool_name in self.vector_search_tools:
                    result = self.vector_search_tools[tool_name].execute(**args)
                else:
                    uc_function_name = tool_name.replace("__", ".")
                    result = self.uc_client.execute_function(
                        function_name=uc_function_name, parameters=args
                    ).value

                span.set_outputs(result)
                return result

            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                span.set_attributes({"error": True, "error_message": error_msg})
                span.set_outputs({"error": error_msg})
                logger.error(error_msg)
                raise ToolExecutionError(tool_name, error_msg, e) from e

    @backoff.on_exception(backoff.expo, Exception)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call LLM with provided message history."""
        try:
            params = {
                "model": self.llm_endpoint,
                "messages": prepare_messages_for_llm(messages, self.llm_endpoint),
                "tools": self.get_tool_specs(),
                **self.llm_params,
            }

            response = (
                self.model_serving_client.chat.completions.create(**params)
                .choices[0]
                .message.to_dict()
            )

            return response

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

    def handle_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, Any]], list[ResponsesAgentStreamEvent]]:
        """Execute tool calls and return updated messages and response events."""
        updated_messages = messages.copy()
        events = []

        for tool_call in tool_calls:
            function = tool_call["function"]
            args = json.loads(function["arguments"])

            try:
                enhanced_args = self.parameter_injector.inject_parameters(
                    function["name"], args, custom_inputs or {}
                )

                result = self.execute_tool(
                    tool_name=function["name"], args=enhanced_args
                )
                result_str = str(result)
            except ToolExecutionError as e:
                logger.error(f"Tool execution failed: {e}")
                result_str = f"Error executing tool: {str(e)}"

            updated_messages.append(
                {"role": "tool", "content": result_str, "tool_call_id": tool_call["id"]}
            )

            events.append(
                ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item={
                        "type": "function_call_output",
                        "call_id": tool_call["id"],
                        "output": result_str,
                    },
                )
            )

        return updated_messages, events

    def call_and_run_tools(
        self, request: ResponsesAgentRequest, max_iter: int = 10
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Run the call-tool-response loop up to max_iter times.

        Args:
            request: ResponsesAgentRequest model input.
            max_iter: Maximum number of iterations
        Yields:
            Responses Agent Stream Event objects
        """
        self.validate_request(request)

        messages = [{"role": "system", "content": self.system_prompt}] + [
            i.model_dump() for i in request.input
        ]

        current_messages = messages.copy()

        for _ in range(max_iter):
            last_msg = current_messages[-1]

            if tool_calls := last_msg.get("tool_calls", None):
                updated_messages, events = self.handle_tool_call(
                    current_messages, tool_calls, request.custom_inputs
                )
                current_messages = updated_messages
                yield from events
            elif last_msg.get("role", None) == "assistant":
                return
            else:
                llm_output = self.call_llm(current_messages)
                current_messages.append(llm_output)

                if tool_calls := llm_output.get("tool_calls", None):
                    for tool_call in tool_calls:
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item={
                                "type": "function_call",
                                "call_id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                                "id": str(uuid4()),
                            },
                        )
                else:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item={
                            "role": llm_output["role"],
                            "type": "message",
                            "id": str(uuid4()),
                            "content": [
                                {"type": "output_text", "text": llm_output["content"]}
                            ],
                        },
                    )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "id": str(uuid4()),
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Max iterations ({max_iter}) reached. Stopping.",
                    }
                ],
                "role": "assistant",
                "type": "message",
            },
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Make prediction based on input request."""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream predictions."""
        yield from self.call_and_run_tools(request)

    # convenience methods for trace utilities
    def update_trace_preview(self, **kwargs):
        """Convenience method for updating trace previews."""
        return update_trace_preview(**kwargs)

    def create_request_structure(self, **kwargs):
        """Convenience method for creating request structures."""
        return create_request_structure(**kwargs)

    def create_response_structure(self, **kwargs):
        """Convenience method for creating response structures."""
        return create_response_structure(**kwargs)
