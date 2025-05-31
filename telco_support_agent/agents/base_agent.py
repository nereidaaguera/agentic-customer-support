"""Base agent class for telco support agents."""

import abc
import json
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

import backoff
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.entities.trace_info import TraceInfo
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from telco_support_agent.agents import AgentConfig
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Monkey patch TraceInfo.__init__ so we have better previews in the UI.
TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH = 10000


class MissingCustomInputError(ValueError):
    """Raise when custom inputs are missing from the request."""

    pass


def compute_request_preview(request: str) -> str:
    """Compute preview of request for tracing.

    Extracts most recent user message content for display in trace previews.

    Args:
        request: raw request string to process

    Returns:
        preview string truncated to max length
    """
    preview = ""

    if isinstance(request, str):
        try:
            data = json.loads(request)
        except (json.JSONDecodeError, TypeError):
            preview = request
            return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]
    else:
        data = request

    if isinstance(data, dict):
        try:
            input_list = data.get("request", {}).get("input", [])
            if isinstance(input_list, list):
                for item in reversed(input_list):
                    if (
                        isinstance(item, dict)
                        and item.get("role") == "user"
                        and isinstance(item.get("content"), str)
                    ):
                        preview = item["content"]
                        break
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Error extracting user content from request: {e}")

    if not preview:
        preview = str(request)

    return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]


def compute_response_preview(response: str) -> str:
    """Compute preview of response for tracing.

    Extracts assistant response for display in trace previews.

    Args:
        response: The raw response string to process

    Returns:
        A preview string truncated to max length
    """
    preview = ""

    if isinstance(response, str):
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            preview = response
            return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]
    else:
        data = response

    if isinstance(data, dict):
        try:
            output = data.get("output")
            if isinstance(output, list):
                for item in reversed(output):
                    if (
                        isinstance(item, dict)
                        and item.get("role") == "assistant"
                        and isinstance(item.get("content"), list)
                    ):
                        for part in reversed(item["content"]):
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "output_text"
                                and isinstance(part.get("text"), str)
                            ):
                                preview = part["text"]
                                break
                        if preview:
                            break
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Error extracting assistant content from response: {e}")

    if not preview:
        preview = str(response)

    return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]


is_patched = False

if not is_patched:
    # Monkey-patch the __init__
    original_init = TraceInfo.__init__

    def patched_init(self, request_preview=None, response_preview=None, **kwargs):  # NoQA: D417
        """Patched TraceInfo.__init__ that computes better previews.

        Args:
            request_preview: Raw request preview data
            response_preview: Raw response preview data
            **kwargs: Additional arguments passed to original __init__
        """
        if request_preview is not None:
            request_preview = compute_request_preview(request_preview)
        if response_preview is not None:
            response_preview = compute_response_preview(response_preview)
        original_init(
            self,
            request_preview=request_preview,
            response_preview=response_preview,
            **kwargs,
        )

    TraceInfo.__init__ = patched_init
    is_patched = True


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
        self.tools = self._filter_disable_tools(raw_tools)

        # Filter vector search tools as well
        self.vector_search_tools = vector_search_tools or {}
        if self.disable_tools and self.vector_search_tools:
            filtered_vector_tools = {}
            for tool_name, tool_obj in self.vector_search_tools.items():
                simple_name = (
                    tool_name.split(".")[-1] if "." in tool_name else tool_name
                )
                is_disabled = (
                    tool_name in self.disable_tools or simple_name in self.disable_tools
                )
                if not is_disabled:
                    filtered_vector_tools[tool_name] = tool_obj
                else:
                    logger.info(f"Disabling vector search tool: {tool_name}")
            self.vector_search_tools = filtered_vector_tools

        # init parameter injector
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
            config = AgentConfig(**config_dict)
            cls._config_cache[agent_type] = config

            return config

        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Invalid configuration for {agent_type}: {e}") from e

    def _load_tools_from_config(self) -> list[dict]:
        """Load UC function tools based on the agent's domain.

        Returns:
            List of UC function tool specifications
        """
        try:
            # get UC functions for the agent / sub-agent's domain
            function_names = (
                self.config.uc_functions if hasattr(self.config, "uc_functions") else []
            )

            if not function_names:
                logger.warning(
                    f"No UC functions configured for agent type: {self.agent_type}"
                )
                return []

            # toolkit with specified functions
            toolkit = UCFunctionToolkit(function_names=function_names)
            return toolkit.tools

        except Exception as e:
            logger.error(f"Error loading UC function tools: {str(e)}")
            return []

    def _filter_disable_tools(self, tools: list[dict]) -> list[dict]:
        """Filter disabled tools from the tools list.

        Args:
            tools: List of tool specifications

        Returns:
            Filtered list of tools with disabled tools removed
        """
        if not self.disable_tools:
            return tools

        filtered_tools = []
        disabled_count = 0

        for tool in tools:
            tool_name = None

            if "function" in tool and "name" in tool["function"]:
                # UC function format
                tool_name = tool["function"]["name"]
            elif "name" in tool:
                # direct name format
                tool_name = tool["name"]
            elif hasattr(tool, "tool_name"):
                # VectorSearchRetrieverTool format
                tool_name = tool.tool_name

            if tool_name:
                # UC functions use underscores: telco_customer_support_dev__agent__get_usage_info
                # Other tools might use dots: some.namespace.tool_name
                if "." in tool_name:
                    simple_name = tool_name.split(".")[-1]
                elif "__" in tool_name:
                    simple_name = tool_name.split("__")[-1]
                else:
                    simple_name = tool_name

                is_disabled = (
                    tool_name in self.disable_tools or simple_name in self.disable_tools
                )

                if is_disabled:
                    logger.info(
                        f"Disabling tool: {tool_name} (simple name: {simple_name})"
                    )
                    disabled_count += 1
                    continue

            filtered_tools.append(tool)

        if disabled_count > 0:
            logger.info(
                f"Filtered out {disabled_count} disabled tools for {self.agent_type} agent"
            )

        return filtered_tools

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
                f"Missing required custom inputs: {missing_inputs}. "
                f"This agent requires: {self.parameter_injector.inject_params}"
            )

    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute tool.

        Handles both Unity Catalog functions and VectorSearchRetrieverTool objects.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
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
                # check if vector search tool
                if tool_name in self.vector_search_tools:
                    vector_tool = self.vector_search_tools[tool_name]
                    result = vector_tool.execute(**args)
                else:
                    # otherwise treat as UC function
                    # replace any underscores to dots in function name
                    uc_function_name = tool_name.replace("__", ".")

                    # execute tool using UC function client
                    result = self.uc_client.execute_function(
                        function_name=uc_function_name, parameters=args
                    )
                    result = result.value

                span.set_outputs(result)
                return result

            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                span.set_attributes({"error": True, "error_message": error_msg})
                span.set_outputs({"error": error_msg})
                logger.error(error_msg)
                return error_msg

    def convert_to_chat_completion_format(
        self, message: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert from Responses API to ChatCompletion compatible."""
        msg_type = message.get("type", None)
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": None
                    if self.llm_endpoint == "databricks-claude-3-7-sonnet"
                    else "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        return [{k: v for k, v in message.items() if k in compatible_keys}]

    def prepare_messages_for_llm(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter out message fields that are not compatible with LLM message formats and convert from Responses API to ChatCompletion compatible."""
        chat_msgs = []
        for msg in messages:
            chat_msgs.extend(self.convert_to_chat_completion_format(msg))
        return chat_msgs

    @backoff.on_exception(backoff.expo, Exception)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call LLM with provided message history."""
        try:
            params = {
                "model": self.llm_endpoint,
                "messages": self.prepare_messages_for_llm(messages),
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
        """Execute tool calls and return updated messages and response events.

        Args:
            messages: Current message history
            tool_calls: Tool calls to execute
            custom_inputs: Optional custom inputs

        Returns:
            Tuple of (updated_messages, response_events)
        """
        updated_messages = messages.copy()
        events = []

        for tool_call in tool_calls:
            function = tool_call["function"]
            args = json.loads(function["arguments"])

            try:
                # inject params before calling tool
                enhanced_args = self.parameter_injector.inject_parameters(
                    function["name"], args, custom_inputs or {}
                )

                result = self.execute_tool(
                    tool_name=function["name"], args=enhanced_args
                )
                result_str = str(result)
            except Exception as e:
                logger.error(f"Error executing tool {function['name']}: {e}")
                result_str = f"Error executing tool: {str(e)}"

            # add tool result to message history
            updated_messages.append(
                {"role": "tool", "content": result_str, "tool_call_id": tool_call["id"]}
            )

            # create response event
            responses_tool_call_output = {
                "type": "function_call_output",
                "call_id": tool_call["id"],
                "output": result_str,
            }

            events.append(
                ResponsesAgentStreamEvent(
                    type="response.output_item.done", item=responses_tool_call_output
                )
            )

        return updated_messages, events

    def call_and_run_tools(
        self,
        request: ResponsesAgentRequest,
        max_iter: int = 10,
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

            # handle tool calls if present
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
                                {
                                    "type": "output_text",
                                    "text": llm_output["content"],
                                }
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
