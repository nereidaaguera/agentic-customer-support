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
        inject_tool_args: Optional[dict[str, str]] = None,
    ):
        """Initialize base agent from config file.

        Args:
            agent_type: Type of agent (used for config loading)
            llm_endpoint: Optional override for LLM endpoint
            tools: Optional list of UC function tools
            vector_search_tools: Optional dict mapping tool names to VectorSearchRetrieverTool objects
            system_prompt: Optional override for system prompt
            config_dir: Optional directory for config files
            inject_tool_args: Optional dict of additional tool arguments to be injected into tool calls.
        """
        # load config file
        self.agent_type = agent_type
        self.config = self._load_config(agent_type, config_dir)

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
        self.tools = tools or self._load_tools_from_config()
        self.vector_search_tools = vector_search_tools or {}
        self.inject_tool_args = inject_tool_args or {}
        self.tools_with_injected = defaultdict(list)
        self.cleaned_tools = self.generated_cleaned_tools()

        logger.info(f"Initialized {agent_type} agent with {len(self.tools)} tools")

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
            from telco_support_agent.agents.config import config_manager

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

    def get_tool_specs(self) -> list[dict]:
        """Return tool specifications in the format LLM expects."""
        return self.tools

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute tool.

        Handles both Unity Catalog functions and VectorSearchRetrieverTool objects.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        try:
            # check if vector search tool
            if tool_name in self.vector_search_tools:
                vector_tool = self.vector_search_tools[tool_name]
                return vector_tool.execute(**args)

            # otherwise treat as UC function
            # replace any underscores to dots in function name
            uc_function_name = tool_name.replace("__", ".")

            # execute tool using UC function client
            result = self.uc_client.execute_function(
                function_name=uc_function_name, parameters=args
            )
            return result.value

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
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
        """Filter out message fields that are not compatible with LLM message formats."""
        chat_msgs = []
        for msg in messages:
            chat_msgs.extend(self.convert_to_chat_completion_format(msg))
        return chat_msgs

    def remove_injected_params(self, cleaned_tool):
        """Remove parameters that will be injected at runtime from the cleaned tool before calling the LLM."""
        parameters = cleaned_tool["function"]["parameters"]["properties"]
        required_parameters = cleaned_tool["function"]["parameters"]["required"]
        func_name = cleaned_tool["function"]["name"]
        for param in self.inject_tool_args.keys():
            if param in parameters:
                logger.info(f"Removing param '{param}' from tool: {func_name}")
                parameters.pop(param)
                self.tools_with_injected[func_name].append(param)
            if param in required_parameters:
                required_parameters.remove(param)
        return cleaned_tool

    def generated_cleaned_tools(self):
        """Create cleaned tools for calling LLM with tools in the expected format."""
        cleaned_tools = []
        for tool in self.get_tool_specs():
            if "function" in tool:
                cleaned_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }

                if (
                    "parameters" in tool["function"]
                    and "properties" in tool["function"]["parameters"]
                ):
                    cleaned_tool["function"]["parameters"]["properties"] = tool[
                        "function"
                    ]["parameters"]["properties"]

                if (
                    "parameters" in tool["function"]
                    and "required" in tool["function"]["parameters"]
                ):
                    cleaned_tool["function"]["parameters"]["required"] = tool[
                        "function"
                    ]["parameters"]["required"]

                cleaned_tools.append(self.remove_injected_params(cleaned_tool))

        return cleaned_tools

    @backoff.on_exception(backoff.expo, Exception)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call LLM with provided message history."""
        try:
            params = {
                "model": self.llm_endpoint,
                "messages": self.prepare_messages_for_llm(messages),
                "tools": self.cleaned_tools,
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

            # execute tool and convert result to string
            try:
                # Inject arguments to tool.
                if function["name"] in self.tools_with_injected:
                    for param in self.tools_with_injected[function["name"]]:
                        args[param] = custom_inputs[param]

                result = self.execute_tool(tool_name=function["name"], args=args)
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
        messages: list[dict[str, Any]],
        custom_inputs: Optional[dict[str, Any]] = None,
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Run the call-tool-response loop up to max_iter times.

        Args:
            messages: Initial message history
            custom_inputs: Optional custom inputs
            max_iter: Maximum number of iterations
        Yields:
            ResponsesAgentStreamEvent objects
        """
        current_messages = messages.copy()

        for _ in range(max_iter):
            last_msg = current_messages[-1]

            # handle tool calls if present
            if tool_calls := last_msg.get("tool_calls", None):
                updated_messages, events = self.handle_tool_call(
                    current_messages, tool_calls, custom_inputs
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

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, model_input: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Make prediction based on input request."""
        outputs = [
            event.item
            for event in self.predict_stream(model_input)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=model_input.custom_inputs
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, model_input: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream predictions."""
        messages = [{"role": "system", "content": self.system_prompt}] + [
            i.model_dump() for i in model_input.input
        ]

        yield from self.call_and_run_tools(messages, model_input.custom_inputs)
