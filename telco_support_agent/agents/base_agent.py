"""Base agent class for telco support agents."""

import abc
import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import backoff
import mlflow
import yaml
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
)
from pydantic import ValidationError

from telco_support_agent.agents import AgentConfig
from telco_support_agent.tools import ToolInfo

logger = logging.getLogger(__name__)


class BaseAgent(ResponsesAgent, abc.ABC):
    """Base agent class all agents inherit from."""

    _config_cache: dict[str, AgentConfig] = {}

    def __init__(
        self,
        agent_type: str,
        llm_endpoint: Optional[str] = None,
        tools: Optional[list[ToolInfo]] = None,
        system_prompt: Optional[str] = None,
        config_dir: Optional[Path | str] = None,
    ):
        """Initialize base agent from config file.

        Args:
            agent_type: Type of agent (used for config loading)
            llm_endpoint: Optional override for LLM endpoint
            tools: Optional list of tools (overrides config)
            system_prompt: Optional override for system prompt
            config_dir: Optional directory for config files
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

        # system prompt
        self.system_prompt = system_prompt or self.config.system_prompt

        # set up tools
        self.tools = tools or self._create_tools_from_config()
        self._tools_dict = {tool.name: tool for tool in self.tools}

        logger.info(f"Initialized {agent_type} agent with {len(self.tools)} tools")

    @classmethod
    def _load_config(
        cls, agent_type: str, config_dir: Optional[Path | str] = None
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

        # get config file
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs" / "agents"
        else:
            config_dir = Path(config_dir)
        config_path = config_dir / f"{agent_type}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"No config file found for agent type: {agent_type}"
            )

        # load and validate
        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            config = AgentConfig(**config_dict)
            cls._config_cache[agent_type] = config

            return config

        except (yaml.YAMLError, ValidationError) as e:
            raise ValueError(f"Invalid configuration for {agent_type}: {e}") from e

    def _create_tools_from_config(self) -> list[ToolInfo]:
        """Create tool objects from configuration.

        Returns:
            List of configured tools
        """
        tools = []

        for tool_config in self.config.tools:
            # get tool implementation method
            method_name = f"tool_{tool_config.name}"
            if not hasattr(self, method_name):
                logger.warning(f"No implementation found for tool {tool_config.name}")
                continue

            tool_method = getattr(self, method_name)

            # create tool spec
            parameters_dict = {"type": "object", "properties": {}, "required": []}

            for param_name, param_config in tool_config.parameters.items():
                param_dict = {
                    "type": param_config.type,
                    "description": param_config.description,
                }

                if param_config.enum:
                    param_dict["enum"] = param_config.enum

                parameters_dict["properties"][param_name] = param_dict
                parameters_dict["required"].append(param_name)

            tool_spec = {
                "type": "function",
                "function": {
                    "name": tool_config.name,
                    "description": tool_config.description,
                    "parameters": parameters_dict,
                },
            }
            tool = ToolInfo(name=tool_config.name, spec=tool_spec, exec_fn=tool_method)

            tools.append(tool)

        return tools

    def get_tool_specs(self) -> list[dict]:
        """Return tool specifications in the format LLM expects."""
        return [tool.spec for tool in self.tools]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute tool with given args."""
        if tool_name not in self._tools_dict:
            return f"Error: Tool '{tool_name}' not found."

        return self._tools_dict[tool_name].exec_fn(**args)

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

    @backoff.on_exception(backoff.expo, Exception)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call LLM with provided message history.

        Args:
            messages: List of messages to send to the LLM

        Returns:
            LLM response message
        """
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

            logger.debug(f"LLM response: {response}")

            return response

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

    def handle_tool_call(
        self, messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[ResponsesStreamEvent]]:
        """Execute tool calls and return updated messages and response events.

        Args:
            messages: Current message history
            tool_calls: Tool calls to execute

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
                ResponsesStreamEvent(
                    type="response.output_item.done", item=responses_tool_call_output
                )
            )

        return updated_messages, events

    def call_and_run_tools(
        self, messages: list[dict[str, Any]], max_iter: int = 10
    ) -> Generator[ResponsesStreamEvent, None, None]:
        """Run the call-tool-response loop up to max_iter times.

        Args:
            messages: Initial message history
            max_iter: Maximum number of iterations

        Yields:
            ResponsesStreamEvent objects
        """
        current_messages = messages.copy()

        for _ in range(max_iter):
            last_msg = current_messages[-1]

            # handle tool calls if present
            if tool_calls := last_msg.get("tool_calls", None):
                updated_messages, events = self.handle_tool_call(
                    current_messages, tool_calls
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
                        yield ResponsesStreamEvent(
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
                    yield ResponsesStreamEvent(
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

        yield ResponsesStreamEvent(
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
    def predict(self, model_input: ResponsesRequest) -> ResponsesResponse:
        """Make prediction based on input request."""
        outputs = [
            event.item
            for event in self.predict_stream(model_input)
            if event.type == "response.output_item.done"
        ]
        return ResponsesResponse(
            output=outputs, custom_outputs=model_input.custom_inputs
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, model_input: ResponsesRequest
    ) -> Generator[ResponsesStreamEvent, None, None]:
        """Stream predictions."""
        messages = [{"role": "system", "content": self.system_prompt}] + [
            i.model_dump() for i in model_input.input
        ]

        yield from self.call_and_run_tools(messages)

    @abc.abstractmethod
    def get_description(self) -> str:
        """Return a description of this agent."""
        pass
