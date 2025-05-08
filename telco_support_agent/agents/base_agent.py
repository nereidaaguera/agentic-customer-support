"""Base agent class for telco support agents."""

import abc
import json
from collections.abc import Generator
from typing import Any
from uuid import uuid4

import backoff
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
)
from openai import OpenAI

from telco_support_agent.tools import ToolInfo


class BaseAgent(ResponsesAgent, abc.ABC):
    """Base agent class all agents will inherit from."""

    def __init__(
        self,
        llm_endpoint: str,
        tools: list[ToolInfo],
        system_prompt: str,
    ):
        """Initialize base agent.

        Args:
            llm_endpoint: Name of LLM endpoint to use
            tools: List of tools available to this agent
            system_prompt: System prompt for specific agent
        """
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        # internal state will be standard chat completion messages
        self.messages: list[dict[str, Any]] = []

    def get_tool_specs(self) -> list[dict]:
        """Return tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute tool with given args."""
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
    def call_llm(self) -> dict[str, Any]:
        """Call LLM with current message history."""
        return (
            self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=self.prepare_messages_for_llm(self.messages),
                tools=self.get_tool_specs(),
            )
            .choices[0]
            .message.to_dict()
        )

    def handle_tool_call(
        self, tool_calls: list[dict[str, Any]]
    ) -> Generator[ResponsesStreamEvent, None, None]:
        """Execute tool calls and return a ResponsesStreamEvent with tool output."""
        for tool_call in tool_calls:
            function = tool_call["function"]
            args = json.loads(function["arguments"])
            # Cast tool result to a string, since not all tools return as string
            result = str(self.execute_tool(tool_name=function["name"], args=args))
            self.messages.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call["id"]}
            )
            responses_tool_call_output = {
                "type": "function_call_output",
                "call_id": tool_call["id"],
                "output": result,
            }
            # Following the example exactly - using item parameter
            yield ResponsesStreamEvent(
                type="response.output_item.done", item=responses_tool_call_output
            )

    def call_and_run_tools(
        self,
        max_iter: int = 10,
    ) -> Generator[ResponsesStreamEvent, None, None]:
        """Run the call-tool-response loop up to max_iter times."""
        for _ in range(max_iter):
            last_msg = self.messages[-1]
            if tool_calls := last_msg.get("tool_calls", None):
                yield from self.handle_tool_call(tool_calls)
            elif last_msg.get("role", None) == "assistant":
                return
            else:
                llm_output = self.call_llm()
                self.messages.append(llm_output)

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
                        "text": "Max iterations reached. Stopping.",
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
        self.messages = [{"role": "system", "content": self.system_prompt}] + [
            i.model_dump() for i in model_input.input
        ]
        yield from self.call_and_run_tools()
