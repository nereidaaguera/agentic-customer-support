import json
from typing import Any, Generator, List
from uuid import uuid4

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from databricks.sdk import WorkspaceClient
from tools import get_mcp_tool_infos

LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
SYSTEM_PROMPT = "You are a helpful assistant."
MCP_SERVER_URLS = [
    "https://telco-outage-server-dev-3888667486068890.aws.databricksapps.com/mcp/",
]

def chat_message_to_responses_item(msg: dict) -> dict:
    """
    Convert a ChatCompletions-style message to a ResponsesAgent output item.
    Handles assistant messages, tool calls, and tool outputs.
    """
    # Tool call (function_call in OpenAI) => function_call item
    if msg.get("tool_calls"):
        tool_calls = []
        for tool_call in msg["tool_calls"]:
            tool_calls.append({
                "type": "function_call",
                "id": tool_call["id"],
                "call_id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"],
                "status": "completed",
            })
        # If you want to surface each call as a separate item
        return tool_calls[0] if len(tool_calls) == 1 else tool_calls
    # Normal assistant text
    if msg.get("role") == "assistant":
        return {
            "type": "message",
            "id": str(uuid4()),
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": msg.get("content", ""),
                "annotations": [],
            }],
            "status": "completed"
        }
    # Tool output (role=tool)
    if msg.get("role") == "tool":
        return {
            "type": "function_call_output",
            "call_id": msg.get("tool_call_id"),
            "output": msg.get("content"),
        }
    # User messages should never be returned by the LLM
    return {}

class ToolCallingAgent(ResponsesAgent):
    def get_tool_specs(self, workspace_client: WorkspaceClient) -> List[dict]:
        tool_infos = get_mcp_tool_infos(
            workspace_client=workspace_client, server_urls=MCP_SERVER_URLS
        )
        return [tool_info.spec for tool_info in tool_infos]

    def get_tools_dict(self, workspace_client: WorkspaceClient):
        tool_infos = get_mcp_tool_infos(
            workspace_client=workspace_client, server_urls=MCP_SERVER_URLS
        )
        return {tool.name: tool for tool in tool_infos}

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(
            self, tool_name: str, args: dict, workspace_client: WorkspaceClient
    ) -> Any:
        return self.get_tools_dict(workspace_client)[tool_name].exec_fn(**args)

    def chat_completion(self, messages: List[dict], workspace_client: WorkspaceClient) -> dict:
        # This uses Databricks' OpenAI-compatible endpoint
        client = workspace_client.serving_endpoints.get_open_ai_client()
        return client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=messages,
            tools=self.get_tool_specs(workspace_client),
        )

    def handle_tool_call(
            self, tool_call: dict, workspace_client: WorkspaceClient
    ) -> ResponsesAgentStreamEvent:
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(
            tool_name=tool_call["name"],
            args=args,
            workspace_client=workspace_client,
        ))
        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "type": "function_call_output",
                "call_id": tool_call["call_id"],
                "output": result,
            },
        )

    def call_and_run_tools(
            self,
            chat_history: List[dict],
            workspace_client: WorkspaceClient,
            max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Loop: chat_completion, parse response, yield function_call as needed, run tool, inject tool outputs, repeat.
        """
        for _ in range(max_iter):
            # Chat completion step
            response = self.chat_completion(chat_history, workspace_client)
            llm_msg = response.choices[0].message.to_dict()
            responses_item = chat_message_to_responses_item(llm_msg)
            # Handle tool calls (may be multiple per message)
            if isinstance(responses_item, list):  # multiple tool calls
                for tc in responses_item:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=tc,
                    )
                    tool_out_event = self.handle_tool_call(tc, workspace_client)
                    yield tool_out_event
                    # Inject tool output into chat_history for next LLM step
                    chat_history.append({
                        "role": "tool",
                        "content": tool_out_event.item["output"],
                        "tool_call_id": tc["call_id"],
                    })
                continue
            # Assistant text response: yield and stop
            if responses_item.get("type") == "message":
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=responses_item,
                )
                return
            # Single tool call
            if responses_item.get("type") == "function_call":
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=responses_item,
                )
                tool_out_event = self.handle_tool_call(responses_item, workspace_client)
                yield tool_out_event
                # Inject tool output into chat_history for next LLM step
                chat_history.append({
                    "role": "tool",
                    "content": tool_out_event.item["output"],
                    "tool_call_id": responses_item["call_id"],
                })
                continue
            # Tool output (shouldn't happen directly)
            if responses_item.get("type") == "function_call_output":
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=responses_item,
                )
                continue
        # If max_iter exceeded
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "id": str(uuid4()),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": f"Sorry, I couldn't answer after {max_iter} iterations.",
                    "annotations": [],
                }],
                "status": "completed"
            },
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        events = list(self.predict_stream(request))
        outputs = [
            event.item
            for event in events
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
            self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        workspace_client = WorkspaceClient()
        # Start with the system prompt and the incoming user message(s)
        chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.input:
            chat_history.append(msg.model_dump(exclude_none=True))
        yield from self.call_and_run_tools(
            chat_history=chat_history,
            workspace_client=workspace_client,
        )


mlflow.openai.autolog()
AGENT = ToolCallingAgent()
mlflow.models.set_model(AGENT)
