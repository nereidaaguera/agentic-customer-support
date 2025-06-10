import json
from typing import Any, Generator, List, Optional
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

# -----------------------------------------------------------------------------
# Configurable constants
# -----------------------------------------------------------------------------
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
SYSTEM_PROMPT = ("You are a helpful customer support assistant for a telco company. "
                 "Focus on providing truthful answers grounded in available knowledge sources")
MCP_SERVER_URLS = [
    "https://db-ml-models-dev-us-west.cloud.databricks.com/api/2.0/mcp/vector-search/telco_customer_support_dev/mcp_agent",
    "https://db-ml-models-dev-us-west.cloud.databricks.com/api/2.0/mcp/functions/telco_customer_support_dev/mcp_agent",
    "https://mcp-telco-outage-server-3217006663075879.aws.databricksapps.com/mcp/"
]


class ToolCallingAgent(ResponsesAgent):
    """
    A lightweight ResponsesAgent that (1) calls a Claude-like LLM endpoint
    via Databricks’ model serving client, and (2) executes MCP‐based tools in‐between
    LLM calls. We keep it as simple as possible: no base‐class magic, no parameter‐injector.
    """

    def __init__(self, llm_endpoint: str):
        super().__init__()
        self.llm_endpoint = llm_endpoint

    def get_tool_specs(self, workspace_client: WorkspaceClient) -> List[dict]:
        """
        Fetch all MCP tool specs (OpenAI‐style) from each configured URL.
        Returns a list of {"type": "function", "function": {...}} dicts.
        """
        tool_infos = get_mcp_tool_infos(
            workspace_client=workspace_client, server_urls=MCP_SERVER_URLS
        )
        return [tool_info.spec for tool_info in tool_infos]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(
            self, tool_name: str, args: dict, tool_infos,
    ) -> Any:
        """
        Execute exactly one MCP tool by name (using its exec_fn).
        We re‐fetch tool_infos each time to keep things simple (no cache).
        """
        tools_dict = {t.name: t for t in tool_infos}
        return tools_dict[tool_name].exec_fn(**args)

    def convert_to_chat_completion_format(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert from Responses API to ChatCompletion compatible"""
        msg_type = message.get("type", None)
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"]
                            }

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

    def prepare_messages_for_llm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out message fields that are not compatible with LLM message formats and convert from Responses API to ChatCompletion compatible"""
        chat_msgs = []
        for msg in messages:
            chat_msgs.extend(self.convert_to_chat_completion_format(msg))
        return chat_msgs

    def chat_completion(
            self, messages: List[dict], workspace_client: WorkspaceClient, tool_infos,
    ) -> dict:
        """
        Invoke the Databricks‐hosted Chat Completions endpoint.
        We pass “model=self.llm_endpoint,” the filtered messages, plus any tool specs.
        """
        model_serving_client = workspace_client.serving_endpoints.get_open_ai_client()
        return model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=self.prepare_messages_for_llm(messages),
            tools=[tool_info.spec for tool_info in tool_infos],
        )

    def call_and_run_tools(
            self,
            request_messages: List[dict],
            workspace_client: WorkspaceClient,
            max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Core “call‐tool‐response” loop.
        Starting from request_messages (including SYSTEM_PROMPT + user messages),
        iteratively:
          1. Call the LLM → get a response dict → wrap it as a “function_call” event or normal assistant message.
          2. If the LLM issued a tool_call, run the tool, append a tool_output message, and yield that as an event.
          3. Stop when no more tool_calls or max_iter is reached.
        """
        # current history → always keep re‐using the evolving list
        current_history = request_messages.copy()
        tool_infos = get_mcp_tool_infos(workspace_client, server_urls=MCP_SERVER_URLS)
        for iteration in range(max_iter):
            with mlflow.start_span(name=f"agent_iteration_{iteration + 1}", span_type=SpanType.AGENT):
                # 1) Call LLM
                llm_resp = self.chat_completion(messages=current_history, workspace_client=workspace_client, tool_infos=tool_infos)
                # Extract the “assistant” message
                choice = llm_resp.choices[0].message.to_dict()
                assistant_dict = {
                    "role": choice["role"],
                    "content": choice.get("content"),
                    "tool_calls": choice.get("tool_calls"),
                    "name": choice.get("name"),
                    # tool_call_id only appears if this message is in response to a function output.
                }
                assistant_dict["id"] = str(uuid4())
                current_history.append(assistant_dict)

                # 2) If the LLM asked for a tool, emit a “function_call” event
                tool_calls = assistant_dict.get("tool_calls")
                if tool_calls:
                    # Yield a ResponsesAgentStreamEvent for each function_call
                    for fc in tool_calls:
                        function_call_payload = {
                            "type": "function_call",
                            "call_id": fc["id"],
                            "name": fc["function"]["name"],
                            "arguments": fc["function"]["arguments"],
                            "id": str(uuid4()),
                        }
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done", item=function_call_payload
                        )

                    # 3) Now execute each tool, append “tool” messages, yield their output events
                    for fc in tool_calls:
                        fname = fc["function"]["name"]
                        fargs = json.loads(fc["function"]["arguments"])
                        try:
                            result = self.execute_tool(
                                tool_name=fname, args=fargs, tool_infos=tool_infos
                            )
                        except Exception as e:
                            result = f"Error executing tool {fname}: {e}"

                        # wrap the result as a “tool” message
                        tool_output_msg = {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": fc["id"],
                            "id": str(uuid4()),
                        }
                        current_history.append(tool_output_msg)

                        # emit a function_call_output event
                        output_payload = {
                            "type": "function_call_output",
                            "call_id": fc["id"],
                            "output": str(result),
                        }
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done", item=output_payload
                        )

                    # continue to next iteration (the newly‐appended tool outputs are in history now)
                    continue

                # 4) If no tool_calls, it's a plain “assistant” text reply → emit a message event and exit
                text_payload = {
                    "role": "assistant",
                    "type": "message",
                    "id": str(uuid4()),
                    "content": [{"type": "output_text", "text": assistant_dict["content"]}],
                }
                yield ResponsesAgentStreamEvent(type="response.output_item.done", item=text_payload)
                return

        # 5) If max_iter reached without a normal assistant reply, send a final fallback
        fallback_payload = {
            "id": str(uuid4()),
            "content": [
                {
                    "type": "output_text",
                    "text": f"Max iterations ({max_iter}) reached. Stopping.",
                }
            ],
            "role": "assistant",
            "type": "message",
        }
        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=fallback_payload)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
            self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Entrypoint for streaming. We take the incoming request, prepend a SYSTEM message,
        then call our call_and_run_tools generator.
        """
        workspace_client = WorkspaceClient()

        # Build the initial message list: first the system prompt, then whatever the user passed in.
        initial_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]

        # Yield everything from call_and_run_tools(...)
        yield from self.call_and_run_tools(
            request_messages=initial_messages,
            workspace_client=workspace_client,
            max_iter=10,
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Non‐streaming version. We simply collect all “.item” dicts from predict_stream()
        that were marked type=“response.output_item.done” and bundle them into a ResponsesAgentResponse.
        """
        all_items = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=all_items, custom_outputs=request.custom_inputs)


# -----------------------------------------------------------------------------
# Specify the model object to use when loaded for deployment
# -----------------------------------------------------------------------------
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME)
mlflow.models.set_model(AGENT)
