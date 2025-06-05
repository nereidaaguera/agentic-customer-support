import json
from typing import Any, Callable, Generator, List, Optional
from uuid import uuid4

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from databricks.sdk import WorkspaceClient
from tools import get_mcp_tool_infos

LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
SYSTEM_PROMPT = "You are a helpful assistant."
MCP_SERVER_URLS = [
    "https://telco-outage-server-dev-3888667486068890.aws.databricksapps.com/mcp/",
]

class ToolCallingAgent(ChatAgent):
    def __init__(self, llm_endpoint: str):
        super().__init__()
        self.llm_endpoint = llm_endpoint

    def get_tool_specs(self, workspace_client):
        tool_infos = get_mcp_tool_infos(workspace_client=workspace_client, server_urls=MCP_SERVER_URLS)
        return [tool_info.spec for tool_info in tool_infos]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict, workspace_client) -> Any:
        tool_infos = get_mcp_tool_infos(workspace_client=workspace_client, server_urls=MCP_SERVER_URLS)
        tools_dict = {tool.name: tool for tool in tool_infos}
        return tools_dict[tool_name].exec_fn(**args)

    def prepare_messages_for_llm(self, messages: list[ChatAgentMessage]) -> list[dict[str, Any]]:
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        return [
            {k: v for k, v in m.model_dump_compat(exclude_none=True).items() if k in compatible_keys} for m in messages
        ]

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
            self,
            messages: List[ChatAgentMessage],
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        workspace_client = WorkspaceClient()
        response_messages = [
            chunk.delta
            for chunk in self.predict_stream(messages, context, custom_inputs)
        ]
        return ChatAgentResponse(messages=response_messages)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
            self,
            messages: List[ChatAgentMessage],
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        workspace_client = WorkspaceClient()
        all_messages = [
                           ChatAgentMessage(role="system", content=SYSTEM_PROMPT)
                       ] + messages

        for message in self.call_and_run_tools(messages=all_messages, workspace_client=workspace_client):
            yield ChatAgentChunk(delta=message)

    def chat_completion(self, messages: List[ChatAgentMessage], workspace_client):
        model_serving_client = workspace_client.serving_endpoints.get_open_ai_client()
        return model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=self.prepare_messages_for_llm(messages),
            tools=self.get_tool_specs(workspace_client),
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def call_and_run_tools(
            self, messages, workspace_client, max_iter=10
    ) -> Generator[ChatAgentMessage, None, None]:
        current_msg_history = messages.copy()
        for i in range(max_iter):
            with mlflow.start_span(span_type="AGENT", name=f"iteration_{i + 1}"):
                response = self.chat_completion(messages=current_msg_history, workspace_client=workspace_client)
                llm_message = response.choices[0].message
                assistant_message = ChatAgentMessage(**llm_message.to_dict(), id=str(uuid4()))
                current_msg_history.append(assistant_message)
                yield assistant_message

                tool_calls = assistant_message.tool_calls
                if not tool_calls:
                    return

                for tool_call in tool_calls:
                    function = tool_call.function
                    args = json.loads(function.arguments)
                    result = str(self.execute_tool(tool_name=function.name, args=args, workspace_client=workspace_client))
                    tool_call_msg = ChatAgentMessage(
                        role="tool", name=function.name, tool_call_id=tool_call.id, content=result, id=str(uuid4())
                    )
                    current_msg_history.append(tool_call_msg)
                    yield tool_call_msg

        yield ChatAgentMessage(
            content=f"I'm sorry, I couldn't determine the answer after trying {max_iter} times.",
            role="assistant",
            id=str(uuid4())
        )

mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME)
mlflow.models.set_model(AGENT)
