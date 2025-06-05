import json
from typing import Any, Generator, List, Optional
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
from mlflow.types.chat import ToolCall, Function

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

    def get_tool_specs(self, workspace_client: WorkspaceClient) -> List[dict[str, Any]]:
        """
        Pull all MCP tools for this workspace, then return a list of OpenAI-style specs.
        """
        tool_infos = get_mcp_tool_infos(
            workspace_client=workspace_client, server_urls=MCP_SERVER_URLS
        )
        return [tool_info.spec for tool_info in tool_infos]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict, workspace_client: WorkspaceClient) -> Any:
        """
        Given a tool name + arguments, find the correct ToolInfo and run it.
        """
        tool_infos = get_mcp_tool_infos(
            workspace_client=workspace_client, server_urls=MCP_SERVER_URLS
        )
        tools_dict = {tool.name: tool for tool in tool_infos}
        return tools_dict[tool_name].exec_fn(**args)

    def prepare_messages_for_llm(
            self, messages: List[ChatAgentMessage]
    ) -> List[dict[str, Any]]:
        """
        Convert MLflow ChatAgentMessage objects into the minimal dict form the
        Databricks “OpenAI-compatible” client expects.
        """
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        out: List[dict[str, Any]] = []
        for m in messages:
            raw = m.model_dump_compat(exclude_none=True)
            filtered = {k: v for k, v in raw.items() if k in compatible_keys}
            out.append(filtered)
        return out

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
            self,
            messages: List[ChatAgentMessage],
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """
        Non‐streaming wrapper: collects all streamed chunks into a full response.
        """
        workspace_client = WorkspaceClient()
        response_messages = [
            chunk.delta for chunk in self.predict_stream(messages, context, custom_inputs)
        ]
        return ChatAgentResponse(messages=response_messages)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
            self,
            messages: List[ChatAgentMessage],
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """
        Streaming version: yield ChatAgentChunk tokens (or full messages) as they arrive.
        """
        workspace_client = WorkspaceClient()
        all_messages = [ChatAgentMessage(role="system", content=SYSTEM_PROMPT)] + messages

        # Delegate to call_and_run_tools, which yields ChatAgentChunk objects in streaming fashion.
        for chunk in self.call_and_run_tools(all_messages, workspace_client):
            yield chunk

    def chat_completion(
            self,
            messages: List[ChatAgentMessage],
            workspace_client: WorkspaceClient,
            stream: bool = False,
    ):
        """
        Perform a chat completion against the Databricks OpenAI-compatible endpoint.
        If stream=True, returns an iterator of streaming chunks; otherwise a single response.
        """
        model_serving_client = workspace_client.serving_endpoints.get_open_ai_client()
        if stream:
            return model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=self.prepare_messages_for_llm(messages),
                tools=self.get_tool_specs(workspace_client),
                stream=True,
            )
        else:
            return model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=self.prepare_messages_for_llm(messages),
                tools=self.get_tool_specs(workspace_client),
                stream=False,
            )

    @mlflow.trace(span_type=SpanType.AGENT)
    def call_and_run_tools(
            self, messages: List[ChatAgentMessage], workspace_client: WorkspaceClient, max_iter: int = 10
    ) -> Generator[ChatAgentChunk, None, None]:
        """
        Each iteration:
          1. Start streaming chat_completion (stream=True).
          2. For each incoming chunk:
             - If chunk.delta.content is present, stream it immediately.
             - If chunk.delta.tool_calls is non-empty, begin assembling the tool call.
          3. Once we detect a complete tool_call (i.e. have at least a name and arguments):
             - Yield a final ChatAgentChunk wrapping the full ChatAgentMessage(role="assistant", tool_calls=[...]).
             - Execute the tool, yield its output as ChatAgentChunk(role="tool", ...), append to history.
             - Loop again (next iteration) so the LLM can continue after the tool.
          4. If no tool_call appears and the LLM finishes, yield the final assistant message once (all content
             tokens were already streamed), append to history, and return.
        """
        # Copy the full history so far
        current_history: List[ChatAgentMessage] = messages.copy()

        for iteration in range(max_iter):
            with mlflow.start_span(span_type=SpanType.AGENT, name=f"iteration_{iteration + 1}"):
                # 1) Kick off a streaming LLM call
                stream_resp = self.chat_completion(
                    messages=current_history, workspace_client=workspace_client, stream=True
                )

                # Buffers to accumulate the current assistant message
                content_buffer = ""
                function_name: Optional[str] = None
                arguments_buffer = ""
                tool_call_id: Optional[str] = None

                # 2) Iterate over streaming chunks until we either see a tool call or finish normally
                for chunk in stream_resp:
                    delta_obj = chunk.choices[0].delta

                    # (a) If there's partial content, yield it immediately
                    if getattr(delta_obj, "content", None) is not None:
                        text_piece = delta_obj.content
                        content_buffer += text_piece
                        partial_msg = ChatAgentMessage(
                            role="assistant", content=text_piece, id=str(uuid4())
                        )
                        yield ChatAgentChunk(delta=partial_msg)

                    # (b) If there's any tool_calls in this delta, process them
                    tool_calls_list = getattr(delta_obj, "tool_calls", None)
                    if tool_calls_list:
                        # We'll assume only one tool_call is being streamed at a time.
                        for tc in tool_calls_list:
                            fc = tc.function  # ChoiceDeltaToolCallFunction
                            # If name appears, record it
                            if getattr(fc, "name", None):
                                function_name = fc.name
                                if tool_call_id is None:
                                    tool_call_id = str(uuid4())
                            # If arguments appear, append to our JSON buffer
                            if getattr(fc, "arguments", None):
                                arguments_buffer += fc.arguments

                        # If we have seen both name and some arguments, break out to execute
                        if function_name is not None and arguments_buffer is not None:
                            break

                # 3) After exiting the chunk loop:
                #    (a) If function_name is still None → no tool call was emitted
                if function_name is None:
                    # The LLM produced a plain text answer. All content was already streamed above.
                    # Now append one final assistant message and return.
                    final_assistant = ChatAgentMessage(
                        role="assistant", content=content_buffer, id=str(uuid4())
                    )
                    current_history.append(final_assistant)
                    return

                # 4) We detected a tool call. Assemble the full ToolCall
                full_tool_call = ToolCall(
                    id=tool_call_id, function=Function(name=function_name, arguments=arguments_buffer)
                )
                assistant_with_tool = ChatAgentMessage(
                    role="assistant",
                    content=content_buffer if content_buffer != "" else None,
                    tool_calls=[full_tool_call],
                    id=str(uuid4()),
                )

                # Yield a chunk indicating “assistant is calling a tool now”
                yield ChatAgentChunk(delta=assistant_with_tool)
                current_history.append(assistant_with_tool)

                # 5) Execute the tool synchronously
                try:
                    parsed_args = json.loads(arguments_buffer)
                except json.JSONDecodeError:
                    parsed_args = {}
                tool_result = self.execute_tool(
                    tool_name=function_name, args=parsed_args, workspace_client=workspace_client
                )
                tool_content = str(tool_result)

                # Wrap the tool’s output in a ChatAgentMessage(role="tool", …) and yield it
                tool_msg = ChatAgentMessage(
                    role="tool",
                    name=function_name,
                    tool_call_id=tool_call_id,
                    content=tool_content,
                    id=str(uuid4()),
                )
                current_history.append(tool_msg)
                yield ChatAgentChunk(delta=tool_msg)

                # Loop around for the next iteration (the LLM will now see the tool response in history).

        # If we exit the loop without returning, we hit max_iter without an answer
        yield ChatAgentChunk(
            delta=ChatAgentMessage(
                role="assistant",
                content=f"I'm sorry, I couldn't determine the answer after trying {max_iter} times.",
                id=str(uuid4()),
            )
        )

# Register autolog, then set the model as before
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME)
mlflow.models.set_model(AGENT)
