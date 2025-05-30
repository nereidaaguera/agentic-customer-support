import json
from typing import Any, Callable, Generator, List, Optional
from uuid import uuid4

import backoff
import mlflow
import openai
from mlflow.entities import SpanType
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from typing import Optional

import logging
from databricks.sdk.credentials_provider import ModelServingUserCredentials


from databricks.sdk import WorkspaceClient
from tools import get_mcp_tool_infos

logger = logging.getLogger(__name__)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

##################
# Set up MCP tools
##################
MCP_SERVER_URLS = [
    "https://uc-mcp-server-aravind-6051921418418893.staging.aws.databricksapps.com/mcp/"
]

class ToolCallingAgent(ChatAgent):
    """
    Class representing a tool-calling Agent
    """

    def __init__(self, llm_endpoint: str):
        """
        Initializes the ToolCallingAgent with tools.

        Args:
            tools (Dict[str, Dict[str, Any]]): A dictionary where each key is a tool name,
            and the value is a dictionary containing:
                - "spec" (dict): JSON description of the tool (matches OpenAI format)
                - "function" (Callable): Function that implements the tool logic
        """
        super().__init__()
        self.llm_endpoint = llm_endpoint
        self._tools_dict = None
        self.invokers_workspace_client = None
        self.definers_workspace_client = None
        self.model_serving_client = None
    
    def initialize_clients(self):
        if self.definers_workspace_client is None:
            self.definers_workspace_client = WorkspaceClient()
            self.invokers_workspace_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

        if self.model_serving_client is None:
            self.model_serving_client = self.definers_workspace_client.serving_endpoints.get_open_ai_client()

    def get_tool_specs(self):
        """
        Returns tool specifications in the format OpenAI expects.
        """
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """
        Executes the specified tool with the given arguments.

        Args:
            tool_name (str): The name of the tool to execute.
            args (dict): Arguments for the tool.

        Returns:
            Any: The tool's output.
        """
        return self._tools_dict[tool_name].exec_fn(**args)
    
    def initialize_tools(self):
        if self._tools_dict is None:
            all_tool_infos = get_mcp_tool_infos(self.invokers_workspace_client)
            self._tools_dict = {tool.name: tool for tool in all_tool_infos}

    def prepare_messages_for_llm(self, messages: list[ChatAgentMessage]) -> list[dict[str, Any]]:
        """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
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
        self.initialize_clients()
        self.initialize_tools()
        """
        Primary function that takes a user's request and generates a response.
        """
        # NOTE: this assumes that each chunk streamed by self.call_and_run_tools contains
        # a full message (i.e. chunk.delta is a complete message).
        # This is simple to implement, but you can also stream partial response messages from predict_stream,
        # and aggregate them in predict_stream by message ID
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
        self.initialize_clients()
        self.initialize_tools()

        all_messages = [
            ChatAgentMessage(role="system", content=SYSTEM_PROMPT)
        ] + messages
        
        for message in self.call_and_run_tools(messages=all_messages):
            yield ChatAgentChunk(delta=message)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def chat_completion(self, messages: List[ChatAgentMessage]):
        return self.model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=self.prepare_messages_for_llm(messages),
            tools=self.get_tool_specs(),
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def call_and_run_tools(
        self, messages, max_iter=10
    ) -> Generator[ChatAgentMessage, None, None]:
        current_msg_history = messages.copy()
        for i in range(max_iter):
            with mlflow.start_span(span_type="AGENT", name=f"iteration_{i + 1}"):
                # Get an assistant response from the model, add it to the running history
                # and yield it to the caller
                # NOTE: we perform a simple non-streaming chat completions here
                # Use the streaming API if you'd like to additionally do token streaming
                # of agent output.
                response = self.chat_completion(messages=current_msg_history)
                llm_message = response.choices[0].message
                assistant_message = ChatAgentMessage(**llm_message.to_dict(), id=str(uuid4()))
                current_msg_history.append(assistant_message)
                yield assistant_message

                tool_calls = assistant_message.tool_calls
                if not tool_calls:
                    return  # Stop streaming if no tool calls are needed

                # Execute tool calls, add them to the running message history,
                # and yield their results as tool messages
                for tool_call in tool_calls:
                    function = tool_call.function
                    args = json.loads(function.arguments)
                    # Cast tool result to a string, since not all tools return as tring
                    result = str(self.execute_tool(tool_name=function.name, args=args))
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


# Log the model using MLflow
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME)
mlflow.models.set_model(AGENT)
