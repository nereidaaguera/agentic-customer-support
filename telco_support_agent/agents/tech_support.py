"""Tech support agent for handling technical support queries."""

import asyncio
import copy
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools.tech_support import TechSupportRetriever
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class ToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    spec: dict
    exec_fn: Callable


@asynccontextmanager
async def mcp_session(server_url: str, workspace_client: WorkspaceClient):
    """Async context manager that yields an initialized MCP ClientSession."""
    async with streamablehttp_client(
        url=server_url, auth=DatabricksOAuthClientProvider(workspace_client)
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def list_mcp_tools(server_url: str, workspace_client: WorkspaceClient):
    """List available MCP tools synchronously."""

    async def _inner():
        logger.debug(f"Listing tools from MCP server: {server_url}")
        async with mcp_session(server_url, workspace_client) as session:
            return await session.list_tools()

    return asyncio.run(_inner())


def make_mcp_exec_fn(
    mcp_server_url: str, tool_name: str, workspace_client: WorkspaceClient
):
    """Return a synchronous exec_fn that calls the named MCP tool."""

    def exec_fn(**kwargs):
        async def _call():
            async with mcp_session(mcp_server_url, workspace_client) as session:
                tool_call_res = await session.call_tool(
                    name=tool_name, arguments=kwargs
                )
                return "".join(
                    [content_obj.text for content_obj in tool_call_res.content]
                )

        return asyncio.run(_call())

    return exec_fn


def get_mcp_tool_infos(workspace_client: WorkspaceClient, server_urls: list[str]):
    """Discover and create ToolInfo objects for all MCP tools from given servers."""
    tool_infos = []
    for mcp_server_url in server_urls:
        try:
            # Pull dynamic MCP tool specs
            mcp_tools = list_mcp_tools(mcp_server_url, workspace_client)

            # Convert each into a ToolInfo and append
            for t in mcp_tools.tools:
                # Deep-copy t.inputSchema so we don't mutate the original MCP object
                final_schema = copy.deepcopy(t.inputSchema)
                final_tool_name = t.name[:64]

                if "properties" not in final_schema:
                    final_schema["properties"] = {}

                spec = {
                    "type": "function",
                    "function": {
                        "name": final_tool_name,
                        "description": t.description,
                        "parameters": final_schema,
                    },
                }
                tool_infos.append(
                    ToolInfo(
                        name=final_tool_name,
                        spec=spec,
                        exec_fn=make_mcp_exec_fn(
                            mcp_server_url, t.name, workspace_client
                        ),
                    )
                )
        except Exception:
            logger.exception(f"Failed to load MCP tools from {mcp_server_url}")

    return tool_infos


class TechSupportAgent(BaseAgent):
    """Tech support agent to handle technical support queries.

    This agent answers technical questions by searching both:
    - Knowledge base articles (official documentation, FAQs, guides)
    - Historical support tickets (similar issues and resolutions)
    - MCP servers for dynamic tool discovery and execution (outage info, network metrics, etc.)

    The agent combines information from both sources to provide technical support
    responses including troubleshooting steps, known issues, and proven resolution approaches.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        config_dir: Optional[str] = None,
        environment: str = "prod",
        override_mcp_server_urls: Optional[list[str]] = None,
        disable_tools: Optional[list[str]] = None,
    ) -> None:
        """Init agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
            environment: Environment to use for retrievers (dev, prod)
            disable_tools: Optional list of tool names to disable
            override_mcp_server_urls: Optional list of MCP server URLs to discover tools from. If specified,
            takes precedence over MCP servers specified in the config
        """
        # Initialize traditional retrieval tools
        self.retriever = TechSupportRetriever(environment=environment)
        retriever_tools = self.retriever.get_tools()

        # mapping of tool names to their executable objects
        vector_search_tools = {
            "knowledge_base_vector_search": self.retriever.kb_retriever.retriever,
            "support_tickets_vector_search": self.retriever.tickets_retriever.retriever,
        }

        # Discover and setup MCP tools
        mcp_tools = []
        self.mcp_tool_infos = []
        self.config = BaseAgent.load_config(agent_type="tech_support", config_dir=config_dir)
        mcp_server_urls = override_mcp_server_urls or [
            server_spec.server_url for server_spec in self.config.mcp_servers
        ]
        if mcp_server_urls:
            from databricks.sdk import WorkspaceClient
            workspace_client = WorkspaceClient()
            self.mcp_tool_infos = get_mcp_tool_infos(
                workspace_client, mcp_server_urls
            )
            mcp_tools.extend([tool_info.spec for tool_info in self.mcp_tool_infos])

        # Combine traditional retrieval tools with MCP tools
        all_tools = retriever_tools + mcp_tools

        logger.info(
            f"Tech support agent initialized with {len(retriever_tools)} retrieval tools and {len(mcp_tools)} MCP tools"
        )

        super().__init__(
            agent_type="tech_support",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=all_tools,  # tool specs for LLM (retrieval + MCP)
            vector_search_tools=vector_search_tools,  # executable objects
            disable_tools=disable_tools,
        )

    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute tool (UC function, vector search, or MCP tool)."""
        # Check if it's an MCP tool first
        if tool_name in [tool_info.name for tool_info in self.mcp_tool_infos]:
            tool_info = next(t for t in self.mcp_tool_infos if t.name == tool_name)
            return tool_info.exec_fn(**args)

        # Fall back to parent implementation for UC functions and vector search
        return super().execute_tool(tool_name, args)

    def get_description(self) -> str:
        """Return a description of this agent."""
        return (
            "Tech support agent that searches knowledge base articles, "
            "historical support tickets, and connects to MCP servers to provide "
            "comprehensive technical assistance including outage information and network metrics"
        )

if __name__ == "__main__":
    import logging
    import os
    logger = logging.getLogger("mlflow")
    logger.setLevel(logging.WARNING)
    AGENT = TechSupportAgent(
        llm_endpoint="dbdemos-openai-gpt4",
        config_dir=os.path.expanduser("~/genai-customer-support-demo/configs")
    )

    def _query_agent(query):
        for event in AGENT.predict_stream({"input": [{"role": "user", "content": query}]}):
            if item:= getattr(event, "item", None):
                item_type = item.get("type")

                # When the model requests a function/tool call:
                if item_type == "function_call":
                    func_name = item.get("name")
                    args_json = item.get("arguments", "{}")
                    # Pretty-print the tool name and its arguments
                    print(f"\n→ Calling tool: {func_name}({args_json})")

                # When the tool returns its output:
                elif item_type == "function_call_output":
                    raw_output = item.get("output", "")
                    print(f"Tool call result: {raw_output}")

                # 2) Otherwise, if this event has assistant content (text chunks), print them:
                elif item_type == "message":
                    for chunk in item.get("content", []):
                        if chunk.get("type") == "output_text":
                            # 'text' is a string containing whatever the assistant is saying
                            print(chunk["text"], end="")
                    print("")
                else:
                    print(f"Unexpected agent output item, displaying it anyways: {item}")

    query = "Is there an outage in Moscone center?"
    _query_agent(query=query)

