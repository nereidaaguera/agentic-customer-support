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
        except Exception as e:
            logger.error(f"Failed to load MCP tools from {mcp_server_url}: {e}")

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
        disable_tools: Optional[list[str]] = None,
        mcp_server_urls: Optional[list[str]] = None,
    ) -> None:
        """Init agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
            environment: Environment to use for retrievers (dev, prod)
            disable_tools: Optional list of tool names to disable
            mcp_server_urls: Optional list of MCP server URLs to discover tools from
        """
        # Initialize traditional retrieval tools
        self.retriever = TechSupportRetriever(environment=environment)
        retriever_tools = self.retriever.get_tools()

        # mapping of tool names to their executable objects
        vector_search_tools = {
            "knowledge_base_vector_search": self.retriever.kb_retriever.retriever,
            "support_tickets_vector_search": self.retriever.tickets_retriever.retriever,
        }

        # Initialize MCP tools if server URLs are provided
        self.mcp_server_urls = mcp_server_urls or [
            "https://e2-demo-west.cloud.databricks.com/api/2.0/mcp/vector-search/telco_customer_support_dev/agent",
            "https://e2-demo-west.cloud.databricks.com/api/2.0/mcp/functions/telco_customer_support_dev/agent",
            "https://mcp-telco-outage-info-server-2556758628403379.aws.databricksapps.com/mcp/",
        ]

        # Discover and setup MCP tools
        mcp_tools = []
        self.mcp_tool_infos = []
        if self.mcp_server_urls:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

            self.mcp_tool_infos = get_mcp_tool_infos(
                workspace_client, self.mcp_server_urls
            )
            mcp_tools = [tool_info.spec for tool_info in self.mcp_tool_infos]

            # Add MCP tools to vector_search_tools mapping for execution
            for tool_info in self.mcp_tool_infos:
                vector_search_tools[tool_info.name] = tool_info

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
