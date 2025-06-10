from typing import Any, Callable, Generator, List, Optional
from pydantic import BaseModel

from databricks_mcp import DatabricksOAuthClientProvider

import asyncio
from contextlib import asynccontextmanager
from databricks.sdk import WorkspaceClient
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable

@asynccontextmanager
async def mcp_session(server_url: str, workspace_client: WorkspaceClient):
    """
    Async context manager that yields an initialized MCP ClientSession
    """
    async with streamablehttp_client(
            url=server_url,
            auth=DatabricksOAuthClientProvider(workspace_client)
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def list_mcp_tools(server_url: str, workspace_client: WorkspaceClient):
    """
    List available MCP tools synchronously using the shared mcp_session context
    """
    async def _inner():
        print("Listing tools from MCP server:", server_url)
        async with mcp_session(server_url, workspace_client) as session:
            return await session.list_tools()

    return asyncio.run(_inner())


def make_mcp_exec_fn(
        mcp_server_url: str,
        tool_name: str,
        workspace_client: WorkspaceClient
):
    """
    Return a synchronous exec_fn that calls the named MCP tool using shared mcp_session
    """
    def exec_fn(**kwargs):
        async def _call():
            async with mcp_session(mcp_server_url, workspace_client) as session:
                tool_call_res = await session.call_tool(name=tool_name, arguments=kwargs)
                return "".join([content_obj.text for content_obj in tool_call_res.content])

        return asyncio.run(_call())

    return exec_fn

import copy

def get_mcp_tool_infos(workspace_client: WorkspaceClient, server_urls: list[str]):
    tool_infos = []
    for mcp_server_url in server_urls:
        # 1. pull dynamic MCP tool specs
        mcp_tools = list_mcp_tools(mcp_server_url, workspace_client)

        # 2. convert each into a ToolInfo and append
        for t in mcp_tools.tools:

            # Deep‐copy t.inputSchema so we don't mutate the original MCP object
            final_schema = copy.deepcopy(t.inputSchema)
            final_tool_name = t.name[:64]

            if "properties" not in final_schema:
                final_schema["properties"] = {}

            spec = {
                "type": "function",
                "function": {
                    "name":        final_tool_name,
                    "description": t.description,
                    "parameters":  final_schema
                }
            }
            tool_infos.append(
                ToolInfo(
                    name=final_tool_name,
                    spec=spec,
                    exec_fn=make_mcp_exec_fn(mcp_server_url, t.name, workspace_client),
                )
            )

    return tool_infos
