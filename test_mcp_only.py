#!/usr/bin/env python3
"""Test MCP functionality only, without vector search dependencies."""

import os
import sys
import copy
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

# Add project root to path
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_path)

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel

from telco_support_agent.utils.logging_utils import setup_logging, get_logger

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
        url=server_url,
        auth=DatabricksOAuthClientProvider(workspace_client)
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def list_mcp_tools(server_url: str, workspace_client: WorkspaceClient):
    """List available MCP tools synchronously."""
    async def _inner():
        logger.info(f"Listing tools from MCP server: {server_url}")
        async with mcp_session(server_url, workspace_client) as session:
            return await session.list_tools()

    return asyncio.run(_inner())


def make_mcp_exec_fn(
    mcp_server_url: str,
    tool_name: str,
    workspace_client: WorkspaceClient
):
    """Return a synchronous exec_fn that calls the named MCP tool."""
    def exec_fn(**kwargs):
        async def _call():
            async with mcp_session(mcp_server_url, workspace_client) as session:
                tool_call_res = await session.call_tool(name=tool_name, arguments=kwargs)
                return "".join([content_obj.text for content_obj in tool_call_res.content])

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
                        "parameters": final_schema
                    }
                }
                tool_infos.append(
                    ToolInfo(
                        name=final_tool_name,
                        spec=spec,
                        exec_fn=make_mcp_exec_fn(mcp_server_url, t.name, workspace_client),
                    )
                )
        except Exception as e:
            logger.error(f"Failed to load MCP tools from {mcp_server_url}: {e}")

    return tool_infos


def test_mcp_discovery():
    """Test MCP tool discovery."""
    print("Testing MCP tool discovery...")
    
    try:
        workspace_client = WorkspaceClient()
        
        # Test with your outage server
        mcp_server_urls = [
            "https://mcp-telco-outage-server-3217006663075879.aws.databricksapps.com/mcp/"
        ]
        
        print(f"Discovering tools from {len(mcp_server_urls)} MCP servers...")
        tool_infos = get_mcp_tool_infos(workspace_client, mcp_server_urls)
        
        print(f"✓ Discovered {len(tool_infos)} MCP tools")
        
        for i, tool_info in enumerate(tool_infos, 1):
            spec = tool_info.spec["function"]
            print(f"  {i}. {spec['name']}: {spec.get('description', 'No description')[:80]}...")
        
        return tool_infos
        
    except Exception as e:
        print(f"✗ Error discovering MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_mcp_execution(tool_infos):
    """Test executing an MCP tool."""
    if not tool_infos:
        print("⚠ No tools to test")
        return False
        
    print("\nTesting MCP tool execution...")
    
    # Find the outage checking tool
    outage_tool = None
    for tool_info in tool_infos:
        if "outage" in tool_info.name.lower():
            outage_tool = tool_info
            break
    
    if not outage_tool:
        print("⚠ No outage tool found for testing")
        return False
    
    try:
        print(f"Executing tool: {outage_tool.name}")
        result = outage_tool.exec_fn(region="Moscone Center")
        print(f"✓ Tool executed successfully")
        print(f"Result: {str(result)[:200]}...")
        return True
        
    except Exception as e:
        print(f"✗ Error executing tool: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("MCP-Only Integration Test")
    print("=" * 40)
    
    # Test tool discovery
    tool_infos = test_mcp_discovery()
    
    if not tool_infos:
        print("\n✗ MCP tool discovery failed!")
        return False
    
    # Test tool execution
    success = test_mcp_execution(tool_infos)
    
    if success:
        print("\n✓ MCP integration test completed successfully!")
        return True
    else:
        print("\n✗ MCP integration test failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)