import json
from typing import Any, Callable, Generator, List, Optional
from uuid import uuid4

import backoff
import mlflow
import openai
from databricks_openai import VectorSearchRetrieverTool, UCFunctionToolkit
from unitycatalog.ai.core.base import get_uc_function_client
from mlflow.entities import SpanType
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel

from typing import Optional

import logging
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from unitycatalog.ai.core.databricks import DatabricksFunctionClient


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
    # Open bidirectional stream
    async with streamablehttp_client(
            url=server_url,
            auth=workspace_client.mcp.get_oauth_provider()
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def list_mcp_tools(server_url: str, workspace_client: WorkspaceClient):
    """
    List available MCP tools synchronously using the shared mcp_session context
    """
    async def _inner():
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
                return await session.call_tool(name=tool_name, arguments=kwargs)

        return asyncio.run(_call())

    return exec_fn


def get_mcp_tool_infos(workspace_client: WorkspaceClient, server_urls: list[str]):
    tool_infos = []
    for mcp_server_url in server_urls:
        # 1. pull dynamic MCP tool specs
        mcp_tools = list_mcp_tools(mcp_server_url, workspace_client)
        # 2. convert each into a ToolInfo and append
        for t in mcp_tools.tools:
            # build an OpenAI-style spec from the MCP schema
            spec = {
                "type": "function",
                "function": {
                    "name":        t.name,
                    "description": t.description,
                    "parameters":  t.inputSchema
                }
            }
            tool_infos.append(
                ToolInfo(
                    name=t.name,
                    spec=spec,
                    exec_fn=make_mcp_exec_fn(mcp_server_url, t.name, workspace_client),
                )
            )
    return tool_infos
