import os
from dotenv import load_dotenv

from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn

from mcp.server.fastmcp import FastMCP
import mcp.types as types
from homepage import demo_homepage
from telco_api import call_telco_service

load_dotenv()
DATABRICKS_APP_PORT = int(os.getenv("DATABRICKS_APP_PORT", 8000))  # Default to 8000 if not set

# --- FastMCP setup ---
mcp_app = FastMCP(name="telco-mcp-server", stateless_http=True)

@mcp_app.tool()
async def check_outage_status_tool(region: str):
    """Handle outage status check by simulating GET /outages."""
    params = {"region": region} if region else {}
    return await call_telco_service("GET", "/outages", params=params)

@mcp_app.tool()
async def get_network_metrics_tool(region: str):
    """Handle network metrics request by simulating GET /metrics."""
    params = {"region": region} if region else {}
    return await call_telco_service("GET", "/metrics", params=params)

@mcp_app.tool()
async def report_network_issue_tool(issue_type: str, region: str, description: str) -> types.TextContent:
    """Handle network issue reporting by simulating POST /report."""
    payload = {"issue_type": issue_type, "region": region, "description": description}
    return await call_telco_service("POST", "/report", body=payload)

# --- Mounting FastMCP under /mcp ---
starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/", demo_homepage),
        Mount("/", app=mcp_app.streamable_http_app()),
    ],
    lifespan=lambda app: mcp_app.session_manager.run(),
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=DATABRICKS_APP_PORT)