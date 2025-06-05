import os
import json
import random
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn

from mcp.server.fastmcp import FastMCP
import mcp.types as types
from homepage import demo_homepage

load_dotenv()
DATABRICKS_APP_PORT = int(os.getenv("DATABRICKS_APP_PORT", 8000))  # Default to 8000 if not set

logger_name = "telco-mcp-server"

# --- Mock telco data for demo purposes ---
MOCK_OUTAGES = [
    {
        "outage_id": "OUT-2024-001",
        "region": "North Bay Area",
        "service_type": "5G",
        "status": "Active",
        "affected_customers": 15420,
        "started_at": "2024-01-15T10:30:00Z",
        "estimated_resolution": "2024-01-15T18:00:00Z",
        "description": "5G tower maintenance causing service disruption"
    },
    {
        "outage_id": "OUT-2024-002",
        "region": "Downtown LA",
        "service_type": "Fiber",
        "status": "Resolved",
        "affected_customers": 8750,
        "started_at": "2024-01-14T14:15:00Z",
        "resolved_at": "2024-01-14T22:45:00Z",
        "description": "Fiber cable damage due to construction work"
    }
]

MOCK_NETWORK_METRICS = {
    "regions": {
        "North Bay Area": {"uptime": 98.2, "latency_ms": 12, "packet_loss": 0.1},
        "Downtown LA": {"uptime": 99.8, "latency_ms": 8, "packet_loss": 0.02},
        "Orange County": {"uptime": 99.5, "latency_ms": 10, "packet_loss": 0.05},
        "San Diego": {"uptime": 99.9, "latency_ms": 6, "packet_loss": 0.01}
    }
}


async def simulate_telco_api_call(endpoint: str, delay_ms: int = None):
    """Simulate calling an external telco API with realistic delay"""
    if delay_ms is None:
        delay_ms = random.randint(200, 800)
    await asyncio.sleep(delay_ms / 1000)  # Convert to seconds
    return True

# --- FastMCP setup ---
mcp_app = FastMCP(name="telco-mcp-server", stateless_http=True)


@mcp_app.tool()
async def check_outage_status_tool(region: str):
    """Handle outage status check - simulates calling external telco API"""

    # Simulate API call
    await simulate_telco_api_call("outages")

    # Filter outages by region if specified
    outages = MOCK_OUTAGES
    if region:
        outages = [o for o in MOCK_OUTAGES if region.lower() in o["region"].lower()]

    response = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "region_queried": region or "All regions",
        "outages_found": len(outages),
        "outages": outages
    }

    return json.dumps(response, indent=2)


@mcp_app.tool()
async def get_network_metrics_tool(region: str):
    """Handle network metrics request - simulates calling external telco API"""

    # Simulate API call
    await simulate_telco_api_call("metrics")

    if region and region in MOCK_NETWORK_METRICS["regions"]:
        metrics = {region: MOCK_NETWORK_METRICS["regions"][region]}
    else:
        metrics = MOCK_NETWORK_METRICS["regions"]

    response = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "region_queried": region or "All regions",
        "metrics": metrics
    }
    return json.dumps(response, indent=2)


@mcp_app.tool()
async def report_network_issue_tool(issue_type: str, region: str, description: str) -> types.TextContent:
    """Handle network issue reporting - simulates calling external telco API"""

    # Simulate API call
    await simulate_telco_api_call("report-issue", delay_ms=1200)

    # Generate fake ticket
    ticket_id = f"TKT-{datetime.utcnow().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    response = {
        "ticket_id": ticket_id,
        "status": "Created",
        "issue_type": issue_type,
        "region": region,
        "description": description,
        "priority": "High" if issue_type in ["outage", "security"] else "Medium",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "estimated_resolution": (datetime.utcnow() + timedelta(hours=4)).isoformat() + "Z"
    }

    return json.dumps(response, indent=2)

# --- Mounting FastMCP under /mcp ---

import pdb
# pdb.set_trace()
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
