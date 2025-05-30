import contextlib
import logging
from collections.abc import AsyncIterator
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from starlette.responses import HTMLResponse, JSONResponse
import uvicorn
from dotenv import load_dotenv
import os
import json
import asyncio
import random
from datetime import datetime, timedelta

load_dotenv()
DATABRICKS_APP_PORT = int(os.getenv("DATABRICKS_APP_PORT", 8000))  # Default to 8000 if not set
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telco-mcp-server")

app = Server("telco-mcp-server")

# Fake telco data for demo purposes
FAKE_OUTAGES = [
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

FAKE_NETWORK_METRICS = {
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

# Web interface routes for demo purposes
async def demo_homepage(request):
    """Serve a demo page showing the telco operations tools"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telco Operations MCP Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .tool-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            .button:hover { background: #005a8a; }
            .response-box { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 10px; border-left: 4px solid #007cba; }
            .loading { color: #666; font-style: italic; }
            pre { white-space: pre-wrap; }
            .status { padding: 8px 12px; border-radius: 4px; margin: 10px 0; }
            .status.running { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏢 Telco Operations MCP Server</h1>
                <p>Network Operations Team's Established Tools</p>
                <div class="status running">✅ Server Status: RUNNING</div>
            </div>
            
            <div class="tool-section">
                <h2>📡 Check Outage Status</h2>
                <p>Monitor current network outages across regions</p>
                <button class="button" onclick="callTool('outage-all')">Check All Regions</button>
                <button class="button" onclick="callTool('outage-bay')">Bay Area Only</button>
                <div id="outage-response" class="response-box" style="display:none;"></div>
            </div>
            
            <div class="tool-section">
                <h2>📊 Network Metrics</h2>
                <p>Get real-time network performance metrics</p>
                <button class="button" onclick="callTool('metrics-all')">All Regions</button>
                <button class="button" onclick="callTool('metrics-la')">Downtown LA</button>
                <div id="metrics-response" class="response-box" style="display:none;"></div>
            </div>
            
            <div class="tool-section">
                <h2>⚠️ Report Network Issue</h2>
                <p>Report incidents to the operations team</p>
                <button class="button" onclick="callTool('report-issue')">Report Degraded Performance</button>
                <div id="report-response" class="response-box" style="display:none;"></div>
            </div>
            
            <div class="tool-section">
                <h2>🔧 Available MCP Tools</h2>
                <button class="button" onclick="listTools()">List All Tools</button>
                <div id="tools-response" class="response-box" style="display:none;"></div>
            </div>
        </div>
        
        <script>
            async function callTool(toolType) {
                let endpoint, data, responseDiv;
                
                switch(toolType) {
                    case 'outage-all':
                        endpoint = '/demo/check-outage-status';
                        data = {};
                        responseDiv = 'outage-response';
                        break;
                    case 'outage-bay':
                        endpoint = '/demo/check-outage-status';
                        data = { region: 'Bay Area' };
                        responseDiv = 'outage-response';
                        break;
                    case 'metrics-all':
                        endpoint = '/demo/get-network-metrics';
                        data = {};
                        responseDiv = 'metrics-response';
                        break;
                    case 'metrics-la':
                        endpoint = '/demo/get-network-metrics';
                        data = { region: 'Downtown LA' };
                        responseDiv = 'metrics-response';
                        break;
                    case 'report-issue':
                        endpoint = '/demo/report-network-issue';
                        data = { 
                            issue_type: 'degraded_performance', 
                            region: 'Bay Area',
                            description: 'Multiple customer reports of slow data speeds during peak hours'
                        };
                        responseDiv = 'report-response';
                        break;
                }
                
                const div = document.getElementById(responseDiv);
                div.style.display = 'block';
                div.innerHTML = '<div class="loading">🔄 Calling telco operations API...</div>';
                
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    div.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                } catch (error) {
                    div.innerHTML = '<div style="color: red;">Error: ' + error.message + '</div>';
                }
            }
            
            async function listTools() {
                const div = document.getElementById('tools-response');
                div.style.display = 'block';
                div.innerHTML = '<div class="loading">🔄 Fetching available tools...</div>';
                
                try {
                    const response = await fetch('/demo/list-tools');
                    const result = await response.json();
                    div.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                } catch (error) {
                    div.innerHTML = '<div style="color: red;">Error: ' + error.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

async def demo_check_outage_status(request):
    """Demo endpoint for outage status"""
    data = await request.json() if request.method == "POST" else {}
    region = data.get("region")
    
    # Simulate API delay
    await simulate_telco_api_call("outages")
    
    # Filter outages by region if specified
    outages = FAKE_OUTAGES
    if region:
        outages = [o for o in FAKE_OUTAGES if region.lower() in o["region"].lower()]
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "region_queried": region or "All regions",
        "outages_found": len(outages),
        "outages": outages,
        "source": "Telco Operations MCP Server"
    }
    
    return JSONResponse(response)

async def demo_get_network_metrics(request):
    """Demo endpoint for network metrics"""
    data = await request.json() if request.method == "POST" else {}
    region = data.get("region")
    
    # Simulate API delay
    await simulate_telco_api_call("metrics")
    
    if region and region in FAKE_NETWORK_METRICS["regions"]:
        metrics = {region: FAKE_NETWORK_METRICS["regions"][region]}
    else:
        metrics = FAKE_NETWORK_METRICS["regions"]
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "region_queried": region or "All regions", 
        "metrics": metrics,
        "source": "Telco Operations MCP Server"
    }
    
    return JSONResponse(response)

async def demo_report_network_issue(request):
    """Demo endpoint for reporting issues"""
    data = await request.json()
    issue_type = data.get("issue_type")
    region = data.get("region") 
    description = data.get("description")
    
    # Simulate API delay
    await simulate_telco_api_call("report-issue", delay_ms=1200)
    
    # Generate fake ticket
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    
    response = {
        "ticket_id": ticket_id,
        "status": "Created",
        "issue_type": issue_type,
        "region": region,
        "description": description,
        "priority": "High" if issue_type in ["outage", "security"] else "Medium",
        "created_at": datetime.now().isoformat(),
        "estimated_resolution": (datetime.now() + timedelta(hours=4)).isoformat(),
        "source": "Telco Operations MCP Server"
    }
    
    return JSONResponse(response)

async def demo_list_tools(request):
    """Demo endpoint to list available tools"""
    tools = [
        {
            "name": "check-outage-status",
            "description": "Check current network outage status across regions (from Telco Operations team)",
            "parameters": ["region (optional)"]
        },
        {
            "name": "get-network-metrics",
            "description": "Get real-time network performance metrics by region (from Telco Operations team)",
            "parameters": ["region (optional)"]
        },
        {
            "name": "report-network-issue",
            "description": "Report a network issue to the operations team (from Telco Operations team)",
            "parameters": ["issue_type", "region", "description"]
        }
    ]
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "server": "Telco Operations MCP Server",
        "tools_available": len(tools),
        "tools": tools
    }
    
    return JSONResponse(response)

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls for telco operations."""
    ctx = app.request_context
    
    if name == "check-outage-status":
        return await handle_check_outage_status(arguments, ctx)
    elif name == "get-network-metrics":
        return await handle_get_network_metrics(arguments, ctx)
    elif name == "report-network-issue":
        return await handle_report_network_issue(arguments, ctx)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_check_outage_status(arguments: dict, ctx) -> list[types.TextContent]:
    """Handle outage status check - simulates calling external telco API"""
    region = arguments.get("region")
    
    await ctx.session.send_log_message(
        level="info",
        data=f"Checking outage status for region: {region}...",
        logger="telco-mcp-server",
        related_request_id=ctx.request_id,
    )
    
    # Simulate API call
    await simulate_telco_api_call("outages")
    
    # Filter outages by region if specified
    outages = FAKE_OUTAGES
    if region:
        outages = [o for o in FAKE_OUTAGES if region.lower() in o["region"].lower()]
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "region_queried": region or "All regions",
        "outages_found": len(outages),
        "outages": outages
    }
    
    await ctx.session.send_log_message(
        level="info",
        data=f"Found {len(outages)} outages for region",
        logger="telco-mcp-server", 
        related_request_id=ctx.request_id,
    )
    
    return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

async def handle_get_network_metrics(arguments: dict, ctx) -> list[types.TextContent]:
    """Handle network metrics request - simulates calling external telco API"""
    region = arguments.get("region")
    
    await ctx.session.send_log_message(
        level="info", 
        data=f"Fetching network metrics for region: {region}...",
        logger="telco-mcp-server",
        related_request_id=ctx.request_id,
    )
    
    # Simulate API call
    await simulate_telco_api_call("metrics")
    
    if region and region in FAKE_NETWORK_METRICS["regions"]:
        metrics = {region: FAKE_NETWORK_METRICS["regions"][region]}
    else:
        metrics = FAKE_NETWORK_METRICS["regions"]
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "region_queried": region or "All regions", 
        "metrics": metrics
    }
    
    await ctx.session.send_log_message(
        level="info",
        data="Network metrics retrieved successfully",
        logger="telco-mcp-server",
        related_request_id=ctx.request_id,
    )
    
    return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

async def handle_report_network_issue(arguments: dict, ctx) -> list[types.TextContent]:
    """Handle network issue reporting - simulates calling external telco API"""
    issue_type = arguments.get("issue_type")
    region = arguments.get("region") 
    description = arguments.get("description")
    
    await ctx.session.send_log_message(
        level="info",
        data=f"Reporting {issue_type} issue in {region}...",
        logger="telco-mcp-server", 
        related_request_id=ctx.request_id,
    )
    
    # Simulate API call
    await simulate_telco_api_call("report-issue", delay_ms=1200)
    
    # Generate fake ticket
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    
    response = {
        "ticket_id": ticket_id,
        "status": "Created",
        "issue_type": issue_type,
        "region": region,
        "description": description,
        "priority": "High" if issue_type in ["outage", "security"] else "Medium",
        "created_at": datetime.now().isoformat(),
        "estimated_resolution": (datetime.now() + timedelta(hours=4)).isoformat()
    }
    
    await ctx.session.send_log_message(
        level="info",
        data=f"Issue reported successfully. Ticket ID: {ticket_id}",
        logger="telco-mcp-server",
        related_request_id=ctx.request_id,
    )
    
    return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available telco operations tools."""
    return [
        types.Tool(
            name="check-outage-status",
            description="Check current network outage status across regions (from Telco Operations team)",
            inputSchema={
                "type": "object", 
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Specific region to check (optional). Examples: 'Bay Area', 'LA', 'Orange County'",
                    }
                },
            },
        ),
        types.Tool(
            name="get-network-metrics",
            description="Get real-time network performance metrics by region (from Telco Operations team)",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string", 
                        "description": "Specific region to get metrics for (optional)",
                    }
                },
            },
        ),
        types.Tool(
            name="report-network-issue",
            description="Report a network issue to the operations team (from Telco Operations team)",
            inputSchema={
                "type": "object",
                "required": ["issue_type", "region", "description"],
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "enum": ["outage", "degraded_performance", "security", "maintenance"],
                        "description": "Type of network issue being reported",
                    },
                    "region": {
                        "type": "string",
                        "description": "Region where the issue is occurring",
                    },
                    "description": {
                        "type": "string", 
                        "description": "Detailed description of the issue",
                    }
                },
            },
        )
    ]

session_manager = StreamableHTTPSessionManager(
    app=app,
    event_store=None,  
    stateless=True,
)

async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
    await session_manager.handle_request(scope, receive, send)

@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        logger.info("Telco Operations MCP server started!")
        try:
            yield
        finally:
            logger.info("Telco Operations MCP server stopped!")


starlette_app = Starlette(
    debug=False,
    routes=[
        Route("/", demo_homepage),
        Route("/demo/check-outage-status", demo_check_outage_status, methods=["GET", "POST"]),
        Route("/demo/get-network-metrics", demo_get_network_metrics, methods=["GET", "POST"]),
        Route("/demo/report-network-issue", demo_report_network_issue, methods=["POST"]),
        Route("/demo/list-tools", demo_list_tools),
        Mount("/api/mcp", app=handle_streamable_http)
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=DATABRICKS_APP_PORT)



