"""Mocked telco API client module.

Exposes a generic `call_telco_service` function that simulates HTTP requests to a backend.
Internally routes calls to in-memory handlers based on method and path.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta

# In-memory mock data for outages and network metrics
_OUTAGES = [
    {
        "outage_id": "OUT-2025-001",
        "region": "North Bay Area",
        "service_type": "5G",
        "status": "Resolved",
        "affected_customers": 15420,
        "started_at": "2025-06-03T10:30:00Z",
        "estimated_resolution": "2025-06-03T18:00:00Z",
        "description": "5G tower maintenance causing service disruption",
    },
    {
        "outage_id": "OUT-2025-002",
        "region": "Downtown LA",
        "service_type": "Fiber",
        "status": "Resolved",
        "affected_customers": 8750,
        "started_at": "2025-05-31T14:15:00Z",
        "resolved_at": "2025-06-01T22:45:00Z",
        "description": "Fiber cable damage due to construction work",
    },
    {
        "outage_id": "OUT-2025-003",
        "region": "San Francisco",
        "service_type": "5G",
        "status": "Active",
        "affected_customers": 20050,
        "started_at": "2025-06-11T10:30:00Z",
        "estimated_resolution": "2025-06-11T18:00:00Z",
        "description": "5G tower maintenance causing service disruption",
    },
]

_NETWORK_METRICS = {
    "North Bay Area": {"uptime": 98.2, "latency_ms": 12, "packet_loss": 0.1},
    "Downtown LA": {"uptime": 99.8, "latency_ms": 8, "packet_loss": 0.02},
    "Orange County": {"uptime": 99.5, "latency_ms": 10, "packet_loss": 0.05},
    "San Diego": {"uptime": 99.9, "latency_ms": 6, "packet_loss": 0.01},
}


async def _simulate_delay(min_ms: int = 200, max_ms: int = 500):
    """Simulate network latency."""
    await asyncio.sleep(random.randint(min_ms, max_ms) / 1000)


async def _handle_get_outages(params: dict) -> str:
    region = params.get("region") if params else None
    await _simulate_delay()
    outages = _OUTAGES
    response = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "region_queried": region or "All regions",
        "outages": outages,
    }
    return json.dumps(response, indent=2)


async def _handle_get_metrics(params: dict) -> str:
    region = params.get("region") if params else None
    await _simulate_delay()
    if region and region in _NETWORK_METRICS:
        metrics = {region: _NETWORK_METRICS[region]}
    else:
        metrics = _NETWORK_METRICS

    response = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "region_queried": region or "All regions",
        "metrics": metrics,
    }
    return json.dumps(response, indent=2)


async def _handle_post_report(payload: dict) -> str:
    issue_type = payload.get("issue_type")
    region = payload.get("region")
    description = payload.get("description")
    await _simulate_delay(min_ms=500, max_ms=500)
    ticket_id = (
        f"TKT-{datetime.utcnow().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    )
    response = {
        "ticket_id": ticket_id,
        "status": "Created",
        "issue_type": issue_type,
        "region": region,
        "description": description,
        "priority": "High" if issue_type in ["outage", "security"] else "Medium",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "estimated_resolution": (datetime.utcnow() + timedelta(hours=4)).isoformat()
        + "Z",
    }
    return json.dumps(response, indent=2)


async def call_telco_service(
    method: str, path: str, params: dict = None, body: dict = None
) -> str:
    """Generic telco API call simulation. Routes requests based on HTTP method and API path.

    Example:
        await call_telco_service("GET", "/outages", params={"region": "Downtown LA"})
        await call_telco_service("POST", "/report", body={"issue_type": "outage", "region": "San Diego", "description": "Dropped calls"})
    """
    # Log incoming request (in real code, you might log method/path)
    # For the demo, we just route based on path
    if method.upper() == "GET" and path == "/outages":
        return await _handle_get_outages(params)
    elif method.upper() == "GET" and path == "/metrics":
        return await _handle_get_metrics(params)
    elif method.upper() == "POST" and path == "/report":
        return await _handle_post_report(body)
    else:
        # Simulate a 404-like response
        await _simulate_delay()
        return json.dumps({"error": "Unknown endpoint"}, indent=2)
