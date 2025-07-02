"""Utilities for telco support agent evaluation."""

from typing import Any, Optional

from databricks.sdk import WorkspaceClient


def get_llm_client():
    """LLM client for evals.

    Returns:
        OpenAI client for Databricks model serving
    """
    workspace_client = WorkspaceClient()
    return workspace_client.serving_endpoints.get_open_ai_client()


def extract_request_text(request: Any) -> str:
    """Extract request text.

    Args:
        request: Request (expected to be string)

    Returns:
        Request text
    """
    return str(request)


def extract_response_text(response: Any) -> str:
    """Extract response text.

    Args:
        response: Response (expected to be string)

    Returns:
        Response text
    """
    return str(response)


def extract_trace_routing_info(trace: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Extract routing information from trace data.

    Args:
        trace: Trace dictionary containing execution information

    Returns:
        Dictionary with routing information (agent_type, topic, etc.)
    """
    routing_info = {}

    if not trace or not isinstance(trace, dict):
        return routing_info

    # custom_outputs for routing info
    if "custom_outputs" in trace:
        custom_outputs = trace["custom_outputs"]
        if isinstance(custom_outputs, dict):
            if "routing" in custom_outputs:
                routing = custom_outputs["routing"]
                if isinstance(routing, dict):
                    routing_info["agent_type"] = routing.get("agent_type", "unknown")
                    routing_info["disabled_tools"] = routing.get("disable_tools", [])

            if "topic" in custom_outputs:
                routing_info["topic"] = custom_outputs["topic"]

    # check spans for tool usage info
    if "spans" in trace:
        spans = trace["spans"]
        if isinstance(spans, list):
            tool_calls = []
            for span in spans:
                if isinstance(span, dict):
                    span_type = span.get("span_type")
                    if span_type == "TOOL":
                        tool_name = span.get("name", "").replace("tool_", "")
                        tool_calls.append(tool_name)
            routing_info["tools_used"] = tool_calls

    return routing_info
