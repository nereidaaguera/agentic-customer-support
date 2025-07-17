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
    """Extract request text from inputs.

    Args:
        request: Request - can be string, list of messages, or inputs dictionary

    Returns:
        Request text
    """
    # Handle list of messages (chat format)
    if isinstance(request, list):
        # Get the last user message
        for msg in reversed(request):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return str(request)
    
    # Handle inputs dictionary format from mlflow.genai.evaluate
    if isinstance(request, dict):
        # Check for 'messages' key (evaluation format)
        if "messages" in request:
            return extract_request_text(request["messages"])
        # Check for 'input' key (Databricks agent format)
        elif "input" in request:
            input_data = request["input"]
            # Handle chat format
            if isinstance(input_data, list):
                return extract_request_text(input_data)
            return str(input_data)
        # Check for other common keys
        elif "question" in request:
            return str(request["question"])
        elif "query" in request:
            return str(request["query"])
        elif "text" in request:
            return str(request["text"])
    
    return str(request)


def extract_response_text(response: Any) -> str:
    """Extract response text from outputs.

    Args:
        response: Response - can be string or response dictionary

    Returns:
        Response text
    """
    # Handle response dictionary format
    if isinstance(response, dict):
        # Check common response keys
        if "response" in response:
            return str(response["response"])
        elif "output" in response:
            return str(response["output"])
        elif "content" in response:
            return str(response["content"])
        elif "text" in response:
            return str(response["text"])
        elif "answer" in response:
            return str(response["answer"])
        # For agent responses, might have nested structure
        elif "choices" in response and isinstance(response["choices"], list):
            if len(response["choices"]) > 0:
                choice = response["choices"][0]
                if isinstance(choice, dict) and "message" in choice:
                    msg = choice["message"]
                    if isinstance(msg, dict) and "content" in msg:
                        return str(msg["content"])
    
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
