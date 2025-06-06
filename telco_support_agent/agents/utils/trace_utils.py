"""Trace and preview utilities for agent tracing."""

import json
from typing import Any, Optional

import mlflow
from mlflow.entities.trace_info import TraceInfo

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH = 10000


def compute_request_preview(request: Any) -> str:
    """Compute preview of request for tracing.

    Extracts most recent user message content for display in trace previews.

    Args:
        request: raw request string or dict to process

    Returns:
        preview string truncated to max length
    """
    preview = ""

    if isinstance(request, str):
        try:
            data = json.loads(request)
        except (json.JSONDecodeError, TypeError):
            preview = request
            return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]
    else:
        data = request

    if isinstance(data, dict):
        try:
            input_list = data.get("request", {}).get("input", [])
            if isinstance(input_list, list):
                for item in reversed(input_list):
                    if (
                        isinstance(item, dict)
                        and item.get("role") == "user"
                        and isinstance(item.get("content"), str)
                    ):
                        preview = item["content"]
                        break
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Error extracting user content from request: {e}")

    if not preview:
        preview = str(request)

    return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]


def compute_response_preview(response: Any) -> str:
    """Compute preview of response for tracing.

    Extracts assistant response for display in trace previews.

    Args:
        response: The raw response string or dict to process

    Returns:
        A preview string truncated to max length
    """
    preview = ""

    if isinstance(response, str):
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            preview = response
            return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]
    else:
        data = response

    if isinstance(data, dict):
        try:
            output = data.get("output")
            if isinstance(output, list):
                for item in reversed(output):
                    if (
                        isinstance(item, dict)
                        and item.get("role") == "assistant"
                        and isinstance(item.get("content"), list)
                    ):
                        for part in reversed(item["content"]):
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "output_text"
                                and isinstance(part.get("text"), str)
                            ):
                                preview = part["text"]
                                break
                        if preview:
                            break
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Error extracting assistant content from response: {e}")

    if not preview:
        preview = str(response)

    return preview[:TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH]


def patch_trace_info() -> None:
    """Apply monkey patch to TraceInfo for better trace previews."""
    # check if already patched to avoid double-patching
    if hasattr(TraceInfo, "_is_patched"):
        return

    original_init = TraceInfo.__init__

    def patched_init(self, request_preview=None, response_preview=None, **kwargs):
        """Patched TraceInfo.__init__ that computes better previews."""
        if request_preview is not None:
            request_preview = compute_request_preview(request_preview)
        if response_preview is not None:
            response_preview = compute_response_preview(response_preview)
        original_init(
            self,
            request_preview=request_preview,
            response_preview=response_preview,
            **kwargs,
        )

    TraceInfo.__init__ = patched_init
    TraceInfo._is_patched = True
    logger.info("Applied TraceInfo monkey patch for better previews")


def update_trace_preview(
    request_data: Optional[dict] = None,
    response_data: Optional[dict] = None,
    user_query: Optional[str] = None,
    assistant_response: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> None:
    """Update current trace with computed previews.

    Args:
        request_data: Full request data structure
        response_data: Full response data structure
        user_query: Simple user query string (alternative to request_data)
        assistant_response: Simple assistant response (alternative to response_data)
        customer_id: Customer ID for additional context
    """
    request_preview = None
    response_preview = None

    # handle request preview
    if request_data:
        request_preview = compute_request_preview(request_data)
    elif user_query:
        request_structure = create_request_structure(
            user_query, {"customer": customer_id} if customer_id else None
        )
        request_preview = compute_request_preview(request_structure)

    # handle response preview
    if response_data:
        response_preview = compute_response_preview(response_data)
    elif assistant_response:
        response_structure = create_response_structure(assistant_response)
        response_preview = compute_response_preview(response_structure)

    if request_preview is not None or response_preview is not None:
        mlflow.update_current_trace(
            request_preview=request_preview,
            response_preview=response_preview,
        )


def create_request_structure(
    user_query: str, custom_inputs: Optional[dict] = None
) -> dict:
    """Create properly formatted request structure for preview computation.

    Args:
        user_query: User's input query
        custom_inputs: Optional custom inputs (e.g., customer ID)

    Returns:
        Properly formatted request structure
    """
    structure = {"request": {"input": [{"role": "user", "content": user_query}]}}

    if custom_inputs:
        structure["request"]["custom_inputs"] = custom_inputs

    return structure


def create_response_structure(assistant_text: str) -> dict:
    """Create properly formatted response structure for preview computation.

    Args:
        assistant_text: Assistant's response text

    Returns:
        Properly formatted response structure
    """
    return {
        "output": [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            }
        ]
    }
