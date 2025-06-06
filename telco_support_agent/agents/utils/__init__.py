"""Agent utilities package."""

from .exceptions import (
    AgentConfigurationError,
    AgentRoutingError,
    MissingCustomInputError,
    ToolExecutionError,
    VectorSearchError,
)
from .message_formatting import (
    convert_to_chat_completion_format,
    extract_response_text,
    extract_user_query,
    prepare_messages_for_llm,
)
from .tool_injection import ToolParameterInjector
from .trace_utils import (
    compute_request_preview,
    compute_response_preview,
    create_request_structure,
    create_response_structure,
    patch_trace_info,
    update_trace_preview,
)

__all__ = [
    # Trace utilities
    "compute_request_preview",
    "compute_response_preview",
    "patch_trace_info",
    "update_trace_preview",
    "create_request_structure",
    "create_response_structure",
    # Tool injection
    "ToolParameterInjector",
    # Message formatting
    "convert_to_chat_completion_format",
    "prepare_messages_for_llm",
    "extract_response_text",
    "extract_user_query",
    # Exceptions
    "MissingCustomInputError",
    "AgentConfigurationError",
    "ToolExecutionError",
    "AgentRoutingError",
    "VectorSearchError",
]
