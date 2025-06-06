"""Message formatting and conversion utilities."""

from typing import Any


def convert_to_chat_completion_format(
    message: dict[str, Any], llm_endpoint: str = ""
) -> list[dict[str, Any]]:
    """Convert from Responses API to ChatCompletion compatible.

    Args:
        message: Message in Responses API format
        llm_endpoint: LLM endpoint name for special handling

    Returns:
        List of messages in ChatCompletion format
    """
    msg_type = message.get("type", None)
    if msg_type == "function_call":
        return [
            {
                "role": "assistant",
                "content": None
                if llm_endpoint == "databricks-claude-3-7-sonnet"
                else "tool call",
                "tool_calls": [
                    {
                        "id": message["call_id"],
                        "type": "function",
                        "function": {
                            "arguments": message["arguments"],
                            "name": message["name"],
                        },
                    }
                ],
            }
        ]
    elif msg_type == "message" and isinstance(message["content"], list):
        return [
            {"role": message["role"], "content": content["text"]}
            for content in message["content"]
        ]
    elif msg_type == "function_call_output":
        return [
            {
                "role": "tool",
                "content": message["output"],
                "tool_call_id": message["call_id"],
            }
        ]
    compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
    return [{k: v for k, v in message.items() if k in compatible_keys}]


def prepare_messages_for_llm(
    messages: list[dict[str, Any]], llm_endpoint: str = ""
) -> list[dict[str, Any]]:
    """Filter out message fields that are not compatible with LLM message formats.

    Converts from Responses API to ChatCompletion compatible format.

    Args:
        messages: List of messages in Responses API format
        llm_endpoint: LLM endpoint name for special handling

    Returns:
        List of messages in ChatCompletion format
    """
    chat_msgs = []
    for msg in messages:
        chat_msgs.extend(convert_to_chat_completion_format(msg, llm_endpoint))
    return chat_msgs


def extract_response_text(
    output_items: list[dict[str, Any]], max_length: int = 200
) -> str:
    """Extract text from response output items.

    Args:
        output_items: List of output items from agent response
        max_length: Maximum length of extracted text

    Returns:
        Extracted text, truncated to max_length
    """
    for item in output_items:
        if isinstance(item, dict) and item.get("type") == "message":
            content = item.get("content", [])
            for content_item in content:
                if content_item.get("type") == "output_text":
                    text = content_item.get("text", "")
                    return text[:max_length] if text else ""
    return "No text response found"


def extract_user_query(input_messages: list[Any]) -> str:
    """Extract the most recent user query from input messages.

    Args:
        input_messages: List of input messages

    Returns:
        Most recent user query content
    """
    for msg in reversed(input_messages):
        if hasattr(msg, "role") and msg.role == "user":
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""
