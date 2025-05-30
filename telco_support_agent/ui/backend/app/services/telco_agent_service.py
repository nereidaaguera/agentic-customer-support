"""Service for interacting with the Databricks Telco Support Agent."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from ..config import Settings

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str
    content: str


class AgentResponse(BaseModel):
    """Agent response model."""

    response: str
    agent_type: Optional[str] = None
    custom_outputs: Optional[dict] = None
    tools_used: Optional[list[dict]] = None


class TelcoAgentService:
    """Service for interacting with Databricks Telco Support Agent."""

    def __init__(self, settings: Settings):
        """Initialize the service."""
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            headers=settings.databricks_headers,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    def _build_databricks_payload(
        self, message: str, customer_id: str, conversation_history: list[ChatMessage]
    ) -> dict[str, Any]:
        """Build the payload for Databricks API."""
        # Convert conversation history to the expected format
        input_messages = []

        # Add conversation history
        for msg in conversation_history:
            input_messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        input_messages.append({"role": "user", "content": message})

        return {"input": input_messages, "custom_inputs": {"customer": customer_id}}

    def _parse_agent_response(
        self, databricks_response: dict[str, Any]
    ) -> AgentResponse:
        """Parse the response from Databricks into our format."""
        try:
            # Extract the main response text
            response_text = ""
            agent_type = None
            tools_used = []

            # Parse the output array
            output = databricks_response.get("output", [])

            for item in output:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    # Extract text content from message
                    content = item.get("content", [])
                    for content_item in content:
                        if content_item.get("type") == "output_text":
                            response_text += content_item.get("text", "")

                elif item.get("type") == "function_call":
                    # Track tool usage
                    tools_used.append(
                        {"name": item.get("name"), "arguments": item.get("arguments")}
                    )

            # Extract custom outputs for agent routing info
            custom_outputs = databricks_response.get("custom_outputs", {})
            routing_info = custom_outputs.get("routing", {})
            if routing_info:
                agent_type = routing_info.get("agent_type")

            return AgentResponse(
                response=response_text
                or "I apologize, but I couldn't generate a proper response. Please try again.",
                agent_type=agent_type,
                custom_outputs=custom_outputs,
                tools_used=tools_used if tools_used else None,
            )

        except Exception as e:
            logger.error(f"Error parsing agent response: {e}")
            return AgentResponse(
                response="I encountered an error processing your request. Please try again.",
                agent_type=None,
                custom_outputs=None,
                tools_used=None,
            )

    async def send_message(
        self,
        message: str,
        customer_id: str,
        conversation_history: list[ChatMessage] = None,
    ) -> AgentResponse:
        """Send a message to the telco support agent."""
        if conversation_history is None:
            conversation_history = []

        try:
            # Build the request payload
            payload = self._build_databricks_payload(
                message, customer_id, conversation_history
            )

            logger.info(f"Sending request to Databricks for customer {customer_id}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            # Make the request
            response = await self.client.post(
                self.settings.databricks_endpoint, json=payload
            )

            response.raise_for_status()
            response_data = response.json()

            logger.debug(f"Databricks response: {json.dumps(response_data, indent=2)}")

            # Parse and return the response
            return self._parse_agent_response(response_data)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from Databricks: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(f"Agent service error: {e.response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"Request error to Databricks: {e}")
            raise Exception("Unable to connect to agent service")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Agent processing error: {str(e)}")

    async def send_message_stream(
        self,
        message: str,
        customer_id: str,
        conversation_history: list[ChatMessage] = None,
    ) -> AsyncGenerator[str, None]:
        """Send a message with streaming response (if supported by endpoint)."""
        if conversation_history is None:
            conversation_history = []

        try:
            # For now, we'll simulate streaming by sending the regular response
            # If your Databricks endpoint supports streaming, you can modify this
            agent_response = await self.send_message(
                message, customer_id, conversation_history
            )

            # Simulate streaming by yielding chunks
            response_text = agent_response.response
            chunk_size = 10  # characters per chunk

            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i : i + chunk_size]
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                # Add small delay to simulate streaming
                await asyncio.sleep(0.1)

            # Send final metadata
            yield f"data: {json.dumps({'done': True, 'agent_type': agent_response.agent_type, 'tools_used': agent_response.tools_used})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    async def health_check(self) -> bool:
        """Check if the Databricks endpoint is healthy."""
        try:
            # Send a simple test request
            test_payload = self._build_databricks_payload("Hello", "CUS-10001", [])

            response = await self.client.post(
                self.settings.databricks_endpoint, json=test_payload
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
