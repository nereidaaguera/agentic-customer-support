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

        if not settings.has_auth:
            logger.warning(
                "No Databricks authentication configured. "
                "Set DATABRICKS_TOKEN for local development or ensure app has OAuth credentials."
            )
            self.client = None
            self.access_token = None
        else:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.request_timeout),
            )
            self.access_token = None
            logger.info(
                f"Initialized agent service with {settings.auth_method} authentication"
            )

    async def _get_oauth_token(self) -> str:
        """Get OAuth access token using client credentials flow."""
        if (
            not self.settings.databricks_client_id
            or not self.settings.databricks_client_secret
        ):
            raise ValueError("OAuth credentials not available")

        workspace_host = self.settings.databricks_host
        token_url = f"{workspace_host}/oidc/v1/token"

        logger.info(f"Getting OAuth token from: {token_url}")

        data = {"grant_type": "client_credentials", "scope": "all-apis"}

        auth = (
            self.settings.databricks_client_id,
            self.settings.databricks_client_secret,
        )

        response = await self.client.post(
            token_url,
            data=data,
            auth=auth,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            logger.error(
                f"OAuth token request failed: {response.status_code} - {response.text}"
            )
            raise ValueError(f"Failed to get OAuth token: {response.status_code}")

        token_data = response.json()
        access_token = token_data.get("access_token")

        if not access_token:
            logger.error(f"No access token in response: {token_data}")
            raise ValueError("No access token received from OAuth endpoint")

        logger.info("Successfully obtained OAuth access token")
        return access_token

    async def _get_headers(self) -> dict[str, str]:
        """Get headers with proper authentication."""
        headers = {"Content-Type": "application/json"}

        if self.settings.auth_method == "oauth":
            if not self.access_token:
                try:
                    self.access_token = await self._get_oauth_token()
                except Exception as e:
                    logger.error(f"Failed to get OAuth token: {e}")
                    logger.info("Falling back to direct client secret usage")
                    headers["Authorization"] = (
                        f"Bearer {self.settings.databricks_client_secret}"
                    )
                    return headers

            headers["Authorization"] = f"Bearer {self.access_token}"

        elif self.settings.auth_method == "token":
            headers["Authorization"] = f"Bearer {self.settings.databricks_token}"

        return headers

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    def _build_databricks_payload(
        self, message: str, customer_id: str, conversation_history: list[ChatMessage]
    ) -> dict[str, Any]:
        """Build the payload for Databricks API."""
        input_messages = []
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
            # extract main response text
            response_text = ""
            agent_type = None
            tools_used = []
            execution_steps = []

            # parse output array to get execution details
            output = databricks_response.get("output", [])

            for item in output:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    # extract text content from message
                    content = item.get("content", [])
                    for content_item in content:
                        if content_item.get("type") == "output_text":
                            response_text += content_item.get("text", "")

                elif item.get("type") == "function_call":
                    # capture function call details
                    function_name = item.get("name", "unknown_function")
                    function_args = item.get("arguments", "{}")
                    call_id = item.get("call_id", "")

                    try:
                        if isinstance(function_args, str):
                            parsed_args = json.loads(function_args)
                        else:
                            parsed_args = function_args
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {"raw_arguments": function_args}

                    tool_call = {
                        "name": function_name,
                        "arguments": parsed_args,
                        "call_id": call_id,
                        "type": "function_call",
                    }

                    tools_used.append(tool_call)

                    # add to execution steps for detailed view
                    execution_steps.append(
                        {
                            "step_type": "tool_call",
                            "tool_name": function_name,
                            "description": f"Calling {function_name}",
                            "arguments": parsed_args,
                            "reasoning": f"Using {function_name} to retrieve relevant information",
                        }
                    )

                elif item.get("type") == "function_call_output":
                    # capture function results
                    call_id = item.get("call_id", "")
                    output_data = item.get("output", "")

                    try:
                        if isinstance(output_data, str) and output_data.startswith("{"):
                            parsed_output = json.loads(output_data)
                        else:
                            parsed_output = output_data
                    except (json.JSONDecodeError, TypeError):
                        parsed_output = output_data

                    # add result to execution steps
                    execution_steps.append(
                        {
                            "step_type": "tool_result",
                            "call_id": call_id,
                            "description": "Tool execution completed",
                            "result": parsed_output,
                            "reasoning": "Retrieved relevant information to answer the query",
                        }
                    )

            # extract custom outputs for agent routing info
            custom_outputs = databricks_response.get("custom_outputs", {})
            routing_info = custom_outputs.get("routing", {})
            if routing_info:
                agent_type = routing_info.get("agent_type")

            # add routing to execution steps
            if agent_type:
                execution_steps.insert(
                    0,
                    {
                        "step_type": "routing",
                        "description": f"Query routed to {agent_type} agent",
                        "reasoning": f"This query is best handled by the {agent_type} specialist",
                    },
                )

            return AgentResponse(
                response=response_text
                or "I apologize, but I couldn't generate a proper response. Please try again.",
                agent_type=agent_type,
                custom_outputs={**custom_outputs, "execution_steps": execution_steps},
                tools_used=tools_used,
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

        if not self.client:
            return AgentResponse(
                response=(
                    "Chat functionality is currently unavailable. "
                    "Please configure Databricks authentication to enable agent responses. "
                    "The demo customer list and UI are still available for testing."
                ),
                agent_type="error",
                custom_outputs={"error": "no_authentication"},
                tools_used=None,
            )

        try:
            payload = self._build_databricks_payload(
                message, customer_id, conversation_history
            )

            logger.info(f"Sending request to Databricks for customer {customer_id}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            headers = await self._get_headers()

            response = await self.client.post(
                self.settings.databricks_endpoint, json=payload, headers=headers
            )

            response.raise_for_status()
            response_data = response.json()

            logger.debug(f"Databricks response: {json.dumps(response_data, indent=2)}")

            return self._parse_agent_response(response_data)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from Databricks: {e.response.status_code} - {e.response.text}"
            )

            if e.response.status_code == 403 and self.settings.auth_method == "oauth":
                logger.info("Got 403, attempting to refresh OAuth token and retry...")
                try:
                    self.access_token = await self._get_oauth_token()
                    headers = await self._get_headers()

                    response = await self.client.post(
                        self.settings.databricks_endpoint, json=payload, headers=headers
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    return self._parse_agent_response(response_data)

                except Exception as retry_error:
                    logger.error(f"Retry after token refresh failed: {retry_error}")

            return AgentResponse(
                response=f"Service temporarily unavailable (HTTP {e.response.status_code}). Please try again.",
                agent_type="error",
                custom_outputs={
                    "error": "http_error",
                    "status_code": e.response.status_code,
                },
                tools_used=None,
            )

        except httpx.RequestError as e:
            logger.error(f"Request error to Databricks: {e}")
            return AgentResponse(
                response="Unable to connect to agent service. Please check your connection and try again.",
                agent_type="error",
                custom_outputs={"error": "connection_error"},
                tools_used=None,
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return AgentResponse(
                response=f"An unexpected error occurred: {str(e)}",
                agent_type="error",
                custom_outputs={"error": "unexpected_error"},
                tools_used=None,
            )

    async def send_message_stream(
        self,
        message: str,
        customer_id: str,
        conversation_history: list[ChatMessage] = None,
    ) -> AsyncGenerator[str, None]:
        """Send a message with streaming response."""
        if conversation_history is None:
            conversation_history = []

        try:
            agent_response = await self.send_message(
                message, customer_id, conversation_history
            )

            response_text = agent_response.response
            chunk_size = 10  # characters per chunk

            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i : i + chunk_size]
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'done': True, 'agent_type': agent_response.agent_type, 'tools_used': agent_response.tools_used})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    async def health_check(self) -> bool:
        """Check if the Databricks endpoint is healthy."""
        if not self.client:
            return False

        try:
            test_payload = self._build_databricks_payload("Hello", "CUS-10001", [])
            headers = await self._get_headers()

            response = await self.client.post(
                self.settings.databricks_endpoint, json=test_payload, headers=headers
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
