"""API routes for telco support agent."""

import json
import logging
import traceback
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..services.telco_agent_service import TelcoAgentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., description="User message")
    customer_id: str = Field(..., description="Customer ID (e.g., CUS-10001)")
    conversation_history: list[ChatMessage] = Field(
        default_factory=list, description="Previous conversation messages"
    )


class AgentResponse(BaseModel):
    """Agent response model."""

    response: str = Field(..., description="Agent response text")
    agent_type: Optional[str] = Field(
        None, description="Which agent handled the request"
    )
    custom_outputs: Optional[dict] = Field(None, description="Additional agent outputs")
    tools_used: Optional[list[dict]] = Field(
        None, description="Tools/functions called by agent"
    )


class CustomerInfo(BaseModel):
    """Customer information model."""

    customer_id: str
    display_name: str


def get_agent_service(settings: Settings = Depends(get_settings)) -> TelcoAgentService:
    """Dependency to get agent service instance."""
    return TelcoAgentService(settings)


@router.get("/customers", response_model=list[CustomerInfo])
async def get_demo_customers(settings: Settings = Depends(get_settings)):
    """Get list of demo customer IDs."""
    try:
        customers = []
        for customer_id in settings.demo_customer_ids:
            # extract number from customer ID for display
            number = customer_id.replace("CUS-", "")
            customers.append(
                CustomerInfo(customer_id=customer_id, display_name=f"Customer {number}")
            )

        logger.info(f"Retrieved {len(customers)} demo customers")
        return customers

    except Exception as e:
        logger.error(f"Error getting demo customers: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error retrieving customers: {str(e)}"
        ) from e


@router.post("/chat", response_model=AgentResponse)
async def chat(
    request: ChatRequest, agent_service: TelcoAgentService = Depends(get_agent_service)
):
    """Send message to telco support agent (non-streaming)."""
    try:
        logger.info(
            f"Received chat request for customer {request.customer_id}: {request.message}"
        )

        response = await agent_service.send_message(
            message=request.message,
            customer_id=request.customer_id,
            conversation_history=request.conversation_history,
        )

        logger.info(f"Agent response tools_used: {response.tools_used}")
        logger.info(
            f"Agent response custom_outputs keys: {list(response.custom_outputs.keys()) if response.custom_outputs else 'None'}"
        )

        logger.info(
            f"Successfully processed chat request for customer {request.customer_id}"
        )
        return response

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        error_detail = f"Chat processing error: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail) from e


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, agent_service: TelcoAgentService = Depends(get_agent_service)
):
    """Send message to telco support agent with streaming response."""
    try:
        logger.info(
            f"Received streaming chat request for customer {request.customer_id}: {request.message}"
        )

        async def event_generator():
            """Generator that yields Server-Sent Events."""
            try:
                async for event_data in agent_service.send_message_stream(
                    message=request.message,
                    customer_id=request.customer_id,
                    conversation_history=request.conversation_history,
                ):
                    yield event_data

            except Exception as e:
                logger.error(f"Error in streaming generator: {str(e)}")
                logger.error(traceback.format_exc())

                # Send error event
                error_event = {
                    "type": "error",
                    "error": f"Streaming error: {str(e)}",
                    "done": True,
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    except Exception as e:
        logger.error(f"Error setting up streaming chat request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error setting up streaming request: {str(e)}"
        ) from e


@router.get("/health")
async def agent_health(agent_service: TelcoAgentService = Depends(get_agent_service)):
    """Check Databricks agent endpoint is healthy."""
    try:
        logger.info("Checking agent health...")
        is_healthy = await agent_service.health_check()

        health_status = {
            "status": "healthy" if is_healthy else "unhealthy",
            "endpoint": agent_service.settings.databricks_endpoint,
        }

        logger.info(f"Health check result: {health_status}")
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())

        return {
            "status": "unhealthy",
            "endpoint": "unknown",
            "error": str(e),
        }


@router.get("/debug")
async def debug_info(settings: Settings = Depends(get_settings)):
    """Debug endpoint to check configuration."""
    try:
        return {
            "environment": settings.environment,
            "port": settings.port,
            "databricks_host": settings.databricks_host,
            "databricks_endpoint_name": settings.databricks_endpoint_name,
            "databricks_endpoint": settings.databricks_endpoint,
            "has_token": bool(settings.databricks_token),
            "token_preview": settings.databricks_token[:10] + "..."
            if settings.databricks_token
            else "Not set",
            "demo_customers": settings.demo_customer_ids,
        }
    except Exception as e:
        logger.error(f"Debug info error: {str(e)}")
        return {"error": str(e)}
