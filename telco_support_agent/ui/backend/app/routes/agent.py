"""API routes for telco support agent."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..services.telco_agent_service import TelcoAgentService

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
    customers = []
    for customer_id in settings.demo_customer_ids:
        # extract number from customer ID for display
        number = customer_id.replace("CUS-", "")
        customers.append(
            CustomerInfo(customer_id=customer_id, display_name=f"Customer {number}")
        )

    return customers


@router.post("/chat", response_model=AgentResponse)
async def chat(
    request: ChatRequest, agent_service: TelcoAgentService = Depends(get_agent_service)
):
    """Send message to telco support agent."""
    try:
        response = await agent_service.send_message(
            message=request.message,
            customer_id=request.customer_id,
            conversation_history=request.conversation_history,
        )
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        ) from e


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, agent_service: TelcoAgentService = Depends(get_agent_service)
):
    """Send message to telco support agent with streaming response."""
    try:
        return StreamingResponse(
            agent_service.send_message_stream(
                message=request.message,
                customer_id=request.customer_id,
                conversation_history=request.conversation_history,
            ),
            media_type="text/plain",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing streaming request: {str(e)}"
        ) from e


@router.get("/health")
async def agent_health(agent_service: TelcoAgentService = Depends(get_agent_service)):
    """Check Databricks agent endpoint is healthy."""
    try:
        is_healthy = await agent_service.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "endpoint": agent_service.settings.databricks_endpoint,
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "endpoint": agent_service.settings.databricks_endpoint,
            "error": str(e),
        }
