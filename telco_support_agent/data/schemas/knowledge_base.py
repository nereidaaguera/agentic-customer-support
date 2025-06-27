"""Knowledge base data schemas.

Pydantic models for knowledge base and support ticket data validation.
"""

import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class KnowledgeBase(BaseModel):
    """Schema for knowledge base data."""

    kb_id: str = Field(..., pattern=r"^KB-\d{4}$")
    content_type: str = Field(
        ...,
        description="Type of content",
        examples=["FAQ", "Policy", "Guide", "Procedure"],
    )
    category: str = Field(
        ...,
        description="Main category",
        examples=["Billing", "Technical", "Account", "Services"],
    )
    subcategory: str
    title: str
    content: str  # Full content in markdown format
    tags: str  # Comma-separated tags
    last_updated: date

    @field_validator("kb_id")
    def validate_kb_id(cls, v: str) -> str:
        """Validate the knowledge base ID format."""
        if not re.match(r"^KB-\d{4}$", v):
            raise ValueError(
                "Knowledge base ID must be in format KB-XXXX where X is a digit"
            )
        return v

    @field_validator("content_type")
    def validate_content_type(cls, v: str) -> str:
        """Validate the content type."""
        valid_types = ["FAQ", "Policy", "Guide", "Procedure"]
        if v not in valid_types:
            raise ValueError(f"Content type must be one of {valid_types}")
        return v

    @field_validator("category")
    def validate_category(cls, v: str) -> str:
        """Validate the category."""
        valid_categories = ["Billing", "Technical", "Account", "Services"]
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v

    @field_validator("tags")
    def validate_tags(cls, v: str) -> str:
        """Validate and normalize tags."""
        # Normalize the tags by removing extra spaces
        tags = [tag.strip() for tag in v.split(",")]
        return ",".join(tags)


class SupportTicket(BaseModel):
    """Schema for support ticket data."""

    ticket_id: str = Field(..., pattern=r"^TICK-\d{4}$")
    customer_id: str = Field(..., pattern=r"^CUS-\d{5}$")
    subscription_id: str = Field(..., pattern=r"^SUB-\d{8}$")
    created_date: datetime
    status: str = Field(
        ...,
        description="Ticket status",
        examples=["Open", "In Progress", "Resolved", "Closed"],
    )
    category: str = Field(
        ...,
        description="Ticket category",
        examples=["Billing", "Technical", "Account", "Services"],
    )
    priority: str = Field(
        ...,
        description="Ticket priority",
        examples=["Low", "Medium", "High", "Critical"],
    )
    description: str
    resolution: str | None = None
    resolved_date: datetime | None = None
    agent_id: str | None = None

    @field_validator("ticket_id")
    def validate_ticket_id(cls, v: str) -> str:
        """Validate the ticket ID format."""
        if not re.match(r"^TICK-\d{4}$", v):
            raise ValueError("Ticket ID must be in format TICK-XXXX where X is a digit")
        return v

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate the ticket status."""
        valid_statuses = ["Open", "In Progress", "Resolved", "Closed"]
        if v not in valid_statuses:
            raise ValueError(f"Ticket status must be one of {valid_statuses}")
        return v

    @field_validator("category")
    def validate_category(cls, v: str) -> str:
        """Validate the ticket category."""
        valid_categories = ["Billing", "Technical", "Account", "Services"]
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v

    @field_validator("priority")
    def validate_priority(cls, v: str) -> str:
        """Validate the ticket priority."""
        valid_priorities = ["Low", "Medium", "High", "Critical"]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of {valid_priorities}")
        return v

    @field_validator("resolved_date")
    def validate_resolved_date(cls, v: datetime | None, info: Any) -> datetime | None:
        """Validate the resolved date is after the created date if present."""
        if (
            v is not None
            and "created_date" in info.data
            and v < info.data["created_date"]
        ):
            raise ValueError("Resolved date must be after created date")
        return v

    @field_validator("resolution")
    def validate_resolution(cls, v: str | None, info: Any) -> str | None:
        """Validate resolution is present if status is Resolved or Closed."""
        if (
            "status" in info.data
            and info.data["status"] in ["Resolved", "Closed"]
            and not v
        ):
            raise ValueError(
                "Resolution must be provided for Resolved or Closed tickets"
            )
        return v
