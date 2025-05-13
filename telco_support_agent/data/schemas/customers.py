"""Customer data schemas.

Defines Pydantic models for customer and subscription data validation.
"""

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator


class Customer(BaseModel):
    """Schema for customer data."""

    customer_id: str = Field(..., pattern=r"^CUS-\d{5}$")
    customer_segment: str = Field(
        ...,
        description="Customer segment category",
        examples=["Individual", "Family", "Business", "Premium", "Student"],
    )
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    registration_date: date
    customer_status: str = Field(
        ...,
        description="Current account status",
        examples=["Active", "Inactive", "Suspended"],
    )
    preferred_contact_method: str = Field(
        ..., description="Preferred contact method", examples=["Email", "Phone", "SMS"]
    )

    @field_validator("customer_id")
    def validate_customer_id(cls, v: str) -> str:
        """Validate the customer ID format."""
        if not re.match(r"^CUS-\d{5}$", v):
            raise ValueError(
                "Customer ID must be in format CUS-XXXXX where X is a digit"
            )
        return v

    @field_validator("customer_segment")
    def validate_customer_segment(cls, v: str) -> str:
        """Validate the customer segment."""
        valid_segments = ["Individual", "Family", "Business", "Premium", "Student"]
        if v not in valid_segments:
            raise ValueError(f"Customer segment must be one of {valid_segments}")
        return v

    @field_validator("customer_status")
    def validate_customer_status(cls, v: str) -> str:
        """Validate the customer status."""
        valid_statuses = ["Active", "Inactive", "Suspended"]
        if v not in valid_statuses:
            raise ValueError(f"Customer status must be one of {valid_statuses}")
        return v

    @field_validator("preferred_contact_method")
    def validate_preferred_contact_method(cls, v: str) -> str:
        """Validate the preferred contact method."""
        valid_methods = ["Email", "Phone", "SMS"]
        if v not in valid_methods:
            raise ValueError(f"Preferred contact method must be one of {valid_methods}")
        return v


class Subscription(BaseModel):
    """Schema for subscription data."""

    subscription_id: str = Field(..., pattern=r"^SUB-\d{8}$")
    customer_id: str = Field(..., pattern=r"^CUS-\d{5}$")
    plan_id: str = Field(..., pattern=r"^PLAN-\d{4}$")
    device_id: str | None = Field(None, pattern=r"^DEV-\d{4}$")
    promo_id: str | None = Field(None, pattern=r"^PROMO-\d{4}$")
    subscription_start_date: date
    contract_length_months: int = Field(..., ge=0, le=36)
    monthly_charge: float = Field(..., ge=0)
    status: str = Field(
        ...,
        description="Subscription status",
        examples=["Active", "Paused", "Cancelled"],
    )
    autopay_enabled: bool

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate the subscription status."""
        valid_statuses = ["Active", "Paused", "Cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Subscription status must be one of {valid_statuses}")
        return v

    @field_validator("contract_length_months")
    def validate_contract_length(cls, v: int) -> int:
        """Validate the contract length is a standard value."""
        valid_lengths = [0, 12, 24, 36]  # 0 means no contract
        if v not in valid_lengths:
            raise ValueError(f"Contract length must be one of {valid_lengths}")
        return v
