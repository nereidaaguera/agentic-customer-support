"""Billing data schemas.

Pydantic models for billing and usage data validation.
"""

import re
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Billing(BaseModel):
    """Schema for billing data."""

    billing_id: str = Field(..., pattern=r"^BILL-\d{10}$")
    subscription_id: str = Field(..., pattern=r"^SUB-\d{8}$")
    customer_id: str = Field(..., pattern=r"^CUS-\d{5}$")
    billing_date: date
    due_date: date
    base_amount: float = Field(..., ge=0)
    additional_charges: float = Field(..., ge=0)
    tax_amount: float = Field(..., ge=0)
    total_amount: float = Field(..., ge=0)
    payment_amount: float | None = Field(None, ge=0)
    payment_date: date | None = None
    payment_method: str | None = Field(
        None,
        description="Method of payment",
        examples=["Credit Card", "Bank Transfer", "PayPal", "Check"],
    )
    status: str = Field(
        ...,
        description="Payment status",
        examples=["Paid", "Unpaid", "Late", "Partial"],
    )
    billing_cycle: str = Field(..., pattern=r"^\d{4}-\d{2}$")

    @field_validator("billing_id")
    def validate_billing_id(cls, v: str) -> str:
        """Validate the billing ID format."""
        if not re.match(r"^BILL-\d{10}$", v):
            raise ValueError(
                "Billing ID must be in format BILL-XXXXXXXXXX where X is a digit"
            )
        return v

    @field_validator("due_date")
    def validate_due_date(cls, v: date, info: Any) -> date:
        """Validate the due date is after the billing date."""
        if "billing_date" in info.data and v < info.data["billing_date"]:
            raise ValueError("Due date must be after billing date")
        return v

    @model_validator(mode="after")
    def validate_total_amount(self) -> "Billing":
        """Validate the total amount matches the sum of components."""
        expected_total = self.base_amount + self.additional_charges + self.tax_amount

        # allow for small floating-point differences
        if abs(self.total_amount - expected_total) > 0.01:
            raise ValueError(
                f"Total amount ({self.total_amount}) should equal the sum of base amount, "
                f"additional charges, and tax amount ({expected_total})"
            )
        return self

    @field_validator("payment_method")
    def validate_payment_method(cls, v: str | None, info: Any) -> str | None:
        """Validate the payment method if payment was made."""
        if (
            "payment_amount" in info.data
            and info.data["payment_amount"] is not None
            and v is None
        ):
            raise ValueError("Payment method must be provided if payment amount is set")

        if v is not None:
            valid_methods = ["Credit Card", "Bank Transfer", "PayPal", "Check"]
            if v not in valid_methods:
                raise ValueError(f"Payment method must be one of {valid_methods}")

        return v

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate the billing status."""
        valid_statuses = ["Paid", "Unpaid", "Late", "Partial"]
        if v not in valid_statuses:
            raise ValueError(f"Billing status must be one of {valid_statuses}")
        return v

    @field_validator("billing_cycle")
    def validate_billing_cycle(cls, v: str) -> str:
        """Validate the billing cycle format (YYYY-MM)."""
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError("Billing cycle must be in format YYYY-MM")
        return v


class Usage(BaseModel):
    """Schema for usage data."""

    usage_id: str = Field(..., pattern=r"^USG-\d{12}$")
    subscription_id: str = Field(..., pattern=r"^SUB-\d{8}$")
    date: date
    data_usage_mb: float = Field(..., ge=0)
    voice_minutes: float = Field(..., ge=0)
    sms_count: int = Field(..., ge=0)
    billing_cycle: str = Field(..., pattern=r"^\d{4}-\d{2}$")

    @field_validator("usage_id")
    def validate_usage_id(cls, v: str) -> str:
        """Validate the usage ID format."""
        if not re.match(r"^USG-\d{12}$", v):
            raise ValueError(
                "Usage ID must be in format USG-XXXXXXXXXXXX where X is a digit"
            )
        return v

    @field_validator("billing_cycle")
    def validate_billing_cycle(cls, v: str) -> str:
        """Validate the billing cycle format (YYYY-MM)."""
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError("Billing cycle must be in format YYYY-MM")
        return v
