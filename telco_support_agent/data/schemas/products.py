"""Product data schemas.

Pydantic models for plans, devices, and promotions data validation.
"""

import re
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Plan(BaseModel):
    """Schema for plan data."""

    plan_id: str = Field(..., pattern=r"^PLAN-\d{4}$")
    plan_name: str
    plan_type: str = Field(
        ..., description="Type of plan", examples=["Individual", "Family", "Business"]
    )
    monthly_price: float = Field(..., ge=0)
    data_limit_gb: int = Field(
        ..., description="Data limit in GB, 0 means unlimited", ge=0
    )
    unlimited_calls: bool
    unlimited_texts: bool
    contract_required: bool
    description: str
    is_active: bool

    @field_validator("plan_id")
    def validate_plan_id(cls, v: str) -> str:
        """Validate the plan ID format."""
        if not re.match(r"^PLAN-\d{4}$", v):
            raise ValueError("Plan ID must be in format PLAN-XXXX where X is a digit")
        return v

    @field_validator("plan_type")
    def validate_plan_type(cls, v: str) -> str:
        """Validate the plan type."""
        valid_types = ["Individual", "Family", "Business"]
        if v not in valid_types:
            raise ValueError(f"Plan type must be one of {valid_types}")
        return v


class Device(BaseModel):
    """Schema for device data."""

    device_id: str = Field(..., pattern=r"^DEV-\d{4}$")
    device_name: str
    manufacturer: str
    device_type: str = Field(
        ..., description="Type of device", examples=["Smartphone", "Tablet", "Hotspot"]
    )
    retail_price: float = Field(..., ge=0)
    monthly_installment: float = Field(..., ge=0)
    storage_gb: int = Field(..., ge=0)
    color_options: str  # Comma-separated list of colors
    release_date: date
    is_5g_compatible: bool
    is_active: bool

    @field_validator("device_id")
    def validate_device_id(cls, v: str) -> str:
        """Validate the device ID format."""
        if not re.match(r"^DEV-\d{4}$", v):
            raise ValueError("Device ID must be in format DEV-XXXX where X is a digit")
        return v

    @field_validator("device_type")
    def validate_device_type(cls, v: str) -> str:
        """Validate the device type."""
        valid_types = ["Smartphone", "Tablet", "Hotspot"]
        if v not in valid_types:
            raise ValueError(f"Device type must be one of {valid_types}")
        return v

    @field_validator("color_options")
    def validate_color_options(cls, v: str) -> str:
        """Validate the color options."""
        # Ensure comma-separated format
        colors = [c.strip() for c in v.split(",")]
        return ", ".join(colors)


class Promotion(BaseModel):
    """Schema for promotion data."""

    promo_id: str = Field(..., pattern=r"^PROMO-\d{4}$")
    promo_name: str
    discount_type: str = Field(
        ..., description="Type of discount", examples=["Percentage", "Fixed", "Service"]
    )
    discount_value: float = Field(..., ge=0)
    start_date: date
    end_date: date
    description: str
    is_active: bool

    @field_validator("promo_id")
    def validate_promo_id(cls, v: str) -> str:
        """Validate the promotion ID format."""
        if not re.match(r"^PROMO-\d{4}$", v):
            raise ValueError(
                "Promotion ID must be in format PROMO-XXXX where X is a digit"
            )
        return v

    @field_validator("discount_type")
    def validate_discount_type(cls, v: str) -> str:
        """Validate the discount type."""
        valid_types = ["Percentage", "Fixed", "Service"]
        if v not in valid_types:
            raise ValueError(f"Discount type must be one of {valid_types}")
        return v

    @field_validator("end_date")
    def validate_end_date(cls, v: date, info: Any) -> date:
        """Validate the end date is after the start date."""
        # Access start_date from the input data
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("End date must be after start date")
        return v
