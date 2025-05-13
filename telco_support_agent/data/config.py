"""Config for data generation process.

This module contains Config parameters used throughout the data generation
process, including volumes, distributions, and date ranges.
"""

from typing import TypedDict


class PriceTierDict(TypedDict):
    """Type for price tier structure."""

    Basic: float
    Standard: float
    Premium: float
    Unlimited: float


class PlanPriceDict(TypedDict):
    """Type for plan price structure."""

    Individual: PriceTierDict
    Family: PriceTierDict
    Business: PriceTierDict


class PlanConfigDict(TypedDict):
    """Type for plan Config structure."""

    data_tiers: list[int]
    price_tiers: PlanPriceDict


class DevicePriceRangeDict(TypedDict):
    """Type for device price range structure."""

    Smartphone: list[float]
    Tablet: list[float]
    Hotspot: list[float]


class DeviceConfigDict(TypedDict):
    """Type for device Config structure."""

    manufacturers: list[str]
    storage_options: list[int]
    price_ranges: DevicePriceRangeDict


class DiscountRangeDict(TypedDict):
    """Type for discount range structure."""

    Percentage: list[int]
    Fixed: list[int]
    Service: list[int]


class PromotionConfigDict(TypedDict):
    """Type for promotion Config structure."""

    discount_types: dict[str, float]
    discount_ranges: DiscountRangeDict


class ProductConfigDict(TypedDict):
    """Type for product Config structure."""

    plans: PlanConfigDict
    devices: DeviceConfigDict
    promotions: PromotionConfigDict


class DistributionsDict(TypedDict):
    """Type for distributions structure."""

    customer_segments: dict[str, float]
    plan_types: dict[str, float]
    device_types: dict[str, float]
    payment_statuses: dict[str, float]
    subscription_statuses: dict[str, float]
    customer_statuses: dict[str, float]
    contact_methods: dict[str, float]
    kb_content_types: dict[str, float]
    kb_categories: dict[str, float]
    ticket_categories: dict[str, float]
    ticket_priorities: dict[str, float]
    ticket_statuses: dict[str, float]


class VolumesDict(TypedDict):
    """Type for volumes structure."""

    customers: int
    plans: int
    devices: int
    promotions: int
    kb_articles: int
    tickets: int


class DateRangesDict(TypedDict):
    """Type for date ranges structure."""

    customer_registration: list[str]
    billing_cycles: list[str]
    device_release: list[str]


class ConfigDict(TypedDict):
    """Type for the main Config dictionary."""

    seed: int
    volumes: VolumesDict
    date_ranges: DateRangesDict
    distributions: DistributionsDict
    products: ProductConfigDict


# Config dictionary for data generation
CONFIG: ConfigDict = {
    # Random seed for reproducibility
    "seed": 42,
    # Data volumes
    "volumes": {
        "customers": 1000,
        "plans": 10,
        "devices": 20,
        "promotions": 5,
        "kb_articles": 200,
        "tickets": 500,
    },
    # Date ranges
    "date_ranges": {
        "customer_registration": ["2020-01-01", "2025-03-01"],
        "billing_cycles": ["2024-01", "2025-03"],
        "device_release": ["2019-01-01", "2025-01-01"],
    },
    # Distributions for categorical variables
    "distributions": {
        # Customer segments
        "customer_segments": {
            "Individual": 0.6,
            "Family": 0.25,
            "Business": 0.05,
            "Premium": 0.05,
            "Student": 0.05,
        },
        # Plan types
        "plan_types": {"Individual": 0.5, "Family": 0.3, "Business": 0.2},
        # Device types
        "device_types": {"Smartphone": 0.7, "Tablet": 0.2, "Hotspot": 0.1},
        # Payment statuses
        "payment_statuses": {
            "Paid": 0.85,
            "Unpaid": 0.1,
            "Late": 0.04,
            "Partial": 0.01,
        },
        # Subscription statuses
        "subscription_statuses": {"Active": 0.9, "Paused": 0.05, "Cancelled": 0.05},
        # Customer statuses
        "customer_statuses": {"Active": 0.9, "Inactive": 0.05, "Suspended": 0.05},
        # Preferred contact methods
        "contact_methods": {"Email": 0.5, "Phone": 0.3, "SMS": 0.2},
        # Knowledge base content types
        "kb_content_types": {"FAQ": 0.5, "Policy": 0.2, "Guide": 0.2, "Procedure": 0.1},
        # Knowledge base categories
        "kb_categories": {
            "Billing": 0.4,
            "Technical": 0.4,
            "Account": 0.15,
            "Services": 0.05,
        },
        # Ticket categories
        "ticket_categories": {
            "Billing": 0.4,
            "Technical": 0.4,
            "Account": 0.15,
            "Services": 0.05,
        },
        # Ticket priorities
        "ticket_priorities": {
            "Low": 0.4,
            "Medium": 0.4,
            "High": 0.15,
            "Critical": 0.05,
        },
        # Ticket statuses
        "ticket_statuses": {
            "Open": 0.2,
            "In Progress": 0.3,
            "Resolved": 0.3,
            "Closed": 0.2,
        },
    },
    # Product-specific Configs
    "products": {
        # Plan Configs
        "plans": {
            "data_tiers": [2, 5, 10, 20, 50, 100, 0],  # 0 means unlimited
            "price_tiers": {
                "Individual": {
                    "Basic": 29.99,
                    "Standard": 49.99,
                    "Premium": 69.99,
                    "Unlimited": 99.99,
                },
                "Family": {
                    "Basic": 59.99,
                    "Standard": 89.99,
                    "Premium": 119.99,
                    "Unlimited": 159.99,
                },
                "Business": {
                    "Basic": 79.99,
                    "Standard": 129.99,
                    "Premium": 179.99,
                    "Unlimited": 229.99,
                },
            },
        },
        # Device Configs
        "devices": {
            "manufacturers": ["Apple", "Samsung", "Google", "Motorola", "OnePlus"],
            "storage_options": [64, 128, 256, 512, 1024],
            "price_ranges": {
                "Smartphone": [399.99, 1599.99],
                "Tablet": [299.99, 1299.99],
                "Hotspot": [99.99, 399.99],
            },
        },
        # Promotion Configs
        "promotions": {
            "discount_types": {"Percentage": 0.4, "Fixed": 0.4, "Service": 0.2},
            "discount_ranges": {
                "Percentage": [10, 50],  # percentage off
                "Fixed": [5, 100],  # dollars off
                "Service": [
                    0,
                    0,
                ],  # placeholder, service promos don't have direct value
            },
        },
    },
}
