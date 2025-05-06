"""Product data generator.

This module contains the ProductGenerator class for generating plans, devices,
and promotions data.
"""
from datetime import datetime, timedelta
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    BooleanType,
    DateType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from telco_support_agent.data.generators.base import BaseGenerator


class ProductGenerator(BaseGenerator):
    """Generator for product data.

    Generates data for plans, devices, and promotions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the product generator.

        Args:
            config: config dictionary.
        """
        super().__init__(config)

    def generate_plans(self) -> DataFrame:
        """Generate plans data.

        Returns:
            DataFrame containing generated plans data.
        """
        # Get config
        count = self.config["volumes"]["plans"]
        plan_types = self.config["distributions"]["plan_types"]
        price_tiers = self.config["products"]["plans"]["price_tiers"]
        data_tiers = self.config["products"]["plans"]["data_tiers"]

        # Generate plan IDs
        plan_ids = self.generate_id("PLAN", 1001, count)

        # Prepare data
        data = []

        for i, plan_id in enumerate(plan_ids):
            # Select plan type based on distribution
            plan_type = self.select_weighted(plan_types)

            # Determine tier (Basic, Standard, Premium, Unlimited)
            # Use deterministic assignment based on index for good distribution
            tier_index = i % 4
            tiers = ["Basic", "Standard", "Premium", "Unlimited"]
            tier = tiers[tier_index]

            # Set name based on type and tier
            plan_name = f"{tier} {plan_type}"

            # Set price based on tier and type
            monthly_price = price_tiers[plan_type][tier]

            # Set data limit
            # Unlimited plans get 0 (unlimited)
            if tier == "Unlimited":
                data_limit_gb = 0
            else:
                # Otherwise select based on tier
                data_tier_index = min(
                    tier_index, len(data_tiers) - 2
                )  # Avoid last index (unlimited)
                data_limit_gb = data_tiers[data_tier_index]

            # Set feature flags
            unlimited_calls = tier in ["Premium", "Unlimited"]
            unlimited_texts = tier in ["Standard", "Premium", "Unlimited"]
            contract_required = tier in ["Premium", "Unlimited"] or plan_type in [
                "Family",
                "Business",
            ]

            # Generate description
            description = f"{plan_name}: "
            if data_limit_gb == 0:
                description += "Unlimited data, "
            else:
                description += f"{data_limit_gb}GB data, "

            if unlimited_calls:
                description += "unlimited calls, "
            else:
                description += "pay-per-minute calls, "

            if unlimited_texts:
                description += "unlimited texts. "
            else:
                description += "pay-per-text. "

            if contract_required:
                description += "Contract required."
            else:
                description += "No contract required."

            # Most plans are active
            is_active = self.random.random() < 0.9

            # Add to data
            data.append(
                (
                    plan_id,
                    plan_name,
                    plan_type,
                    monthly_price,
                    data_limit_gb,
                    unlimited_calls,
                    unlimited_texts,
                    contract_required,
                    description,
                    is_active,
                )
            )

        # Create schema
        schema = StructType(
            [
                StructField("plan_id", StringType(), False),
                StructField("plan_name", StringType(), False),
                StructField("plan_type", StringType(), False),
                StructField("monthly_price", FloatType(), False),
                StructField("data_limit_gb", IntegerType(), False),
                StructField("unlimited_calls", BooleanType(), False),
                StructField("unlimited_texts", BooleanType(), False),
                StructField("contract_required", BooleanType(), False),
                StructField("description", StringType(), False),
                StructField("is_active", BooleanType(), False),
            ]
        )

        # Create DataFrame
        df = self.create_dataframe_from_schema(schema, data)

        return df

    def generate_devices(self) -> DataFrame:
        """Generate devices data.

        Returns:
            DataFrame containing generated devices data.
        """
        # Get config
        count = self.config["volumes"]["devices"]
        device_types = self.config["distributions"]["device_types"]
        manufacturers = self.config["products"]["devices"]["manufacturers"]
        storage_options = self.config["products"]["devices"]["storage_options"]
        price_ranges = self.config["products"]["devices"]["price_ranges"]

        # Generate device IDs
        device_ids = self.generate_id("DEV", 2001, count)

        # Prepare data
        data = []

        for device_id in device_ids:
            # Select device type based on distribution
            device_type = self.select_weighted(device_types)

            # Select manufacturer
            manufacturer = self.random.choice(manufacturers)

            # Generate device name
            if manufacturer == "Apple":
                if device_type == "Smartphone":
                    model = f"iPhone {self.random.randint(12, 15)} {self.random.choice(['', 'Pro', 'Pro Max'])}"
                elif device_type == "Tablet":
                    model = f"iPad {self.random.choice(['', 'Air', 'Pro'])} {self.random.randint(8, 11)}"
                else:  # Hotspot
                    model = f"MiFi {self.random.randint(1, 5)}"
            elif manufacturer == "Samsung":
                if device_type == "Smartphone":
                    model = f"Galaxy S{self.random.randint(20, 26)} {self.random.choice(['', 'Plus', 'Ultra'])}"
                elif device_type == "Tablet":
                    model = f"Galaxy Tab S{self.random.randint(7, 9)}"
                else:  # Hotspot
                    model = f"Mobile Hotspot {self.random.randint(1, 5)}"
            elif manufacturer == "Google":
                if device_type == "Smartphone":
                    model = f"Pixel {self.random.randint(6, 9)} {self.random.choice(['', 'Pro'])}"
                elif device_type == "Tablet":
                    model = f"Pixel Tablet {self.random.choice(['', 'Pro'])}"
                else:  # Hotspot
                    model = f"Nexus Hotspot {self.random.randint(1, 3)}"
            elif manufacturer == "Motorola":
                if device_type == "Smartphone":
                    model = f"Moto G{self.random.randint(10, 15)} {self.random.choice(['', 'Plus', 'Power'])}"
                elif device_type == "Tablet":
                    model = f"Moto Tab G{self.random.randint(70, 90)}"
                else:  # Hotspot
                    model = f"Moto Hotspot {self.random.randint(1, 5)}"
            else:  # OnePlus
                if device_type == "Smartphone":
                    model = f"OnePlus {self.random.randint(9, 12)} {self.random.choice(['', 'Pro', 'T'])}"
                elif device_type == "Tablet":
                    model = f"OnePlus Pad {self.random.choice(['', 'Pro'])}"
                else:  # Hotspot
                    model = f"OnePlus Connect {self.random.randint(1, 3)}"

            device_name = f"{manufacturer} {model}"

            # Select storage
            storage_gb = self.random.choice(storage_options)

            # Set price based on device type, storage, and manufacturer premium
            price_min, price_max = price_ranges[device_type]

            # Adjust price based on storage and manufacturer
            storage_factor = (
                storage_gb / storage_options[2]
            )  # Normalize to the middle option
            manufacturer_premium = {
                "Apple": 1.3,
                "Samsung": 1.2,
                "Google": 1.1,
                "OnePlus": 0.9,
                "Motorola": 0.8,
            }

            base_price = price_min + (price_max - price_min) * self.random.random()
            retail_price = round(
                base_price * storage_factor * manufacturer_premium[manufacturer], 2
            )

            # Set monthly installment (typically 1/24 to 1/36 of retail price)
            installment_months = self.random.choice([24, 30, 36])
            monthly_installment = round(retail_price / installment_months, 2)

            # Generate color options
            colors = [
                "Black",
                "White",
                "Silver",
                "Gold",
                "Blue",
                "Red",
                "Purple",
                "Green",
            ]
            num_colors = self.random.randint(3, 6)
            color_options = ", ".join(self.random.sample(colors, num_colors))

            # Generate release date
            start_date, end_date = self.config["date_ranges"]["device_release"]
            release_date = self.random_date(start_date, end_date)

            # Set 5G compatibility (more recent devices are more likely to be 5G)
            release_year = release_date.year
            is_5g_compatible = (
                release_year >= 2022
                or (release_year == 2021 and self.random.random() < 0.7)
                or (release_year == 2020 and self.random.random() < 0.3)
            )

            # Most devices are active, but older ones might not be
            is_active = self.random.random() < (0.5 + (release_year - 2019) * 0.1)

            # Add to data
            data.append(
                (
                    device_id,
                    device_name,
                    manufacturer,
                    device_type,
                    retail_price,
                    monthly_installment,
                    storage_gb,
                    color_options,
                    release_date.date(),  # Convert datetime to date
                    is_5g_compatible,
                    is_active,
                )
            )

        # Create schema
        schema = StructType(
            [
                StructField("device_id", StringType(), False),
                StructField("device_name", StringType(), False),
                StructField("manufacturer", StringType(), False),
                StructField("device_type", StringType(), False),
                StructField("retail_price", FloatType(), False),
                StructField("monthly_installment", FloatType(), False),
                StructField("storage_gb", IntegerType(), False),
                StructField("color_options", StringType(), False),
                StructField("release_date", DateType(), False),
                StructField("is_5g_compatible", BooleanType(), False),
                StructField("is_active", BooleanType(), False),
            ]
        )

        # Create DataFrame
        df = self.create_dataframe_from_schema(schema, data)

        return df

    def generate_promotions(self) -> DataFrame:
        """Generate promotions data.

        Returns:
            DataFrame containing generated promotions data.
        """
        # Get config
        count = self.config["volumes"]["promotions"]
        discount_types = self.config["products"]["promotions"]["discount_types"]
        discount_ranges = self.config["products"]["promotions"]["discount_ranges"]

        # Generate promotion IDs
        promo_ids = self.generate_id("PROMO", 4001, count)

        # Prepare data
        data = []

        for promo_id in promo_ids:
            # Select discount type based on distribution
            discount_type = self.select_weighted(discount_types)

            # Generate promotion name and description based on type
            if discount_type == "Percentage":
                discount_value = self.random.randint(
                    discount_ranges["Percentage"][0], discount_ranges["Percentage"][1]
                )
                promo_name = f"{discount_value}% Off Plan"
                description = f"Get {discount_value}% off your monthly plan charges."

            elif discount_type == "Fixed":
                discount_value = self.random.randint(
                    discount_ranges["Fixed"][0], discount_ranges["Fixed"][1]
                )
                promo_name = f"${discount_value} Off Monthly Bill"
                description = f"Get ${discount_value} off your monthly bill."

            else:  # Service
                service_types = [
                    "Premium Content",
                    "Free Device Upgrade",
                    "Free International Calling",
                    "Extra Data",
                    "Priority Support",
                ]
                service = self.random.choice(service_types)

                # Service promotions don't have a direct discount value
                # Set a placeholder value for database consistency
                discount_value = 0.0

                promo_name = f"Free {service}"
                description = f"Get free {service.lower()} with your plan."

            # Generate start and end dates
            # Promotions last 3-6 months
            end_date = datetime.now() + timedelta(days=self.random.randint(30, 180))
            start_date = end_date - timedelta(days=self.random.randint(90, 180))

            # Determine if promotion is active based on dates
            is_active = start_date <= datetime.now() <= end_date

            # Add to data
            data.append(
                (
                    promo_id,
                    promo_name,
                    discount_type,
                    float(discount_value),  # Ensure discount_value is float
                    start_date.date(),  # Convert datetime to date
                    end_date.date(),  # Convert datetime to date
                    description,
                    is_active,
                )
            )

        # Create schema
        schema = StructType(
            [
                StructField("promo_id", StringType(), False),
                StructField("promo_name", StringType(), False),
                StructField("discount_type", StringType(), False),
                StructField("discount_value", FloatType(), False),
                StructField("start_date", DateType(), False),
                StructField("end_date", DateType(), False),
                StructField("description", StringType(), False),
                StructField("is_active", BooleanType(), False),
            ]
        )

        df = self.create_dataframe_from_schema(schema, data)

        return df
