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
        """Generate devices data with realistic market distribution.

        Returns:
            DataFrame containing generated devices data.
        """
        # Get config
        count = self.config["volumes"]["devices"]
        device_types = self.config["distributions"]["device_types"]
        storage_options = self.config["products"]["devices"]["storage_options"]
        price_ranges = self.config["products"]["devices"]["price_ranges"]

        manufacturer_distribution = {
            "Apple": 0.60,
            "Samsung": 0.23,
            "Google": 0.05,
            "Motorola": 0.04,
            "OnePlus": 0.02,
            "Xiaomi": 0.01,
        }

        # Generate device IDs
        device_ids = self.generate_id("DEV", 2001, count)

        # Prepare data
        data = []

        for device_id in device_ids:
            # Select device type based on distribution
            device_type = self.select_weighted(device_types)

            # Select manufacturer based on market share distribution
            manufacturer = self.select_weighted(manufacturer_distribution)

            # Generate device name based on manufacturer and latest models
            if manufacturer == "Apple":
                if device_type == "Smartphone":
                    model = f"iPhone {self.random.randint(13, 16)} {self.random.choice(['', 'Pro', 'Pro Max'])}"
                elif device_type == "Tablet":
                    model = f"iPad {self.random.choice(['', 'Air', 'Pro', 'mini'])} {self.random.randint(9, 11)}"
                elif device_type == "Hotspot":
                    model = f"MiFi {self.random.randint(1, 5)}"
            elif manufacturer == "Samsung":
                if device_type == "Smartphone":
                    model = f"Galaxy S{self.random.randint(22, 25)} {self.random.choice(['', 'Plus', 'Ultra', 'Edge'])}"
                elif device_type == "Tablet":
                    model = f"Galaxy Tab S{self.random.randint(7, 10)}"
                elif device_type == "Hotspot":
                    model = f"Mobile Hotspot {self.random.choice(['A', 'B', 'C'])}{self.random.randint(10, 20)}"
            elif manufacturer == "Google":
                if device_type == "Smartphone":
                    model = f"Pixel {self.random.randint(7, 9)} {self.random.choice(['', 'Pro', 'Pro XL'])}"
                elif device_type == "Tablet":
                    model = f"Pixel Tablet {self.random.choice(['', 'Pro'])}"
                elif device_type == "Hotspot":
                    model = f"Nexus Connect {self.random.randint(1, 5)}"
            elif manufacturer == "Motorola":
                if device_type == "Smartphone":
                    model = f"Moto G{self.random.randint(10, 15)} {self.random.choice(['', 'Plus', 'Power'])}"
                elif device_type == "Tablet":
                    model = f"Moto Tab G{self.random.randint(70, 90)}"
                elif device_type == "Hotspot":
                    model = f"Motorola MiFi {self.random.randint(100, 500)}"
            elif manufacturer == "OnePlus":
                if device_type == "Smartphone":
                    model = f"OnePlus {self.random.randint(11, 13)} {self.random.choice(['', 'Pro', 'T'])}"
                elif device_type == "Tablet":
                    model = f"OnePlus Pad {self.random.choice(['', 'Pro'])}"
                elif device_type == "Hotspot":
                    model = f"OnePlus Connect {self.random.randint(1, 3)}"
            else:
                if device_type == "Smartphone":
                    model = f"Xiaomi {self.random.choice(['Redmi', 'POCO', 'Mi'])} {self.random.randint(12, 14)}"
                elif device_type == "Tablet":
                    model = f"Xiaomi Pad {self.random.randint(6, 8)}"
                elif device_type == "Hotspot":
                    model = f"Mi WiFi {self.random.randint(3, 6)}"

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
                "Xiaomi": 0.7,
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
            ]
            max_colors = min(6, len(colors))
            num_colors = self.random.randint(3, max_colors)
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
                    release_date.date(),
                    is_5g_compatible,
                    is_active,
                )
            )

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

        df = self.create_dataframe_from_schema(schema, data)

        return df

    def generate_promotions(self) -> DataFrame:
        """Generate promotions data with realistic telecom marketing strategies.

        Returns:
            DataFrame containing generated promotions data.
        """
        # Get config
        count = self.config["volumes"]["promotions"]

        # Generate promotion IDs
        promo_ids = self.generate_id("PROMO", 4001, count)

        # Prepare data
        data = []

        promotion_types = [
            # New customer acquisition
            {
                "category": "New Customer",
                "formats": [
                    {
                        "name": "First 3 Months 50% Off",
                        "type": "Percentage",
                        "value": 50.0,
                        "duration": 90,
                    },
                    {
                        "name": "Switch & Save",
                        "type": "Fixed",
                        "value": 20.0,
                        "duration": 180,
                    },
                    {
                        "name": "First Month Free",
                        "type": "Percentage",
                        "value": 100.0,
                        "duration": 30,
                    },
                    {
                        "name": "Port-In Credit",
                        "type": "Fixed",
                        "value": 200.0,
                        "duration": 30,
                    },
                    {
                        "name": "BOGO Line Offer",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 365,
                    },
                    {
                        "name": "Family Migration Bonus",
                        "type": "Fixed",
                        "value": 75.0,
                        "duration": 60,
                    },
                ],
            },
            # Device promotions
            {
                "category": "Device",
                "formats": [
                    {
                        "name": "iPhone 15 Trade-In Credit",
                        "type": "Fixed",
                        "value": 100.0,
                        "duration": 60,
                    },
                    {
                        "name": "Galaxy S25 Upgrade",
                        "type": "Fixed",
                        "value": 150.0,
                        "duration": 90,
                    },
                    {
                        "name": "Free Pixel 9 with Premium Plan",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 60,
                    },
                    {
                        "name": "Free Phone Upgrade",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 30,
                    },
                    {
                        "name": "Device Protection Bundle",
                        "type": "Percentage",
                        "value": 50.0,
                        "duration": 180,
                    },
                    {
                        "name": "Premium Accessory Package",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 30,
                    },
                    {
                        "name": "Early Upgrade Eligibility",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 180,
                    },
                ],
            },
            # Plan promotions
            {
                "category": "Plan",
                "formats": [
                    {
                        "name": "Unlimited Plan Discount",
                        "type": "Percentage",
                        "value": 15.0,
                        "duration": 120,
                    },
                    {
                        "name": "Family Plan Extra Line Free",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 90,
                    },
                    {
                        "name": "Premium Plan Trial",
                        "type": "Percentage",
                        "value": 25.0,
                        "duration": 60,
                    },
                    {
                        "name": "Data Rollover Plus",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 365,
                    },
                    {
                        "name": "International Add-On Included",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 90,
                    },
                    {
                        "name": "Streaming Service Bundle",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 180,
                    },
                    {
                        "name": "Multi-Line Discount",
                        "type": "Percentage",
                        "value": 10.0,
                        "duration": 365,
                    },
                ],
            },
            # Seasonal promotions
            {
                "category": "Seasonal",
                "formats": [
                    {
                        "name": "Summer Data Boost",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 90,
                    },
                    {
                        "name": "Back to School Bundle",
                        "type": "Fixed",
                        "value": 25.0,
                        "duration": 45,
                    },
                    {
                        "name": "Holiday Device Offer",
                        "type": "Fixed",
                        "value": 50.0,
                        "duration": 30,
                    },
                    {
                        "name": "Black Friday Special",
                        "type": "Percentage",
                        "value": 40.0,
                        "duration": 14,
                    },
                    {
                        "name": "Valentine's Couples Plan",
                        "type": "Fixed",
                        "value": 30.0,
                        "duration": 28,
                    },
                    {
                        "name": "Spring Trade-In Event",
                        "type": "Fixed",
                        "value": 75.0,
                        "duration": 45,
                    },
                    {
                        "name": "Graduation Gift Promotion",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 60,
                    },
                ],
            },
            # Loyalty/retention
            {
                "category": "Loyalty",
                "formats": [
                    {
                        "name": "Loyalty Reward",
                        "type": "Percentage",
                        "value": 10.0,
                        "duration": 150,
                    },
                    {
                        "name": "Anniversary Discount",
                        "type": "Fixed",
                        "value": 15.0,
                        "duration": 60,
                    },
                    {
                        "name": "Premium Customer Thank You",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 90,
                    },
                    {
                        "name": "Long-Term Customer Rate Lock",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 365,
                    },
                    {
                        "name": "Priority Support Access",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 180,
                    },
                    {
                        "name": "Annual Upgrade Option",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 365,
                    },
                    {
                        "name": "5-Year Milestone Gift",
                        "type": "Fixed",
                        "value": 100.0,
                        "duration": 30,
                    },
                ],
            },
            {
                "category": "Business",
                "formats": [
                    {
                        "name": "Small Business Starter",
                        "type": "Percentage",
                        "value": 25.0,
                        "duration": 180,
                    },
                    {
                        "name": "Multi-Line Business Discount",
                        "type": "Fixed",
                        "value": 15.0,
                        "duration": 365,
                    },
                    {
                        "name": "Enterprise Solution Package",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 90,
                    },
                    {
                        "name": "Business Device Fleet Upgrade",
                        "type": "Fixed",
                        "value": 50.0,
                        "duration": 120,
                    },
                ],
            },
            {
                "category": "Limited-Time",
                "formats": [
                    {
                        "name": "Flash Sale",
                        "type": "Percentage",
                        "value": 35.0,
                        "duration": 5,
                    },
                    {
                        "name": "Weekend Special",
                        "type": "Fixed",
                        "value": 25.0,
                        "duration": 3,
                    },
                    {
                        "name": "One-Day Activation Bonus",
                        "type": "Fixed",
                        "value": 50.0,
                        "duration": 1,
                    },
                    {
                        "name": "48-Hour Priority Shipping",
                        "type": "Service",
                        "value": 0.0,
                        "duration": 2,
                    },
                ],
            },
        ]

        # Current month for seasonal timing
        current_month = datetime.now().month

        for promo_id in promo_ids:
            # Select promotion category and format
            category = self.random.choice(promotion_types)
            promo_format = self.random.choice(category["formats"])

            # Basic promotion details
            promo_name = promo_format["name"]
            discount_type = promo_format["type"]
            discount_value = promo_format["value"]
            duration_days = promo_format["duration"]

            # Generate realistic descriptions
            if discount_type == "Percentage":
                description = f"{promo_name}: Get {int(discount_value)}% off for {duration_days // 30} months on qualifying plans."
                if duration_days < 30:
                    description = f"{promo_name}: Get {int(discount_value)}% off for {duration_days} days on qualifying plans."
            elif discount_type == "Fixed":
                description = f"{promo_name}: ${int(discount_value)} monthly credit for {duration_days // 30} months with eligible service."
                if duration_days < 30:
                    description = f"{promo_name}: ${int(discount_value)} credit for {duration_days} days with eligible service."
            else:  # Service
                description = f"{promo_name}: Enjoy this special offer for {duration_days // 30} months with qualifying service."
                if duration_days < 30:
                    description = f"{promo_name}: Enjoy this special offer for {duration_days} days with qualifying service."

            # Add plan or device targeting for some promotions
            if category["category"] == "Plan":
                plan_tiers = ["Basic", "Standard", "Premium", "Unlimited"]
                tier = self.random.choice(plan_tiers)
                description += f" Available with {tier} plans."
                promo_name = f"{tier} {promo_name}"

            if category["category"] == "Device":
                # Keep device name in promotion name but add clarity
                description += " Requires new line activation and approved credit."

                # Add specific device details for certain promotions
                if "Free Phone Upgrade" in promo_name:
                    devices = ["iPhone 15", "Galaxy S24", "Pixel 8", "OnePlus 12"]
                    eligible_device = self.random.choice(devices)
                    description += f" Eligible customers can upgrade to a {eligible_device} at no additional cost."

                if "Protection Bundle" in promo_name:
                    description += " Includes device insurance, screen protection, and priority tech support."

            # descriptions for specific promotion types
            if "Port-In Credit" in promo_name:
                description += " Get a $200 bill credit when you bring your number from another carrier."

            if "BOGO" in promo_name:
                description += " Add a second line to your account and get it free for a full year."

            if "Data Rollover" in promo_name:
                description += " Unused data from your monthly allowance automatically rolls over to the next month."

            if "Stream" in promo_name:
                streaming_services = [
                    "Netflix",
                    "Disney+",
                    "HBO Max",
                    "Prime Video",
                    "Hulu",
                ]
                included_services = self.random.sample(streaming_services, 2)
                description += f" Includes 6-month subscriptions to {included_services[0]} and {included_services[1]}."

            if "Black Friday" in promo_name:
                description += (
                    " Our biggest discounts of the year on select devices and plans."
                )

            if "5-Year Milestone" in promo_name:
                description += (
                    " Special reward for customers who have been with us for 5+ years."
                )

            # Handle seasonal timing more realistically
            if category["category"] == "Seasonal":
                # Summer promos in summer, holiday promos in winter, etc.
                if "Summer" in promo_name:
                    start_month = max(5, min(6, current_month))  # May-June
                elif "Holiday" in promo_name or "Black Friday" in promo_name:
                    start_month = max(11, min(12, current_month))  # Nov-Dec
                elif "School" in promo_name or "Graduation" in promo_name:
                    start_month = max(7, min(8, current_month))  # July-Aug
                elif "Valentine" in promo_name:
                    start_month = 2  # February
                elif "Spring" in promo_name:
                    start_month = 3  # March
                else:
                    start_month = current_month
            elif category["category"] == "Limited-Time":
                # Limited-time offers should be very recent or upcoming
                days_offset = self.random.randint(-5, 15)
                start_date = datetime.now() + timedelta(days=days_offset)
                end_date = start_date + timedelta(days=duration_days)

                # Determine if promotion is active
                is_active = start_date <= datetime.now() <= end_date

                # Add to data
                data.append(
                    (
                        promo_id,
                        promo_name,
                        discount_type,
                        float(discount_value),
                        start_date.date(),
                        end_date.date(),
                        description,
                        is_active,
                    )
                )

                # Skip the rest of the loop for limited-time offers as we've already set dates
                continue
            else:
                start_month = current_month

            # Generate start and end dates more realistically
            start_date = datetime(datetime.now().year, start_month, 1) + timedelta(
                days=self.random.randint(0, 15)
            )
            end_date = start_date + timedelta(days=duration_days)

            # Determine if promotion is active
            is_active = start_date <= datetime.now() <= end_date

            # Add to data
            data.append(
                (
                    promo_id,
                    promo_name,
                    discount_type,
                    float(discount_value),
                    start_date.date(),
                    end_date.date(),
                    description,
                    is_active,
                )
            )

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
