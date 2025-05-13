"""Customer data generator.

This module contains the CustomerGenerator class for generating customer and subscription
data.
"""
from datetime import date, timedelta
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


class CustomerGenerator(BaseGenerator):
    """Generator for customer data.

    Generates data for customers and subscriptions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the customer generator.

        Args:
            config: Config dictionary.
        """
        super().__init__(config)

    def generate_customers(self) -> DataFrame:
        """Generate customer data.

        Returns:
            DataFrame containing generated customer data.
        """
        count = self.config["volumes"]["customers"]
        customer_segments = self.config["distributions"]["customer_segments"]
        customer_statuses = self.config["distributions"]["customer_statuses"]
        contact_methods = self.config["distributions"]["contact_methods"]

        # Generate customer IDs
        customer_ids = self.generate_id("CUS", 10001, count)

        data = []

        for customer_id in customer_ids:
            # Select customer segment based on distribution
            customer_segment = self.select_weighted(customer_segments)

            # Generate city and state
            city = self.fake.city()
            state = self.fake.state_abbr()

            # Generate registration date
            start_date, end_date = self.config["date_ranges"]["customer_registration"]
            registration_date = self.random_date(start_date, end_date)

            # Select customer status based on distribution
            customer_status = self.select_weighted(customer_statuses)

            # Select preferred contact method based on distribution
            preferred_contact_method = self.select_weighted(contact_methods)

            # Calculate days as customer (tenure)
            days_as_customer = (date.today() - registration_date.date()).days
            years_as_customer = days_as_customer / 365.0

            # Assign loyalty tier based on segment and tenure
            loyalty_tier_weights = {
                "Bronze": 0.4,
                "Silver": 0.3,
                "Gold": 0.2,
                "Platinum": 0.1,
            }

            # Adjust weights based on customer segment and tenure
            if customer_segment in ["Premium", "Business"]:
                # Premium/Business customers more likely to be Gold/Platinum
                loyalty_tier_weights["Bronze"] -= 0.2
                loyalty_tier_weights["Silver"] -= 0.1
                loyalty_tier_weights["Gold"] += 0.1
                loyalty_tier_weights["Platinum"] += 0.2

            # Longer tenure increases tier probabilities
            if years_as_customer > 3:
                loyalty_tier_weights["Bronze"] -= min(
                    0.2, loyalty_tier_weights["Bronze"]
                )
                loyalty_tier_weights["Silver"] -= min(
                    0.1, loyalty_tier_weights["Silver"]
                )
                loyalty_tier_weights["Gold"] += 0.1
                loyalty_tier_weights["Platinum"] += 0.2

            # normalize
            total = sum(loyalty_tier_weights.values())
            loyalty_tier_weights = {
                k: v / total for k, v in loyalty_tier_weights.items()
            }

            loyalty_tier = self.select_weighted(loyalty_tier_weights)

            # Calculate churn risk score (0-100)
            # Base churn risk - random
            churn_risk_score = self.random.randint(20, 80)

            # Factors that decrease churn risk
            if customer_segment in ["Premium", "Business"]:
                churn_risk_score -= 15
            if years_as_customer > 2:
                churn_risk_score -= min(20, int(years_as_customer * 5))
            if loyalty_tier in ["Gold", "Platinum"]:
                churn_risk_score -= 10

            # Factors that increase churn risk
            if customer_status == "Inactive":
                churn_risk_score += 30
            elif customer_status == "Suspended":
                churn_risk_score += 40

            # Ensure range 1-100
            churn_risk_score = max(1, min(100, churn_risk_score))

            # Calculate customer value score (0-100)
            # Base value tied to segment
            segment_values = {
                "Individual": 40,
                "Family": 60,
                "Business": 75,
                "Premium": 85,
                "Student": 30,
            }
            customer_value_score = segment_values.get(customer_segment, 50)

            # Adjust based on loyalty tier
            tier_adjustments = {"Bronze": 0, "Silver": 10, "Gold": 20, "Platinum": 30}
            customer_value_score += tier_adjustments.get(loyalty_tier, 0)

            # Longer relationships tend to be more valuable
            customer_value_score += min(15, int(years_as_customer * 3))

            # Random factor
            customer_value_score += self.random.randint(-10, 10)

            # Ensure range 1-100
            customer_value_score = max(1, min(100, customer_value_score))

            # Calculate satisfaction score (0-100)
            # Base satisfaction
            satisfaction_score = self.random.randint(50, 90)

            # Factors affecting satisfaction
            if loyalty_tier in ["Gold", "Platinum"]:
                satisfaction_score += 10
            if customer_status == "Suspended":
                satisfaction_score -= 30

            # Inverse correlation with churn risk (but not perfect)
            satisfaction_score -= (churn_risk_score - 50) // 5

            # Ensure range 1-100
            satisfaction_score = max(1, min(100, satisfaction_score))

            data.append(
                (
                    customer_id,
                    customer_segment,
                    city,
                    state,
                    registration_date.date(),  # convert datetime to date
                    customer_status,
                    preferred_contact_method,
                    loyalty_tier,
                    churn_risk_score,
                    customer_value_score,
                    satisfaction_score,
                )
            )

        schema = StructType(
            [
                StructField("customer_id", StringType(), False),
                StructField("customer_segment", StringType(), False),
                StructField("city", StringType(), False),
                StructField("state", StringType(), False),
                StructField("registration_date", DateType(), False),
                StructField("customer_status", StringType(), False),
                StructField("preferred_contact_method", StringType(), False),
                StructField("loyalty_tier", StringType(), False),
                StructField("churn_risk_score", IntegerType(), False),
                StructField("customer_value_score", IntegerType(), False),
                StructField("satisfaction_score", IntegerType(), False),
            ]
        )

        df = self.create_dataframe_from_schema(schema, data)

        return df

    def generate_subscriptions(
        self,
        plans_df: DataFrame,
        devices_df: DataFrame,
        promotions_df: DataFrame,
        customers_df: DataFrame,
    ) -> DataFrame:
        """Generate subscription data with references to customers and products.

        Args:
            plans_df: DataFrame containing plan data.
            devices_df: DataFrame containing device data.
            promotions_df: DataFrame containing promotion data.
            customers_df: DataFrame containing customer data.

        Returns:
            DataFrame containing generated subscription data.
        """
        subscription_statuses = self.config["distributions"]["subscription_statuses"]

        # Collect customers to get their registration dates
        customers = customers_df.collect()

        # Collect plan, device, and promotion info for assignment and pricing
        plans = plans_df.filter("is_active = true").collect()
        devices = devices_df.filter("is_active = true").collect()
        promotions = promotions_df.filter("is_active = true").collect()

        # Create lookup dictionaries for pricing info with explicit float casting
        plan_price_lookup = {row.plan_id: float(row.monthly_price) for row in plans}
        device_price_lookup = {
            row.device_id: float(row.monthly_installment) for row in devices
        }
        promo_lookup = {
            row.promo_id: (row.discount_type, float(row.discount_value))
            for row in promotions
        }

        # On average, each customer will have 1-3 subscriptions
        # Let's generate random counts for each customer
        subscription_counts = []
        total_subscriptions = 0

        for _ in range(len(customers)):
            # Weighted distribution: 60% have 1, 30% have 2, 10% have 3+
            if self.random.random() < 0.6:
                count = 1
            elif self.random.random() < 0.9:
                count = 2
            else:
                count = self.random.randint(3, 5)

            subscription_counts.append(count)
            total_subscriptions += count

        # Generate subscription IDs
        subscription_ids = self.generate_id("SUB", 10000001, total_subscriptions)

        data = []
        subscription_idx = 0

        for customer_idx, customer in enumerate(customers):
            customer_id = customer.customer_id
            registration_date = customer.registration_date
            subscription_count = subscription_counts[customer_idx]

            for _ in range(subscription_count):
                subscription_id = subscription_ids[subscription_idx]
                subscription_idx += 1

                # Select random plan - ensure plan type matches customer segment
                customer_segment = customer.customer_segment
                matching_plans = [
                    p
                    for p in plans
                    if (
                        (
                            customer_segment == "Individual"
                            and p.plan_type == "Individual"
                        )
                        or (customer_segment == "Family" and p.plan_type == "Family")
                        or (
                            customer_segment == "Business" and p.plan_type == "Business"
                        )
                        or (
                            (
                                customer_segment == "Premium"
                                or customer_segment == "Student"
                            )
                            and p.plan_type == "Individual"
                        )
                    )
                ]

                if not matching_plans:
                    # Fallback to any plan if no match (shouldn't happen with proper config)
                    matching_plans = plans

                plan = self.random.choice(matching_plans)
                plan_id = plan.plan_id

                # Devices are optional - 80% chance to have one
                device = None
                device_id = None
                if self.random.random() < 0.8 and devices:
                    device = self.random.choice(devices)
                    device_id = device.device_id

                # Promotions are optional - 30% chance to have one
                promo = None
                promo_id = None
                if self.random.random() < 0.3 and promotions:
                    promo = self.random.choice(promotions)
                    promo_id = promo.promo_id

                # Generate subscription start date (after customer registration)
                # Start date between registration and now, weighted towards more recent
                days_since_reg = (date.today() - registration_date).days
                if days_since_reg <= 0:
                    # If registration date is in the future (which shouldn't happen),
                    # use a small positive value as a fallback
                    days_since_reg = 30

                random_days = int(self.random.betavariate(2, 5) * days_since_reg)
                subscription_start_date = registration_date + timedelta(
                    days=random_days
                )

                # Set contract length (0, 12, 24, or 36 months)
                contract_lengths = [0, 12, 24, 36]
                # If plan requires contract, exclude 0 months
                if plan.contract_required:
                    contract_lengths = [12, 24, 36]
                    contract_weights = [0.3, 0.6, 0.1]  # 24 months most common
                else:
                    contract_weights = [0.2, 0.3, 0.4, 0.1]

                contract_length_months = self.random.choices(
                    contract_lengths, weights=contract_weights, k=1
                )[0]

                # Calculate monthly charge based on plan, device, and promotion
                monthly_charge = float(plan_price_lookup.get(plan_id, 0.0))

                # Add device payment if applicable
                if device_id:
                    monthly_charge += float(device_price_lookup.get(device_id, 0.0))

                # Apply promotion if applicable
                if promo_id:
                    discount_type, discount_value = promo_lookup.get(
                        promo_id, ("", 0.0)
                    )
                    if discount_type == "Percentage":
                        # Apply percentage discount with explicit floats
                        monthly_charge = monthly_charge * (
                            1.0 - (float(discount_value) / 100.0)
                        )
                    elif discount_type == "Fixed":
                        # Apply fixed discount with explicit floats
                        monthly_charge = max(
                            0.0, monthly_charge - float(discount_value)
                        )
                    # Service discounts don't affect the price

                # Round to 2 decimal places and ensure it's a float
                monthly_charge = float(round(monthly_charge, 2))

                # Determine status
                status = self.select_weighted(subscription_statuses)

                # Autopay (70% enabled)
                autopay_enabled = self.random.random() < 0.7

                data.append(
                    (
                        subscription_id,
                        customer_id,
                        plan_id,
                        device_id,
                        promo_id,
                        subscription_start_date,
                        contract_length_months,
                        monthly_charge,
                        status,
                        autopay_enabled,
                    )
                )

        schema = StructType(
            [
                StructField("subscription_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("plan_id", StringType(), False),
                StructField("device_id", StringType(), True),  # Nullable
                StructField("promo_id", StringType(), True),  # Nullable
                StructField("subscription_start_date", DateType(), False),
                StructField("contract_length_months", IntegerType(), False),
                StructField("monthly_charge", FloatType(), False),
                StructField("status", StringType(), False),
                StructField("autopay_enabled", BooleanType(), False),
            ]
        )

        df = self.create_dataframe_from_schema(schema, data)

        return df
