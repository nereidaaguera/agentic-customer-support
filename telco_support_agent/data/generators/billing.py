"""Billing data generator.

This module contains the BillingGenerator class for generating billing and usage data.
"""

from datetime import date, datetime, timedelta
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DateType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from telco_support_agent.data.generators.base import BaseGenerator


class BillingGenerator(BaseGenerator):
    """Generator for billing data.

    Generates data for billing records and usage records.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the billing generator.

        Args:
            config: Config dictionary.
        """
        super().__init__(config)

    def generate_billing(self, subscriptions_df: DataFrame) -> DataFrame:
        """Generate billing data for subscriptions.

        Args:
            subscriptions_df: DataFrame containing subscription data.

        Returns:
            DataFrame containing generated billing data.
        """
        # Get billing cycle dates from config
        start_cycle, end_cycle = self.config["date_ranges"]["billing_cycles"]
        start_date = datetime.strptime(f"{start_cycle}-01", "%Y-%m-%d").date()
        end_date = datetime.strptime(f"{end_cycle}-28", "%Y-%m-%d").date()

        # Get payment status distribution
        payment_statuses = self.config["distributions"]["payment_statuses"]

        # Collect subscriptions data
        subscriptions = subscriptions_df.filter("status = 'Active'").collect()

        # Generate data for each active subscription for each billing cycle
        data = []

        # Create a list of billing cycle dates
        billing_cycles = []
        current_date = start_date
        while current_date <= end_date:
            billing_cycles.append(current_date.strftime("%Y-%m"))
            # Move to the next month
            if current_date.month == 12:
                current_date = date(current_date.year + 1, 1, 1)
            else:
                current_date = date(current_date.year, current_date.month + 1, 1)

        # Generate a billing ID counter
        billing_id_counter = 1234567890

        # Generate billing records for each subscription for each cycle
        for subscription in subscriptions:
            subscription_id = subscription.subscription_id
            customer_id = subscription.customer_id
            monthly_charge = subscription.monthly_charge

            # Skip billing cycles before the subscription started
            subscription_start = subscription.subscription_start_date
            subscription_start_cycle = subscription_start.strftime("%Y-%m")

            for cycle in billing_cycles:
                # Skip cycles before subscription started
                if cycle < subscription_start_cycle:
                    continue

                # Generate billing date (1st of the month)
                billing_cycle_date = datetime.strptime(f"{cycle}-01", "%Y-%m-%d").date()

                # Generate due date (15th of the month)
                due_date = date(billing_cycle_date.year, billing_cycle_date.month, 15)

                # Base charge from subscription
                base_amount = monthly_charge

                # Add random additional charges (overages, etc.) to some bills
                # More likely for higher base charges
                additional_charges = 0.0
                if self.random.random() < min(0.3, base_amount / 200):
                    # Additional charges more common for higher base amounts
                    additional_charges = round(self.random.uniform(5.0, 50.0), 2)

                # Calculate tax (approximately 8-12% of pre-tax total)
                tax_rate = self.random.uniform(0.08, 0.12)
                pre_tax = base_amount + additional_charges
                tax_amount = round(pre_tax * tax_rate, 2)

                # Calculate total amount
                total_amount = round(pre_tax + tax_amount, 2)

                # Set payment status based on distribution (weighted)
                # But make older bills more likely to be paid
                months_ago = (date.today().year - billing_cycle_date.year) * 12 + (
                    date.today().month - billing_cycle_date.month
                )
                paid_boost = min(
                    0.5, months_ago * 0.1
                )  # Older bills are more likely to be paid

                # Adjust probabilities based on age of bill
                adj_payment_statuses = {
                    "Paid": payment_statuses["Paid"] + paid_boost,
                    "Unpaid": payment_statuses["Unpaid"] - (paid_boost * 0.7),
                    "Late": payment_statuses["Late"] - (paid_boost * 0.2),
                    "Partial": payment_statuses["Partial"] - (paid_boost * 0.1),
                }

                # Normalize probabilities
                total_prob = sum(adj_payment_statuses.values())
                adj_payment_statuses = {
                    k: v / total_prob for k, v in adj_payment_statuses.items()
                }

                status = self.select_weighted(adj_payment_statuses)

                # Set payment details based on status
                payment_amount = None
                payment_date = None
                payment_method = None

                if status == "Paid":
                    payment_amount = total_amount
                    # Payment date between billing date and due date (most likely)
                    # or up to 3 days late (less likely)
                    if self.random.random() < 0.9:  # 90% on time
                        days_after_billing = self.random.randint(
                            1, (due_date - billing_cycle_date).days
                        )
                        payment_date = billing_cycle_date + timedelta(
                            days=days_after_billing
                        )
                    else:  # 10% slightly late but still marked as paid
                        days_late = self.random.randint(1, 3)
                        payment_date = due_date + timedelta(days=days_late)

                    # Select payment method
                    payment_method = self.random.choice(
                        ["Credit Card", "Bank Transfer", "PayPal", "Check"]
                    )

                elif status == "Partial":
                    # Pay between 30% and 70% of total
                    payment_amount = round(
                        total_amount * self.random.uniform(0.3, 0.7), 2
                    )
                    # Payment date usually after due date
                    days_late = self.random.randint(1, 15)
                    payment_date = due_date + timedelta(days=days_late)
                    payment_method = self.random.choice(
                        ["Credit Card", "Bank Transfer", "PayPal", "Check"]
                    )

                elif status == "Late":
                    # Bill marked as late, but eventually paid in full
                    if (
                        months_ago >= 2
                    ):  # Only older bills show as paid if they were late
                        payment_amount = total_amount
                        days_late = self.random.randint(5, 30)
                        payment_date = due_date + timedelta(days=days_late)
                        payment_method = self.random.choice(
                            ["Credit Card", "Bank Transfer", "PayPal", "Check"]
                        )

                # Generate billing ID
                billing_id = f"BILL-{billing_id_counter}"
                billing_id_counter += 1

                data.append(
                    (
                        billing_id,
                        subscription_id,
                        customer_id,
                        billing_cycle_date,
                        due_date,
                        float(base_amount),
                        float(additional_charges),
                        float(tax_amount),
                        float(total_amount),
                        None if payment_amount is None else float(payment_amount),
                        payment_date,
                        payment_method,
                        status,
                        cycle,
                    )
                )

        schema = StructType(
            [
                StructField("billing_id", StringType(), False),
                StructField("subscription_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("billing_date", DateType(), False),
                StructField("due_date", DateType(), False),
                StructField("base_amount", FloatType(), False),
                StructField("additional_charges", FloatType(), False),
                StructField("tax_amount", FloatType(), False),
                StructField("total_amount", FloatType(), False),
                StructField("payment_amount", FloatType(), True),  # Nullable
                StructField("payment_date", DateType(), True),  # Nullable
                StructField("payment_method", StringType(), True),  # Nullable
                StructField("status", StringType(), False),
                StructField("billing_cycle", StringType(), False),
            ]
        )

        df = self.create_dataframe_from_schema(schema, data)

        return df

    def generate_usage(self, subscriptions_df: DataFrame) -> DataFrame:
        """Generate usage data for subscriptions.

        Args:
            subscriptions_df: DataFrame containing subscription data.

        Returns:
            DataFrame containing generated usage data.
        """
        # Collect active subscriptions with plan details
        subscriptions = subscriptions_df.filter("status = 'Active'").collect()

        # Get plan data for reference
        # In a real implementation, we should join with plans_df to get data limits
        # For now, we'll extract the data limit from the plan name (e.g. "Premium Individual")

        # Generate data for usage tracking
        data = []

        # Generate a usage ID counter
        usage_id_counter = 123456789012

        # Define billing cycles based on config
        start_cycle, end_cycle = self.config["date_ranges"]["billing_cycles"]
        start_date = datetime.strptime(f"{start_cycle}-01", "%Y-%m-%d").date()
        end_date = datetime.strptime(f"{end_cycle}-28", "%Y-%m-%d").date()

        # Generate daily usage for each active subscription
        for subscription in subscriptions:
            subscription_id = subscription.subscription_id
            subscription_start = subscription.subscription_start_date

            # Skip if subscription started after our usage tracking period
            if subscription_start > end_date:
                continue

            # Start tracking from later of start_date or subscription_start
            tracking_start = max(start_date, subscription_start)

            # Generate usage for each day in the period
            current_date = tracking_start
            while current_date <= end_date:
                # Skip weekends for some randomness in usage patterns
                if current_date.weekday() >= 5 and self.random.random() < 0.3:
                    current_date += timedelta(days=1)
                    continue

                # Get cycle for current date
                cycle = current_date.strftime("%Y-%m")

                # Generate usage based on subscription type (approximated from monthly_charge)
                # Higher monthly charge = higher data/voice/SMS plan
                base_charge = subscription.monthly_charge

                # Base daily limits
                # These are rough approximations - ideally would be based on actual plan limits
                if base_charge >= 100:  # Premium/Unlimited plans
                    daily_data_limit = 5000  # MB (5GB)
                    voice_limit = 120  # minutes
                    sms_limit = 100  # messages
                elif base_charge >= 70:  # High tier plans
                    daily_data_limit = 2000  # MB (2GB)
                    voice_limit = 80  # minutes
                    sms_limit = 60  # messages
                elif base_charge >= 40:  # Mid tier plans
                    daily_data_limit = 1000  # MB (1GB)
                    voice_limit = 60  # minutes
                    sms_limit = 40  # messages
                else:  # Basic plans
                    daily_data_limit = 500  # MB (0.5GB)
                    voice_limit = 30  # minutes
                    sms_limit = 20  # messages

                # Generate random usage amounts
                # Use beta distribution for more realistic clustering
                data_usage_mb = round(
                    self.random.betavariate(2, 5) * daily_data_limit, 1
                )
                voice_minutes = round(self.random.betavariate(1, 4) * voice_limit, 1)
                sms_count = int(self.random.betavariate(1, 3) * sms_limit)

                # Occasionally generate a usage spike (10% chance)
                if self.random.random() < 0.1:
                    # 1.5-3x normal usage for spikes
                    multiplier = self.random.uniform(1.5, 3.0)
                    if self.random.random() < 0.33:  # Choose which type of usage spikes
                        data_usage_mb *= multiplier
                    elif self.random.random() < 0.5:
                        voice_minutes *= multiplier
                    else:
                        sms_count = int(sms_count * multiplier)

                # Generate usage ID
                usage_id = f"USG-{usage_id_counter}"
                usage_id_counter += 1

                data.append(
                    (
                        usage_id,
                        subscription_id,
                        current_date,
                        float(data_usage_mb),
                        float(voice_minutes),
                        sms_count,
                        cycle,
                    )
                )

                # Move to next day
                current_date += timedelta(days=1)

        schema = StructType(
            [
                StructField("usage_id", StringType(), False),
                StructField("subscription_id", StringType(), False),
                StructField("date", DateType(), False),
                StructField("data_usage_mb", FloatType(), False),
                StructField("voice_minutes", FloatType(), False),
                StructField("sms_count", IntegerType(), False),
                StructField("billing_cycle", StringType(), False),
            ]
        )

        df = self.create_dataframe_from_schema(schema, data)

        return df
