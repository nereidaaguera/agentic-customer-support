"""UC functions for account-related operations."""

import json

from telco_support_agent.tools.account.queries import (
    ACCOUNT_INFO_QUERY,
    SUBSCRIPTIONS_INFO_QUERY,
)
from telco_support_agent.tools.registry import uc_function
from telco_support_agent.utils.logging_utils import get_logger
from telco_support_agent.utils.spark_utils import spark

logger = get_logger(__name__)

# default catalog and schema for data
DATA_CATALOG = "telco_customer_support_dev"
DATA_SCHEMA = "bronze"


@uc_function(domain="account")
def get_customer_info(customer_id: str) -> str:
    """Retrieves basic customer information from the database.

    Args:
        customer_id (str): The customer ID in the format 'CUS-XXXXX'

    Returns:
        str: JSON string containing customer information including profile data and account status
    """
    try:
        query = ACCOUNT_INFO_QUERY.format(catalog=DATA_CATALOG, schema=DATA_SCHEMA)

        df = spark.sql(query, {"customer": customer_id})

        # check if customer exists
        if df.isEmpty():
            return json.dumps({"error": f"Customer with ID {customer_id} not found"})

        # convert to dict
        customer = df.first().asDict()

        # format response
        result = {
            "customer_id": customer["customer_id"],
            "customer_segment": customer["customer_segment"],
            "location": f"{customer['city']}, {customer['state']}",
            "registration_date": customer["registration_date"].strftime("%Y-%m-%d"),
            "customer_status": customer["customer_status"],
            "preferred_contact_method": customer["preferred_contact_method"],
            "loyalty_tier": customer["loyalty_tier"],
            "account_metrics": {
                "churn_risk_score": customer["churn_risk_score"],
                "customer_value_score": customer["customer_value_score"],
                "satisfaction_score": customer["satisfaction_score"],
            },
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error retrieving customer information: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


@uc_function(domain="account")
def get_customer_subscriptions(customer_id: str) -> str:
    """Retrieve all subscriptions for a customer including plan, device, and status details.

    Args:
        customer_id (str): The customer ID in the format 'CUS-XXXXX'

    Returns:
        str: JSON string with a list of all subscriptions for the customer
    """
    try:
        query = SUBSCRIPTIONS_INFO_QUERY.format(
            catalog=DATA_CATALOG, schema=DATA_SCHEMA
        )

        df = spark.sql(query, {"customer": customer_id})

        # check if subscriptions exist
        if df.isEmpty():
            return json.dumps(
                {"error": f"No subscriptions found for customer {customer_id}"}
            )

        # convert to list of dictionaries
        subscriptions_data = [row.asDict() for row in df.collect()]

        # format response
        subscriptions = []
        for sub_data in subscriptions_data:
            subscription = {
                "subscription_id": sub_data["subscription_id"],
                "status": sub_data["status"],
                "start_date": sub_data["subscription_start_date"].strftime("%Y-%m-%d"),
                "contract_length_months": sub_data["contract_length_months"],
                "monthly_charge": float(sub_data["monthly_charge"]),
                "autopay_enabled": sub_data["autopay_enabled"],
                "plan": {
                    "plan_id": sub_data["plan_id"],
                    "plan_name": sub_data["plan_name"],
                    "plan_type": sub_data["plan_type"],
                    "data_limit_gb": sub_data["data_limit_gb"],
                    "unlimited_calls": sub_data["unlimited_calls"],
                    "unlimited_texts": sub_data["unlimited_texts"],
                    "description": sub_data["plan_description"],
                },
            }

            # add device info if present
            if sub_data.get("device_id"):
                subscription["device"] = {"device_id": sub_data["device_id"]}

            subscriptions.append(subscription)

        return json.dumps(
            {"customer_id": customer_id, "subscriptions": subscriptions}, indent=2
        )

    except Exception as e:
        error_msg = f"Error retrieving subscription information: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
