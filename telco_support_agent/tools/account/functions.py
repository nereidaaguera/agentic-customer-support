"""UC functions for account-related operations."""

from telco_support_agent.tools.registry import register_sql_function
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Register customer info SQL function
register_sql_function(
    """
    CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_customer_info(
      customer_id STRING COMMENT 'The customer ID in the format CUS-XXXXX'
    )
    RETURNS STRING
    COMMENT 'Retrieves basic customer information including profile data, status, and account metrics'
    RETURN
    SELECT CAST(
      MAP(
        'customer_id', customer_id,
        'customer_segment', customer_segment,
        'location', CONCAT(city, ', ', state),
        'registration_date', CAST(registration_date AS STRING),
        'customer_status', customer_status,
        'preferred_contact_method', preferred_contact_method,
        'loyalty_tier', loyalty_tier,
        'account_metrics', MAP(
          'churn_risk_score', CAST(churn_risk_score AS STRING),
          'customer_value_score', CAST(customer_value_score AS STRING),
          'satisfaction_score', CAST(satisfaction_score AS STRING)
        )
      ) AS STRING
    )
    FROM telco_customer_support_dev.bronze.customers
    WHERE customer_id = customer_id
    LIMIT 1
    """,
    domain="account",
)

# Register customer subscriptions SQL function
register_sql_function(
    """
    CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_customer_subscriptions(
      customer_id STRING COMMENT 'The customer ID in the format CUS-XXXXX'
    )
    RETURNS STRING
    COMMENT 'Retrieves all active subscriptions for a customer including plan details, status, and device information'
    RETURN
    SELECT CAST(
      MAP(
        'customer_id', s.customer_id,
        'subscriptions', COLLECT_LIST(
          MAP(
            'subscription_id', s.subscription_id,
            'status', s.status,
            'start_date', CAST(s.subscription_start_date AS STRING),
            'contract_length_months', CAST(s.contract_length_months AS STRING),
            'monthly_charge', CAST(s.monthly_charge AS STRING),
            'autopay_enabled', CAST(s.autopay_enabled AS STRING),
            'plan', MAP(
              'plan_id', p.plan_id,
              'plan_name', p.plan_name,
              'plan_type', p.plan_type,
              'data_limit_gb', CAST(p.data_limit_gb AS STRING),
              'unlimited_calls', CAST(p.unlimited_calls AS STRING),
              'unlimited_texts', CAST(p.unlimited_texts AS STRING)
            )
          )
        )
      ) AS STRING
    )
    FROM telco_customer_support_dev.bronze.subscriptions s
    JOIN telco_customer_support_dev.bronze.plans p ON s.plan_id = p.plan_id
    WHERE s.customer_id = customer_id
    GROUP BY s.customer_id
    """,
    domain="account",
)
