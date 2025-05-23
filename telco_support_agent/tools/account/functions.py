"""UC functions for account-related operations."""

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()


def register_customer_info():
    """Register the get_customer_info UC function."""
    try:
        sql = """
        CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_customer_info(
          customer_id STRING COMMENT 'The customer ID in the format CUS-XXXXX'
        )
        RETURNS STRING
        COMMENT 'Retrieves basic customer information including profile data, status, and account metrics'
        RETURN
        SELECT to_json(
          named_struct(
            'customer_id', customer_id,
            'customer_segment', customer_segment,
            'location', concat(city, ', ', state),
            'registration_date', cast(registration_date as string),
            'customer_status', customer_status,
            'preferred_contact_method', preferred_contact_method,
            'loyalty_tier', loyalty_tier,
            'account_metrics', named_struct(
              'churn_risk_score', cast(churn_risk_score as string),
              'customer_value_score', cast(customer_value_score as string),
              'satisfaction_score', cast(satisfaction_score as string)
            )
          )
        )
        FROM telco_customer_support_dev.bronze.customers
        WHERE customer_id = customer_id
        LIMIT 1
        """
        client.create_function(sql_function_body=sql)
        print("Registered get_customer_info UC function")
    except Exception as e:
        print(f"Error registering get_customer_info: {str(e)}")


def register_customer_subscriptions():
    """Register the get_customer_subscriptions UC function."""
    try:
        sql = """
        CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_customer_subscriptions(
          customer_id STRING COMMENT 'The customer ID in the format CUS-XXXXX'
        )
        RETURNS STRING
        COMMENT 'Retrieves all active subscriptions for a customer including plan details, status, and device information'
        RETURN
        (
          SELECT to_json(
            named_struct(
              'customer_id', s.customer_id,
              'subscriptions', collect_list(
                named_struct(
                  'subscription_id', s.subscription_id,
                  'status', s.status,
                  'start_date', cast(s.subscription_start_date as string),
                  'contract_length_months', cast(s.contract_length_months as string),
                  'monthly_charge', cast(s.monthly_charge as string),
                  'autopay_enabled', cast(s.autopay_enabled as string),
                  'plan', named_struct(
                    'plan_id', p.plan_id,
                    'plan_name', p.plan_name,
                    'plan_type', p.plan_type,
                    'data_limit_gb', cast(p.data_limit_gb as string),
                    'unlimited_calls', cast(p.unlimited_calls as string),
                    'unlimited_texts', cast(p.unlimited_texts as string)
                  )
                )
              )
            )
          )
          FROM telco_customer_support_dev.bronze.subscriptions s
          JOIN telco_customer_support_dev.bronze.plans p ON s.plan_id = p.plan_id
          WHERE s.customer_id = customer_id
          GROUP BY s.customer_id
          LIMIT 1
        )
        """
        client.create_function(sql_function_body=sql)
        print("Registered get_customer_subscriptions UC function")
    except Exception as e:
        print(f"Error registering get_customer_subscriptions: {str(e)}")


# call registration functions
register_customer_info()
register_customer_subscriptions()
