"""UC functions for billing-related operations."""

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()


def register_get_billing_info():
    """Register the get_billing_info UC function."""
    try:
        sql = """
        CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_billing_info(
          customer_id_input STRING COMMENT 'The customer ID in the format CUS-XXXXX', -- avoid using param name that is the same as table column name
          billing_start_date STRING COMMENT 'The billing_start_date in the format YYYY-MM-DD',
          billing_end_date STRING COMMENT 'The billing_end_date in the format YYYY-MM-DD'
        )
        RETURNS STRING
        COMMENT 'Retrieves all columns of the billing table for all rows for a customer within the specified date range'
        RETURN
        SELECT to_json(
                collect_list(
                -- use collect list to aggregate rows
                  named_struct(
                    'customer_id', customer_id,
                    'billing_id', billing_id,
                    'subscription_id', subscription_id,
                    'billing_date', cast(billing_date as string),
                    'due_date', cast(due_date as string),
                    'base_amount', cast(base_amount as string),
                    'additional_charges', cast(additional_charges as string),
                    'tax_amount', cast(tax_amount as string),
                    'total_amount', cast(total_amount as string),
                    'payment_amount', cast(payment_amount as string),
                    'payment_date', cast(payment_date as string),
                    'payment_method', payment_method,
                    'status', status,
                    'billing_cycle', billing_cycle
                    )
                  )
                )
                FROM telco_customer_support_dev.bronze.billing as billing_table
                WHERE billing_table.customer_id = customer_id_input -- avoid using param name that is the same as table column name
                  AND billing_date >= billing_start_date
                  AND billing_date < billing_end_date
        """
        client.create_function(sql_function_body=sql)
        print("Registered get_billing_info UC function")
    except Exception as e:
        print(f"Error registering get_billing_info: {str(e)}")


def register_get_usage_info():
    """Register the get_usage_info UC function for usage details."""
    try:
        sql = """
        CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_usage_info(
          customer_id_input STRING COMMENT 'The customer ID in the format CUS-XXXXX',
          usage_start_date STRING COMMENT 'The usage start date in the format YYYY-MM-DD',
          usage_end_date STRING COMMENT 'The usage end date in the format YYYY-MM-DD'
        )
        RETURNS STRING
        COMMENT 'Retrieves usage data for a customer within the specified date range including data, voice, and SMS usage'
        RETURN
        SELECT to_json(
          named_struct(
            'customer_id', u.subscription_id,
            'total_usage', named_struct(
              'data_usage_gb', cast(sum(u.data_usage_mb) / 1024.0 as decimal(10,2)),
              'voice_minutes', cast(sum(u.voice_minutes) as int),
              'sms_count', cast(sum(u.sms_count) as int)
            ),
            'daily_usage', collect_list(
              named_struct(
                'date', cast(u.date as string),
                'data_usage_mb', cast(u.data_usage_mb as decimal(10,2)),
                'voice_minutes', cast(u.voice_minutes as int),
                'sms_count', cast(u.sms_count as int),
                'billing_cycle', u.billing_cycle
              )
            )
          )
        )
        FROM telco_customer_support_dev.bronze.usage u
        JOIN telco_customer_support_dev.bronze.subscriptions s ON u.subscription_id = s.subscription_id
        WHERE s.customer_id = customer_id_input
          AND u.date >= usage_start_date
          AND u.date < usage_end_date
        GROUP BY u.subscription_id
        LIMIT 1
        """
        client.create_function(sql_function_body=sql)
        print("Registered get_usage_info UC function")
    except Exception as e:
        print(f"Error registering get_usage_info: {str(e)}")


# call registration functions
register_get_billing_info()
register_get_usage_info()
