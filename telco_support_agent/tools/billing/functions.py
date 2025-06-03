"""UC functions for billing-related operations."""

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()


def register_get_billing_info():
    """Register the get_billing_info UC function."""
    try:
        sql = """
        CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_billing_info(
          -- customer_id_input STRING COMMENT 'The customer ID in the format CUS-XXXXX',
          -- billing_start_date_input STRING DEFAULT date_format(date_trunc("month", current_date()), "yyyy-MM-dd") COMMENT 'The billing start date in YYYY-MM-DD format. Defaults to first day of current month.',
          -- billing_end_date_input STRING DEFAULT date_format(date_trunc("month", add_months(current_date(), 1)), "yyyy-MM-dd") COMMENT 'The billing end date in YYYY-MM-DD format. Defaults to first day of next month.',
          -- additional_charges_input FLOAT DEFAULT NULL COMMENT 'Filter on additional_charges. If NULL, only rows with non-NULL additional_charges are included.',
          -- total_amount_input FLOAT DEFAULT NULL COMMENT 'Filter on total_amount. If NULL, only rows with non-NULL total_amount are included.',
          -- status_input STRING DEFAULT NULL COMMENT 'Billing status. Possible values: Paid, Unpaid, Late, Partial. Defaults to Paid. If NULL, return rows of all statuses.'
          customer_id_input STRING COMMENT 'The customer ID in the format CUS-XXXXX',
          billing_start_date_input STRING COMMENT 'The billing start date in YYYY-MM-DD format. Defaults to first day of current month.',
          billing_end_date_input STRING COMMENT 'The billing end date in YYYY-MM-DD format. Defaults to first day of next month.',
          additional_charges_input FLOAT COMMENT 'Filter on additional_charges. If NULL, only rows with non-NULL additional_charges are included. Default is NULL.',
          total_amount_input FLOAT COMMENT 'Filter on total_amount. If NULL, only rows with non-NULL total_amount are included. Default is NULL.',
          status_input STRING COMMENT 'Billing status. Possible values: Paid, Unpaid, Late, Partial. Defaults to Paid. If NULL, return rows of all statuses. Default is NULL.'
        )
        RETURNS STRING
        COMMENT 'Retrieves all columns of the billing table for all rows for a customer within the specified date range, with optional filters for additional_charges, total_amount, and status. \nExample usage:\n SELECT telco_customer_support_dev.agent.get_billing_info("CUS-10601","2025-06-01","2025-06-01",Null,Null,"Paid")' -- use single quote for comment and double quote for strings variables inside the comment
        RETURN
        SELECT to_json(
                collect_list(
                  named_struct(
                    'customer_id', customer_id,
                    'billing_id', billing_id,
                    'subscription_id', subscription_id,
                    'billing_date', cast(billing_date as string),
                    'base_amount', cast(base_amount as string),
                    'additional_charges', cast(additional_charges as string),
                    'tax_amount', cast(tax_amount as string),
                    'total_amount', cast(total_amount as string),
                    'payment_amount', cast(payment_amount as string),
                    'payment_date', cast(payment_date as string),
                    'payment_method', payment_method,
                    'status', status
                  )
                )
              )
        FROM telco_customer_support_prod.bronze.billing AS billing_table
        WHERE billing_table.customer_id = customer_id_input
          AND billing_table.billing_date >= billing_start_date_input
          AND billing_table.billing_date < billing_end_date_input
          AND (
                (additional_charges_input IS NULL AND billing_table.additional_charges IS NOT NULL)
                OR
                (additional_charges_input IS NOT NULL AND ABS(billing_table.additional_charges - additional_charges_input) <= 1) -- use approx within $1
              )
          AND (
                (total_amount_input IS NULL AND billing_table.total_amount IS NOT NULL)
                OR
                (total_amount_input IS NOT NULL AND ABS(billing_table.total_amount - total_amount_input) <= 1) -- use approx within $1
              )
          AND (
                (status_input IS NULL AND billing_table.status IS NOT NULL)
                OR
                (status_input IS NOT NULL AND billing_table.status = status_input)
          )
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
          customer STRING COMMENT 'The customer ID in the format CUS-XXXXX',
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
        WHERE s.customer_id = customer
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
