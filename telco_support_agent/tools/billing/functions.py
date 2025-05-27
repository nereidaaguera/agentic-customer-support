"""UC functions for account-related operations."""

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()


def register_get_bill_info():
    """Register the get_bill_info UC function."""
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
                    'base_amount', cast(base_amount as string),
                    'additional_charges', cast(additional_charges as string),
                    'tax_amount', cast(tax_amount as string),
                    'total_amount', cast(total_amount as string),
                    'payment_amount', cast(payment_amount as string),
                    'payment_date', cast(payment_date as string),
                    'payment_method',payment_method,
                    'status',status
                    )
                  )
                )
                FROM telco_customer_support_prod.bronze.billing as billing_table
                WHERE billing_table.customer_id = customer_id_input -- avoid using param name that is the same as table column name
                  AND billing_date >= billing_start_date
                  AND billing_date < billing_end_date
        """
        client.create_function(sql_function_body=sql)
        print("Registered get_customer_info UC function")
    except Exception as e:
        print(f"Error registering get_customer_info: {str(e)}")


# call registration functions
register_get_bill_info()
