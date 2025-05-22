"""UC functions for product-related operations."""

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()


def register_plans_info():
    """Register the get_plans_info UC function."""
    try:
        sql = """
            CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_plans_info(
            )
            RETURNS STRING
            COMMENT 'Provides detailed information about the plans available. Use this tool to address queries about plans and reasoning about plan comparisons.'
            RETURN
            SELECT to_json(
              named_struct(
                'plan_name', plan_name,
                'plan_type', plan_type,
                'monthly_price', monthly_price,
                'data_limit_gb', data_limit_gb,
                'customer_status', customer_status,
                'unlimited_calls', unlimited_calls,
                'unlimited_texts', unlimited_texts,
                'plan_description', description
              )
            )
            FROM telco_customer_support_prod.bronze.plans
            """
        client.create_function(sql_function_body=sql)
        print("Registered get_customer_info UC function")
    except Exception as e:
        print(f"Error registering get_customer_info: {str(e)}")


# call registration functions
register_plans_info()
