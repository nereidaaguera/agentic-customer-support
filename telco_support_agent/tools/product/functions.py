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
            COMMENT 'Provides comprehensive information about available subscription plans, including features, pricing, and benefits.
            Use this tool to answer questions about plan options, compare different plans, and assist users in selecting the plan that best meets their needs.'
            RETURN
            SELECT to_json(
              collect_list(named_struct(
                'plan_name', plan_name,
                'plan_type', plan_type,
                'monthly_price', monthly_price,
                'data_limit_gb', data_limit_gb,
                'unlimited_calls', unlimited_calls,
                'unlimited_texts', unlimited_texts,
                'plan_description', description
              ))
            )
            FROM telco_customer_support_dev.bronze.plans
            """
        client.create_function(sql_function_body=sql)
        print("Registered get_plans_info UC function")
    except Exception as e:
        print(f"Error registering get_plans_info: {str(e)}")


def register_devices_info():
    """Register the get_devices_info UC function."""
    try:
        sql = """
            CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_devices_info(
            )
            RETURNS STRING
            COMMENT 'Provides comprehensive information about available devices, including specifications, features, and capabilities.
            Use this tool to answer questions about device models, compare device characteristics, and support users in selecting the most suitable device for their needs.'
            RETURN
            SELECT to_json(
              collect_list(named_struct(
                'device_name', device_name,
                'manufacturer', manufacturer,
                'device_type', monthly_price,
                'retail_price', retail_price,
                'monthly_installment', monthly_installment,
                'storage_gb', storage_gb,
                'release_date', release_date,
                'is_5g_compatible',is_5g_compatible,
                'is_active', is_active
              )
              )
            )
            FROM telco_customer_support_dev.bronze.devices
            """
        client.create_function(sql_function_body=sql)
        print("Registered get_devices_info UC function")
    except Exception as e:
        print(f"Error registering get_devices_info: {str(e)}")


def register_promos_info():
    """Register the get_promotions_info UC function."""
    try:
        sql = """
            CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_promotions_info(
            )
            RETURNS STRING
            COMMENT 'Provides detailed information about current and past promotions, including promotion name, discount type and value, validity period, description, and active status.
            Use this tool to answer questions about available discounts, promotion periods, and to compare promotional offers.'
            RETURN
            SELECT to_json(
              collect_list(named_struct(
                'promotion_name', promo_name,
                'discount_type', discount_type,
                'discount_value', discount_value,
                'validity_period', concat('from ', start_date, ' to ', end_date),
                'validity_period_days', date_diff(end_date, start_date),
                'description', description,
                'is_active', is_active
              )
              )
            )
            FROM telco_customer_support_dev.bronze.promotions
            """
        client.create_function(sql_function_body=sql)
        print("Registered get_promotions_info UC function")
    except Exception as e:
        print(f"Error registering get_promotions_info: {str(e)}")


def register_customer_devices_info():
    """Register the get_customer_devices UC function."""
    try:
        sql = """
            CREATE OR REPLACE FUNCTION telco_customer_support_dev.agent.get_customer_devices(
                customer_id STRING COMMENT 'The customer ID in the format CUS-XXXXX'
            )
            RETURNS STRING
            COMMENT 'Returns detailed information about all devices linked to a customer’s subscription,
            including device types, statuses, and capacity. Use this tool to answer questions about which devices are active on a customer’s account,
            their subscription status, and their connection to particular services—distinct from tools that provide standalone device specifications.'
            RETURN
            SELECT to_json(
                    named_struct(
                      'customer_id', s.customer_id,
                      'devices', collect_list(
                         named_struct(
                        'subscription', s.subscription_id,
                        'subscription_status', s.status,
                        'device_name', device_name,
                        'manufacturer', manufacturer,
                        'device_type', monthly_price,
                        'retail_price', retail_price,
                        'monthly_installment', monthly_installment,
                        'storage_gb', storage_gb,
                        'release_date', release_date,
                        'is_5g_compatible',is_5g_compatible,
                        'is_active', is_active
                        )
                        )
                    )
                )
            FROM telco_customer_support_dev.bronze.subscriptions s
            JOIN telco_customer_support_dev.bronze.devices d ON s.device_id = d.device_id
            WHERE s.customer_id = customer_id
            GROUP BY s.customer_id
            LIMIT 1
            """
        client.create_function(sql_function_body=sql)
        print("Registered get_customer_devices UC function")
    except Exception as e:
        print(f"Error registering get_customer_devices: {str(e)}")


# call registration functions
register_plans_info()
register_devices_info()
register_promos_info()
register_customer_devices_info()
