"""UC functions for product-related operations."""

from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

from telco_support_agent.agents import UCConfig
from telco_support_agent.utils.logging_utils import get_logger
from telco_support_agent.utils.uc_permissions import grant_function_permissions

logger = get_logger(__name__)

client = DatabricksFunctionClient()
workspace_client = WorkspaceClient()


def register_plans_info(uc_config: UCConfig):
    """Register the get_plans_info UC function."""
    function_name = uc_config.get_uc_function_name("get_plans_info")

    try:
        sql = f"""
            CREATE OR REPLACE FUNCTION {function_name}(
            )
            RETURNS STRING
            COMMENT 'Provides comprehensive information about available subscription plans, including features, pricing, and benefits. Use this tool to answer questions about plan options, compare different plans, and assist users in selecting the plan that best meets their needs.'
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
            FROM {uc_config.get_uc_table_name("plans")}
            """

        client.create_function(sql_function_body=sql)
        print(f"Registered {function_name} UC function")

        # grant permissions
        if grant_function_permissions(function_name, uc_config, workspace_client):
            print(f"Granted permissions on {function_name}")
        else:
            print(
                f"Warning: Some permissions may not have been granted on {function_name}"
            )

    except Exception as e:
        print(f"Error registering get_plans_info: {str(e)}")


def register_devices_info(uc_config: UCConfig):
    """Register the get_devices_info UC function."""
    function_name = uc_config.get_uc_function_name("get_devices_info")

    try:
        sql = f"""
            CREATE OR REPLACE FUNCTION {function_name}(
            )
            RETURNS STRING
            COMMENT 'Provides comprehensive information about available devices, including specifications, features, and capabilities. Use this tool to answer questions about device models, compare device characteristics, and support users in selecting the most suitable device for their needs.'
            RETURN
            SELECT to_json(
              collect_list(named_struct(
                'device_name', device_name,
                'manufacturer', manufacturer,
                'device_type', device_type,
                'retail_price', retail_price,
                'monthly_installment', monthly_installment,
                'storage_gb', storage_gb,
                'release_date', release_date,
                'is_5g_compatible', is_5g_compatible,
                'is_active', is_active
              )
              )
            )
            FROM {uc_config.get_uc_table_name("devices")}
            """

        client.create_function(sql_function_body=sql)
        print(f"Registered {function_name} UC function")

        # grant permissions
        if grant_function_permissions(function_name, uc_config, workspace_client):
            print(f"Granted permissions on {function_name}")
        else:
            print(
                f"Warning: Some permissions may not have been granted on {function_name}"
            )

    except Exception as e:
        print(f"Error registering get_devices_info: {str(e)}")


def register_promos_info(uc_config: UCConfig):
    """Register the get_promotions_info UC function."""
    function_name = uc_config.get_uc_function_name("get_promotions_info")

    try:
        sql = f"""
            CREATE OR REPLACE FUNCTION {function_name}(
            )
            RETURNS STRING
            COMMENT 'Provides detailed information about current and past promotions, including promotion name, discount type and value, validity period, description, and active status. Use this tool to answer questions about available discounts, promotion periods, and to compare promotional offers.'
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
            FROM {uc_config.get_uc_table_name("promotions")}
            """

        client.create_function(sql_function_body=sql)
        print(f"Registered {function_name} UC function")

        # grant permissions
        if grant_function_permissions(function_name, uc_config, workspace_client):
            print(f"Granted permissions on {function_name}")
        else:
            print(
                f"Warning: Some permissions may not have been granted on {function_name}"
            )

    except Exception as e:
        print(f"Error registering get_promotions_info: {str(e)}")


def register_customer_devices_info(uc_config: UCConfig):
    """Register the get_customer_devices UC function."""
    function_name = uc_config.get_uc_function_name("get_customer_devices")

    try:
        sql = f"""
            CREATE OR REPLACE FUNCTION {function_name}(
                customer STRING COMMENT 'The customer ID in the format CUS-XXXXX'
            )
            RETURNS STRING
            COMMENT 'Returns detailed information about all devices linked to a customer subscription, including device types, statuses, and capacity. Use this tool to answer questions about which devices are active on a customer account, their subscription status, and their connection to particular services—distinct from tools that provide standalone device specifications.'
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
                        'device_type', device_type,
                        'retail_price', retail_price,
                        'monthly_installment', monthly_installment,
                        'storage_gb', storage_gb,
                        'release_date', release_date,
                        'is_5g_compatible', is_5g_compatible,
                        'is_active', is_active
                        )
                        )
                    )
                )
            FROM {uc_config.get_uc_table_name("subscriptions")} s
            JOIN {uc_config.get_uc_table_name("devices")} d ON s.device_id = d.device_id
            WHERE s.customer_id = customer
            GROUP BY s.customer_id
            LIMIT 1
            """

        client.create_function(sql_function_body=sql)
        print(f"Registered {function_name} UC function")

        # grant permissions
        if grant_function_permissions(function_name, uc_config, workspace_client):
            print(f"Granted permissions on {function_name}")
        else:
            print(
                f"Warning: Some permissions may not have been granted on {function_name}"
            )

    except Exception as e:
        print(f"Error registering get_customer_devices: {str(e)}")


# Auto-registration with default UC config
# Note: This should ideally be called explicitly with proper UC config
if __name__ == "__main__":
    # Default UC config for auto-registration
    uc_config = UCConfig(
        catalog="telco_customer_support_prod",
        agent_schema="agent",
        data_schema="gold",
        model_name="telco_customer_support_agent",
    )
    register_plans_info(uc_config)
    register_devices_info(uc_config)
    register_promos_info(uc_config)
    register_customer_devices_info(uc_config)
