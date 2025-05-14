"""SQL queries for account-related operations."""

# Query to get basic customer account information
ACCOUNT_INFO_QUERY = """
SELECT * 
FROM {catalog}.{schema}.customers 
WHERE customer_id = customer
LIMIT 1
"""

# Query to get subscription information for a customer
SUBSCRIPTIONS_INFO_QUERY = """
SELECT 
    s.*,
    p.plan_name,
    p.plan_type,
    p.monthly_price,
    p.data_limit_gb,
    p.unlimited_calls,
    p.unlimited_texts,
    p.description as plan_description
FROM {catalog}.{schema}.subscriptions s
JOIN {catalog}.{schema}.plans p ON s.plan_id = p.plan_id
WHERE s.customer_id = customer
"""
