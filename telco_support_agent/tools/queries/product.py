# Query to get general plans information
PLANS_INFO_QUERY = """
SELECT
    p.plan_name,
    p.plan_type,
    p.monthly_price,
    p.data_limit_gb,
    p.unlimited_calls,
    p.unlimited_texts,
    p.description as plan_description
FROM {catalog}.{schema}.plans p
"""
