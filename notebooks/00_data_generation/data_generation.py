# Databricks notebook source
# MAGIC %md
# MAGIC # Data Generation

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from telco_support_agent.data.config import CONFIG
from telco_support_agent.data.generators.billing import BillingGenerator
from telco_support_agent.data.generators.customers import CustomerGenerator
from telco_support_agent.data.generators.knowledge_base import KnowledgeGenerator
from telco_support_agent.data.generators.products import ProductGenerator
from pyspark.sql import DataFrame

# COMMAND ----------

env = "dev"

# Configure which tables to generate (set to True for tables to generate)
generate_config = {
    
    # Set to True to regenerate all tables
    "all": False,

    # Product data
    "plans": True,
    "devices": True,
    "promotions": True,
    
    # Customer data
    "customers": True,
    "subscriptions": True,
    
    # Billing data
    "billing": True,
    "usage": True,
    
    # Knowledge base data
    "kb_articles": False,
    "support_tickets": False,
    
}

# COMMAND ----------

def should_generate(table_name: str) -> bool:
    return generate_config["all"] or generate_config.get(table_name, False)

def load_existing_table(table_name: str) -> DataFrame:
    return spark.table(f"telco_customer_support_{env}.bronze.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product Data

# COMMAND ----------

product_gen = ProductGenerator(CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plans

# COMMAND ----------

if should_generate("plans"):
    plans_df = product_gen.generate_plans()
    display(plans_df)
    print(f"Generated {plans_df.count()} plans")
    product_gen.save_to_delta(plans_df, f"telco_customer_support_{env}.bronze.plans")
else:
    plans_df = load_existing_table("plans")
    print("Using existing plans data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Devices

# COMMAND ----------

if should_generate("devices"):
    devices_df = product_gen.generate_devices()
    display(devices_df)
    print(f"Generated {devices_df.count()} devices")
    product_gen.save_to_delta(devices_df, f"telco_customer_support_{env}.bronze.devices")
else:
    devices_df = load_existing_table("devices")
    print("Using existing devices data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promotions

# COMMAND ----------

if should_generate("promotions"):
    promotions_df = product_gen.generate_promotions()
    display(promotions_df)
    print(f"Generated {promotions_df.count()} promotions")
    product_gen.save_to_delta(promotions_df, f"telco_customer_support_{env}.bronze.promotions")
else:
    promotions_df = load_existing_table("promotions")
    print("Using existing promotions data")

# COMMAND ----------

print(f"Plans: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.plans').collect()[0][0]}")
print(f"Devices: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.devices').collect()[0][0]}")
print(f"Promotions: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.promotions').collect()[0][0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer Data

# COMMAND ----------

customer_gen = CustomerGenerator(CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customers

# COMMAND ----------

if should_generate("customers"):
    customers_df = customer_gen.generate_customers()
    display(customers_df)
    print(f"Generated {customers_df.count()} customers")
    customer_gen.save_to_delta(customers_df, f"telco_customer_support_{env}.bronze.customers")
else:
    customers_df = load_existing_table("customers")
    print("Using existing customers data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subscriptions

# COMMAND ----------

if should_generate("subscriptions"):
    subscriptions_df = customer_gen.generate_subscriptions(
        plans_df=plans_df,
        devices_df=devices_df,
        promotions_df=promotions_df,
        customers_df=customers_df
    )
    display(subscriptions_df)
    print(f"Generated {subscriptions_df.count()} subscriptions")
    customer_gen.save_to_delta(subscriptions_df, f"telco_customer_support_{env}.bronze.subscriptions")
else:
    subscriptions_df = load_existing_table("subscriptions")
    print("Using existing subscriptions data")

# COMMAND ----------

print(f"Customers: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.customers').collect()[0][0]}")
print(f"Subscriptions: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.subscriptions').collect()[0][0]}")

# COMMAND ----------

# Sample validation query - customers with subscription counts
display(spark.sql("""
SELECT 
  c.customer_id, 
  c.customer_segment,
  c.city,
  c.state,
  COUNT(s.subscription_id) as subscription_count,
  SUM(s.monthly_charge) as total_monthly_charge
FROM telco_customer_support_dev.bronze.customers c
LEFT JOIN telco_customer_support_dev.bronze.subscriptions s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.customer_segment, c.city, c.state
ORDER BY total_monthly_charge DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Billing Data

# COMMAND ----------

billing_gen = BillingGenerator(CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Billing Records

# COMMAND ----------

if should_generate("billing"):
    billing_df = billing_gen.generate_billing(subscriptions_df)
    display(billing_df)
    print(f"Generated {billing_df.count()} billing records")
    billing_gen.save_to_delta(billing_df, f"telco_customer_support_{env}.bronze.billing")
else:
    billing_df = load_existing_table("billing")
    print("Using existing billing data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Usage Records

# COMMAND ----------

if should_generate("usage"):
    usage_df = billing_gen.generate_usage(subscriptions_df)
    display(usage_df)
    print(f"Generated {usage_df.count()} usage records")
    billing_gen.save_to_delta(usage_df, f"telco_customer_support_{env}.bronze.usage")
else:
    usage_df = load_existing_table("usage")
    print("Using existing usage data")

# COMMAND ----------

print(f"Billing Records: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.billing').collect()[0][0]}")
print(f"Usage Records: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.usage').collect()[0][0]}")

# COMMAND ----------

# Sample query - total monthly revenue
display(spark.sql("""
SELECT 
  billing_cycle,
  COUNT(*) as bill_count,
  SUM(total_amount) as total_billed,
  SUM(CASE WHEN status = 'Paid' THEN payment_amount ELSE 0 END) as total_collected,
  SUM(CASE WHEN status = 'Paid' THEN payment_amount ELSE 0 END) / SUM(total_amount) as collection_rate
FROM telco_customer_support_dev.bronze.billing
GROUP BY billing_cycle
ORDER BY billing_cycle
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross-Reference Example: Devices, Promotions, and Billing
# MAGIC 
# MAGIC Demonstrates how device SKUs connect with promotions and billing records.

# COMMAND ----------

# Simple cross-reference query showing device-promotion-billing relationships
display(spark.sql("""
-- Find customers with premium devices and their associated promotions and billing records
SELECT 
  d.device_id,
  d.manufacturer,
  d.device_name,
  p.promo_id,
  p.promo_name,
  p.discount_type,
  p.discount_value,
  b.billing_id,
  b.billing_cycle,
  b.base_amount,
  b.total_amount,
  b.status AS payment_status
FROM telco_customer_support_dev.bronze.devices d
JOIN telco_customer_support_dev.bronze.subscriptions s ON d.device_id = s.device_id
LEFT JOIN telco_customer_support_dev.bronze.promotions p ON s.promo_id = p.promo_id
JOIN telco_customer_support_dev.bronze.billing b ON s.subscription_id = b.subscription_id
WHERE 
  (d.manufacturer = 'Samsung' AND d.device_name LIKE '%S25%')
  OR (d.manufacturer = 'Apple' AND d.device_name LIKE '%iPhone 16%')
  OR (d.manufacturer = 'Google' AND d.device_name LIKE '%Pixel 9%')
  AND s.status = 'Active'
  AND b.billing_cycle >= '2025-01'
ORDER BY 
  d.manufacturer,
  d.device_name,
  b.billing_cycle DESC
LIMIT 20
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge Base Data

# COMMAND ----------

knowledge_gen = KnowledgeGenerator(CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Knowledge Base Articles

# COMMAND ----------

if should_generate("kb_articles"):
    kb_df = knowledge_gen.generate_kb_articles()
    display(kb_df)
    print(f"Generated {kb_df.count()} knowledge base articles")
    knowledge_gen.save_to_delta(kb_df, f"telco_customer_support_{env}.bronze.knowledge_base")
else:
    kb_df = load_existing_table("knowledge_base")
    print("Using existing knowledge base data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Support Tickets

# COMMAND ----------

if should_generate("support_tickets"):
    tickets_df = knowledge_gen.generate_tickets(customers_df, subscriptions_df)
    display(tickets_df)
    print(f"Generated {tickets_df.count()} support tickets")
    knowledge_gen.save_to_delta(tickets_df, f"telco_customer_support_{env}.bronze.support_tickets")
else:
    tickets_df = load_existing_table("support_tickets")
    print("Using existing support tickets data")

# COMMAND ----------

print(f"Knowledge Base Articles: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.knowledge_base').collect()[0][0]}")
print(f"Support Tickets: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.support_tickets').collect()[0][0]}")

# COMMAND ----------

# Sample query - Top knowledge base categories and content types
display(spark.sql("""
SELECT 
  category,
  content_type,
  COUNT(*) as article_count
FROM telco_customer_support_dev.bronze.knowledge_base
GROUP BY category, content_type
ORDER BY article_count DESC
"""))

# COMMAND ----------

# Sample query - Support ticket distribution by category and status
display(spark.sql("""
SELECT 
  category,
  status,
  COUNT(*) as ticket_count,
  AVG(DATEDIFF(resolved_date, created_date)) as avg_days_to_resolve
FROM telco_customer_support_dev.bronze.support_tickets
WHERE status IN ('Resolved', 'Closed')
GROUP BY category, status
ORDER BY category, status
"""))