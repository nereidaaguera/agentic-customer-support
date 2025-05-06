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

# COMMAND ----------

env = "dev"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product Data

# COMMAND ----------

product_gen = ProductGenerator(CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plans

# COMMAND ----------

plans_df = product_gen.generate_plans()
display(plans_df)

# COMMAND ----------

plans_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Devices

# COMMAND ----------

devices_df = product_gen.generate_devices()
display(devices_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promotions

# COMMAND ----------

promotions_df = product_gen.generate_promotions()
display(promotions_df)

# COMMAND ----------

print(f"Generated {promotions_df.count()} promotions")

# COMMAND ----------

product_gen.save_to_delta(plans_df, f"telco_customer_support_{env}.bronze.plans")
product_gen.save_to_delta(devices_df, f"telco_customer_support_{env}.bronze.devices")
product_gen.save_to_delta(promotions_df, f"telco_customer_support_{env}.bronze.promotions")

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

customers_df = customer_gen.generate_customers()
display(customers_df)

# COMMAND ----------

print(f"Generated {customers_df.count()} customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subscriptions

# COMMAND ----------

subscriptions_df = customer_gen.generate_subscriptions(
    plans_df=plans_df,
    devices_df=devices_df,
    promotions_df=promotions_df,
    customers_df=customers_df
)
display(subscriptions_df)

# COMMAND ----------

print(f"Generated {subscriptions_df.count()} subscriptions")

# COMMAND ----------

customer_gen.save_to_delta(customers_df, f"telco_customer_support_{env}.bronze.customers")
customer_gen.save_to_delta(subscriptions_df, f"telco_customer_support_{env}.bronze.subscriptions")

# COMMAND ----------

print(f"Customers: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.customers').collect()[0][0]}")
print(f"Subscriptions: {spark.sql('SELECT COUNT(*) FROM telco_customer_support_dev.bronze.subscriptions').collect()[0][0]}")

# COMMAND ----------

# sample validation query - customers with subscription counts
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

billing_df = billing_gen.generate_billing(subscriptions_df)
display(billing_df)

# COMMAND ----------

print(f"Generated {billing_df.count()} billing records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Usage Records

# COMMAND ----------

usage_df = billing_gen.generate_usage(subscriptions_df)
display(usage_df)

# COMMAND ----------

print(f"Generated {usage_df.count()} usage records")

# COMMAND ----------

billing_gen.save_to_delta(billing_df, f"telco_customer_support_{env}.bronze.billing")
billing_gen.save_to_delta(usage_df, f"telco_customer_support_{env}.bronze.usage")

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
# MAGIC ## Knowledge Base Data

# COMMAND ----------

knowledge_gen = KnowledgeGenerator(CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Knowledge Base Articles

# COMMAND ----------

kb_df = knowledge_gen.generate_kb_articles()
display(kb_df)

# COMMAND ----------

print(f"Generated {kb_df.count()} knowledge base articles")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Support Tickets

# COMMAND ----------

tickets_df = knowledge_gen.generate_tickets(customers_df, subscriptions_df)
display(tickets_df)

# COMMAND ----------

print(f"Generated {tickets_df.count()} support tickets")

# COMMAND ----------

knowledge_gen.save_to_delta(kb_df, f"telco_customer_support_{env}.bronze.knowledge_base")
knowledge_gen.save_to_delta(tickets_df, f"telco_customer_support_{env}.bronze.support_tickets")

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
