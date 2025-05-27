# Databricks notebook source
# MAGIC %md
# MAGIC # Test Billing Agent

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

root_path = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
print(f"Root path: {root_path}")

if root_path:
    sys.path.append(root_path)
    print(f"Added {root_path} to Python path")

# COMMAND ----------

# init tools - will register UC functions if needed
from telco_support_agent.tools import initialize_tools

from mlflow.types.responses import ResponsesAgentRequest

from telco_support_agent.agents.billing import BillingAgent

# COMMAND ----------

# init agent
billing_agent = BillingAgent()

print(f"Agent type: {billing_agent.agent_type}")
print(f"LLM endpoint: {billing_agent.llm_endpoint}")
print(f"LLM parameters: {billing_agent.llm_params}")
print(f"Number of tools: {len(billing_agent.tools)}")

print("\nAvailable tools:")
for tool in billing_agent.tools:
    if "function" in tool:
        print(f"- {tool['function']['name']}")

# COMMAND ----------

print("Initializing tools for billing agent...")
results = initialize_tools(domains=["billing"])

print("\nFunction initialization status:")
for domain, functions in results.items():
    print(f"\nDomain: {domain}")
    for func_name, status in functions.items():
        status_str = "✅ Available" if status else "❌ Unavailable"
        print(f"  - {func_name}: {status_str}")

if any(not all(functions.values()) for functions in results.values()):
    print("\nWARNING: Some functions could not be initialized")
    print("Tests might fail without the necessary UC functions")
else:
    print("\nAll required functions are available")

# COMMAND ----------

def test_query(query):
    print(f"\n=== TESTING QUERY: \"{query}\" ===\n")

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}]
    )

    try:
        response = billing_agent.predict(request)
        if response and hasattr(response, 'output') and response.output:
            output_item = response.output[-1]
            if "content" in output_item and isinstance(output_item["content"], list):
                print(output_item["content"][0]["text"])
            elif "content" in output_item:
                print(output_item["content"])
            else:
                print("Response:", output_item)
        else:
            print("No response or empty response received")
    except Exception as e:
        print(f"Error processing query: {e}")

    print("\n" + "="*80)

# COMMAND ----------

test_query("How much is my bill for the total amount charge for my billing of 2025-04-01 to 2025-04-30? My ID is CUS-10601")

# COMMAND ----------

test_queries = [
    "Is there an additional charge for my billing of 2025-04-01 to 2025-04-30? My ID is CUS-10601",
    "Is the total charge for my billing of Jun 2025 higher than the total charge for my billing of May 2025? My ID is CUS-10601",
    "Is there a unpaid amount for my billing of 2025-04-01 to 2025-04-30? My ID is CUS-10601",
]

for query in test_queries:
    test_query(query)
