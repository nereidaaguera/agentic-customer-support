# Databricks notebook source
# MAGIC %md
# MAGIC # Test Disabling Tools Functionality
# MAGIC
# MAGIC 1. Test usage queries with normal supervisor
# MAGIC 2. Test usage queries with get_usage_info disabled

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(root_path)

from mlflow.types.responses import ResponsesAgentRequest
from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.tools import initialize_tools

# COMMAND ----------

results = initialize_tools()

success_count = sum(1 for domain in results.values() for status in domain.values() if status)
total_count = sum(len(domain) for domain in results.values())
print(f"Initialized {success_count}/{total_count} UC functions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Supervisor Agents

# COMMAND ----------

# Normal supervisor - all tools enabled
normal_supervisor_agent = SupervisorAgent()
print("Created normal supervisor (all tools enabled)")

# Supervisor with get_usage_info disabled
disable_usage_tool_agent = SupervisorAgent(disable_tools=["get_usage_info"])
print("Created supervisor with get_usage_info disable")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Usage Queries

# COMMAND ----------

usage_queries = [
    "Show me my billing details for March 2025",
    "What are the charges on my bill from 2025-04-01 to 2025-04-30?",
    "What's my usage breakdown for the past 3 months?"
]

test_customer = "CUS-10001"

def test_query(supervisor, query, supervisor_name):
    """Test a single query with the given supervisor."""
    print(f"\n--- {supervisor_name.upper()} ---")
    print(f"Query: '{query}'")
    
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}],
        custom_inputs={"customer": test_customer}
    )
    
    response = supervisor.predict(request)
    
    for output_item in response.output:
        if hasattr(output_item, "content"):
            for content_item in output_item.content:
                if hasattr(content_item, "text"):
                    print(f"Response: {content_item.text}")
                    break
        elif isinstance(output_item, dict) and "content" in output_item:
            for content_item in output_item["content"]:
                if "text" in content_item:
                    print(f"Response: {content_item['text']}")
                    break
    
    if response.custom_outputs and "routing" in response.custom_outputs:
        routing_info = response.custom_outputs["routing"]
        print(f"Routed to: {routing_info.get('agent_type')} agent")
        if "disable_tools" in routing_info:
            print(f"disable tools: {routing_info['disable_tools']}")

# COMMAND ----------

print("="*80)
print("TESTING DISABLE TOOLS FUNCTIONALITY")
print("="*80)

for query in usage_queries:
    print(f"\n{'='*60}")
    print(f"TESTING: {query}")
    print('='*60)
    
    test_query(normal_supervisor_agent, query, "Normal Supervisor")
    
    print("\n" + "-"*40)
    
    test_query(disable_usage_tool_agent, query, "Disable Usage Tool")
    
    print("\n")