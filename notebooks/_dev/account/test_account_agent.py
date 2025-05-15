# Databricks notebook source
# MAGIC %md
# MAGIC # Test Account Agent

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

from telco_support_agent.tools import initialize_tools
initialize_tools()

from mlflow.types.responses import ResponsesRequest
from telco_support_agent.agents.account import AccountAgent

# COMMAND ----------

account_agent = AccountAgent()

print(f"Agent type: {account_agent.agent_type}")
print(f"LLM endpoint: {account_agent.llm_endpoint}")
print(f"LLM parameters: {account_agent.llm_params}")
print(f"Number of tools: {len(account_agent.tools)}")

print("\nAvailable tools:")
for tool in account_agent.tools:
    if "function" in tool:
        print(f"- {tool['function']['name']}")

# COMMAND ----------

request = ResponsesRequest(
    input=[{"role": "user", "content": "How many plans do I have on my account? My ID is CUS-10601"}]
)

# COMMAND ----------

response = account_agent.predict(request)
if response and hasattr(response, 'output') and response.output:
    output_item = response.output[-1]
    if "content" in output_item and isinstance(output_item["content"], list):
        print(output_item["content"][0]["text"])
    elif "content" in output_item:
        print(output_item["content"])
    else:
        print("Unexpected response format:", output_item)
else:
    print("No response or empty response received")

# COMMAND ----------

test_queries = [
    "How many plans do I have on my account? My ID is CUS-10601",
    "What plan am I currently on? My ID is CUS-10601",
    "When did I create my account? My ID is CUS-10601",
    "Is my autopay enabled in my subscriptions? My ID is CUS-10601",
]

def test_query(query):
    print(f"\n=== TESTING QUERY: \"{query}\" ===\n")
    
    request = ResponsesRequest(
        input=[{"role": "user", "content": query}]
    )
    
    response = account_agent.predict(request)
    if response and hasattr(response, 'output') and response.output:
        output_item = response.output[-1]
        if "content" in output_item and isinstance(output_item["content"], list):
            print(output_item["content"][0]["text"])
        elif "content" in output_item:
            print(output_item["content"])
        else:
            print("Unexpected response format:", output_item)
    else:
        print("No response or empty response received")
    
    print("\n" + "="*80)
# COMMAND ----------

for query in test_queries:
    test_query(query)