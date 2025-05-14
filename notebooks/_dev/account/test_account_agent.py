# Databricks notebook source
# MAGIC %md
# MAGIC # Test Account Agent Class

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt -q
# MAGIC %pip install --upgrade "databricks-connect==16.3.*"

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

from mlflow.types.responses import ResponsesRequest

from telco_support_agent.agents.account import AccountAgent

# COMMAND ----------

account_agent = AccountAgent()

print(f"Agent type: {account_agent.agent_type}")
print(f"LLM endpoint: {account_agent.llm_endpoint}")
print(f"LLM parameters: {account_agent.llm_params}")

# COMMAND ----------

request = ResponsesRequest(
    input=[{"role": "user", "content": "How many plans do I have on my account? My ID is CUS-10601"}]
)

# COMMAND ----------

response = account_agent.predict(request)

# COMMAND ----------

response.output[-1].content[0]['text']

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
    print(response.output[-1].content[0]['text'])
    print("\n" + "="*80)

# COMMAND ----------

for query in test_queries:
    test_query(query)
