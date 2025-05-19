# Databricks notebook source
# MAGIC %md
# MAGIC # Test Supervisor Agent Class
# MAGIC
# MAGIC This notebook tests the SupervisorAgent's ability to:
# MAGIC 1. Route queries to the appropriate sub-agent
# MAGIC 2. Handle account-related queries (currently the only implemented sub-agent)
# MAGIC 3. Gracefully handle routing to non-implemented sub-agents

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

from mlflow.types.responses import ResponsesRequest
from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.agents.types import AgentType
from telco_support_agent.agents.config import config_manager

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Available Agent Types

# COMMAND ----------

print("Available agent types (enum):")
for agent_type in AgentType:
    print(f"  - {agent_type.name}: {agent_type.value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Agent Configs

# COMMAND ----------

print("Available agent types (configs):", config_manager.get_all_agent_types())

supervisor_config = config_manager.get_config("supervisor")
print("\nSupervisor LLM endpoint:", supervisor_config["llm"]["endpoint"])
print("Supervisor system prompt:", supervisor_config["system_prompt"][:200], "...\n")

account_config = config_manager.get_config("account")
print("\nAccount agent functions:", account_config["uc_functions"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init Supervisor Agent

# COMMAND ----------

supervisor = SupervisorAgent()

print(f"Agent type: {supervisor.agent_type}")
print(f"LLM endpoint: {supervisor.llm_endpoint}")
print(f"LLM parameters: {supervisor.llm_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Routing Logic

# COMMAND ----------

def test_routing(query):
    agent_type = supervisor.route_query(query)
    print(f"Query: '{query}'")
    print(f"Routed to: {agent_type.name} agent ({agent_type.value})\n")
    return agent_type

# test routing with different query types
test_queries = [
    "What plan am I currently on?",  # account
    "Why is my bill higher this month?",  # billing
    "My phone won't connect to the network",  # tech_support
    "What's the difference between the Standard and Premium plans?"  # product
]

routing_results = {}
for query in test_queries:
    routing_results[query] = test_routing(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test End-to-End Query Processing

# COMMAND ----------

def format_response(response):
    """Format and print a response for better readability."""
    print("\n=== RESPONSE ===")
    
    for output_item in response.output:
        if hasattr(output_item, "type") and output_item.type == "message":
            if hasattr(output_item, "content"):
                for content_item in output_item.content:
                    if hasattr(content_item, "type") and content_item.type == "output_text":
                        print("\n" + content_item.text)
        
        elif hasattr(output_item, "type") and output_item.type == "function_call_output":
            print("\nFunction Output:", output_item.output)
            
        elif hasattr(output_item, "type") and output_item.type == "function_call":
            print(f"\nFunction Call: {output_item.name}")
            print(f"Arguments: {output_item.arguments}")
    
    if response.custom_outputs:
        print("\n=== CUSTOM OUTPUTS ===")
        for key, value in response.custom_outputs.items():
            print(f"{key}: {value}")
    
    print("\n" + "="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Account Query

# COMMAND ----------

account_query = "What plan am I currently on? My customer ID is CUS-10001."

request = ResponsesRequest(
    input=[{"role": "user", "content": account_query}]
)

response = supervisor.predict(request)
format_response(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Billing Query

# COMMAND ----------

billing_query = "Why is my bill higher this month? My customer ID is CUS-10001."

request = ResponsesRequest(
    input=[{"role": "user", "content": billing_query}]
)

response = supervisor.predict(request)
format_response(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Streaming Response

# COMMAND ----------

def display_streaming_response(model_input):
    """Display a streaming response as it comes in."""
    print("=== STREAMING RESPONSE ===\n")
    
    full_text = ""
    
    for i, event in enumerate(supervisor.predict_stream(model_input)):
        if event.type == "response.output_item.done":
            item = event.item
            
            if hasattr(item, "type") and item.type == "message":
                if hasattr(item, "content"):
                    for content_item in item.content:
                        if hasattr(content_item, "type") and content_item.type == "output_text":
                            text = content_item.text
                            print(f"Chunk {i}: {text}")
                            full_text += text
            
            elif hasattr(item, "type") and item.type == "function_call":
                print(f"Function Call: {item.name}")
                print(f"Arguments: {item.arguments}")
            
            elif hasattr(item, "type") and item.type == "function_call_output":
                print(f"Function Output: {item.output}")
    
    print("\n=== FULL RESPONSE ===\n")
    print(full_text)
    print("\n" + "="*50)

# COMMAND ----------

streaming_query = "What are the details of my account? I'm customer CUS-10001."

streaming_request = ResponsesRequest(
    input=[{"role": "user", "content": streaming_query}]
)

display_streaming_response(streaming_request)