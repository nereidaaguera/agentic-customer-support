# Databricks notebook source
# MAGIC %md
# MAGIC # Test Supervisor Agent Class
# MAGIC

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

from mlflow.types.responses import ResponsesAgentRequest

from telco_support_agent.agents.config import config_manager
from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.tools import initialize_tools
from telco_support_agent.agents.types import AgentType

# COMMAND ----------

print("Available agent types (enum):")
for agent_type in AgentType:
    print(f"  - {agent_type.name}: {agent_type.value}")

# COMMAND ----------

print("Available agent types (configs):", config_manager.get_all_agent_types())

supervisor_config = config_manager.get_config("supervisor")
print("\nSupervisor LLM endpoint:", supervisor_config["llm"]["endpoint"])

account_config = config_manager.get_config("account")
print("\nAccount agent functions:", account_config["uc_functions"])

tech_support_config = config_manager.get_config("tech_support")
print("\nTech support agent LLM endpoint:", tech_support_config["llm"]["endpoint"])

product_config = config_manager.get_config("product")
print("\nProduct agent LLM endpoint:", product_config["llm"]["endpoint"])
print("\nProduct agent functions:", product_config["uc_functions"])

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
# MAGIC ## Initialize Required UC Functions

# COMMAND ----------

print("Initializing UC functions...")

results = initialize_tools()

success_count = 0
total_count = 0
for domain, functions in results.items():
    for func_name, status in functions.items():
        total_count += 1
        if status:
            success_count += 1
        status_str = "✅" if status else "❌"
        print(f"  {status_str} {func_name}")

print(f"\nInitialized {success_count}/{total_count} UC functions")

if any(not all(functions.values()) for functions in results.values()):
    print("\nWARNING: Some functions could not be initialized")
    print("Tests might fail without the necessary UC functions")
else:
    print("\nAll required functions are available")

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
print("=== ROUTING TESTS ===\n")

routing_test_queries = [
    # Account queries
    "What plan am I currently on?",
    "When did I create my account?",

    # Billing queries
    "Why is my bill higher this month?",
    "When is my payment due?",

    # Tech support queries
    "My phone won't connect to the network",
    "I can't make calls but data works",

    # Product queries
    "What's the difference between the Standard and Premium plans?",
    "Do you have any promotions for existing customers?",
]

routing_results = {}
for query in routing_test_queries:
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

def test_end_to_end_query(query, description=""):
    """Test an end-to-end query with the supervisor agent."""
    print(f"\n{'='*80}")
    print(f"END-TO-END TEST: {description}")
    print(f"Query: '{query}'")
    print('='*80)

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}]
    )

    response = supervisor.predict(request)
    format_response(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Account Queries

# COMMAND ----------

account_queries = [
    ("What plan am I currently on? My customer ID is CUS-10001.", "Account Plan Query"),
]

for query, description in account_queries:
    test_end_to_end_query(query, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Tech Support Queries

# COMMAND ----------

tech_support_queries = [
    ("I can make calls but my data isn't working. How do I fix this?", "Data Connection Issue"),
]

for query, description in tech_support_queries:
    test_end_to_end_query(query, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO: test billing queries

# COMMAND ----------

billing_queries = [
    ("Why is my bill higher this month? My customer ID is CUS-10001.", "Billing Inquiry"),
]

for query, description in billing_queries:
    test_end_to_end_query(query, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Product Queries

# COMMAND ----------

product_queries = [
    ("What's the difference between the Standard and Premium plans?", "Plan Comparison"),
]

for query, description in product_queries:
    test_end_to_end_query(query, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Streaming Responses

# COMMAND ----------

def display_streaming_response(model_input, description=""):
    """Display a streaming response as it comes in."""
    print(f"\n{'='*80}")
    print(f"STREAMING TEST: {description}")
    print('='*80)

    full_text = ""
    function_calls = []
    function_outputs = []

    for i, event in enumerate(supervisor.predict_stream(model_input)):
        if event.type == "response.output_item.done":
            item = event.item

            if "type" in item and item["type"] == "message":
                if "content" in item:
                    for content_item in item["content"]:
                        if "type" in content_item and content_item["type"] == "output_text":
                            text = content_item["text"]
                            print(f"\n[Message Chunk {i}]")
                            print(text)
                            full_text += text

            elif "type" in item and item["type"] == "function_call":
                print(f"\n[Function Call {i}]: {item["name"]}")
                print(f"Arguments: {item["arguments"]}")
                function_calls.append(f"{item["name"]}({item["arguments"]})")

            elif "type" in item and item["type"] == "function_call_output":
                print(f"\n[Function Output {i}]:")
                print(item["output"][:200] + "..." if len(item["output"]) > 200 else item["output"])
                function_outputs.append(item["output"])

    print(f"\n{'='*50}")
    print("STREAMING SUMMARY:")
    print(f"Function Calls: {len(function_calls)}")
    print(f"Function Outputs: {len(function_outputs)}")
    print(f"Final Response Length: {len(full_text)} characters")
    print(f"{'='*80}\n")

# COMMAND ----------

streaming_test_queries = [
    (ResponsesAgentRequest(input=[{"role": "user", "content": "What are the details of my account? I'm customer CUS-10001."}]), "Account Query Streaming"),
    (ResponsesAgentRequest(input=[{"role": "user", "content": "What's the difference between the Standard and Premium plans?"}]), "Plan Comparison"),
    (ResponsesAgentRequest(input=[{"role": "user", "content": "Why is my bill different this month?"}]), "Billing Query Streaming (Not Implemented)"),
]

for request, description in streaming_test_queries:
    display_streaming_response(request, description)
