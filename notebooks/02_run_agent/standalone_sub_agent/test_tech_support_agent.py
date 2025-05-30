# Databricks notebook source
# MAGIC %md
# MAGIC # Test Tech Support Agent
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

from telco_support_agent.agents.tech_support import TechSupportAgent
from telco_support_agent.utils.logging_utils import setup_logging

setup_logging()

# running async functions in notebook
import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init Tech Support Agent

# COMMAND ----------

tech_agent = TechSupportAgent(environment="prod")

print(f"\nAgent initialized successfully!")
print(f"   Agent type: {tech_agent.agent_type}")
print(f"   LLM endpoint: {tech_agent.llm_endpoint}")
print(f"   LLM parameters: {tech_agent.llm_params}")
print(f"   Number of tools: {len(tech_agent.tools)}")

print(f"\nAvailable tools:")
for i, tool in enumerate(tech_agent.tools, 1):
    if "function" in tool:
        tool_name = tool["function"]["name"]
        tool_desc = tool["function"].get("description", "No description")
        print(f"   {i}. {tool_name}")
        print(f"      {tool_desc}")

print(f"\nRetriever info:")
print(f"   Environment: {tech_agent.retriever.environment}")
print(f"   KB Index: {tech_agent.retriever.kb_retriever.index_name}")
print(f"   Tickets Index: {tech_agent.retriever.tickets_retriever.index_name}")

# COMMAND ----------


def test_query(query, test_id=""):
    print(f"\n{'='*80}")
    print(f"TEST QUERY{test_id}: \"{query}\"")
    print('='*80)
        
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}]
    )

    response = tech_agent.predict(request)
    
    if response and hasattr(response, 'output') and response.output:
        for output_item in reversed(response.output):
            if hasattr(output_item, 'type') and output_item.type == "message" and hasattr(output_item, 'content'):
                if isinstance(output_item.content, list):
                    for content_item in output_item.content:
                        if hasattr(content_item, 'type') and content_item.type == "output_text":
                            print(f"\nRESPONSE:")
                            print(content_item.text)
                            break
                else:
                    print(f"\nRESPONSE:")
                    print(output_item.content)
                break
        else:
            print(f"\nRESPONSE:")
            print("Could not find text response in output")
            print(f"Raw output: {response.output}")
    else:
        print(f"\nNo response or empty response received")
    
    print(f"\n{'='*80}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test individual retriever tools

# COMMAND ----------

print("Knowledge Base Retriever:")
kb_results = tech_agent.retriever.search_knowledge_base("how to manage multiple lines")
kb_count = len(kb_results) if isinstance(kb_results, list) else 0
print(f"   Found {kb_count} knowledge base articles")

print("\nSupport Tickets Retriever:")
tickets_results = tech_agent.retriever.search_tickets("iPhone connectivity issues")
tickets_count = len(tickets_results) if isinstance(tickets_results, list) else 0
print(f"   Found {tickets_count} historical support tickets")

print("\nCombined Search (Async):")
import asyncio
combined_results = asyncio.run(tech_agent.retriever.async_search("network connection problems"))

kb_combined = 0
tickets_combined = 0

if "knowledge_base" in combined_results:
    kb_data = combined_results["knowledge_base"]
    if isinstance(kb_data, list):
        kb_combined = len(kb_data)
    elif isinstance(kb_data, dict) and "error" not in kb_data:
        kb_combined = len(kb_data) if isinstance(kb_data, list) else 0

if "support_tickets" in combined_results:
    tickets_data = combined_results["support_tickets"]
    if isinstance(tickets_data, list):
        tickets_combined = len(tickets_data)
    elif isinstance(tickets_data, dict) and "error" not in tickets_data:
        tickets_combined = len(tickets_data) if isinstance(tickets_data, list) else 0

print(f"   Found {kb_combined} KB + {tickets_combined} tickets")

if kb_results and isinstance(kb_results, list) and len(kb_results) > 0:
    print(f"\nSample KB Result:")
    sample_kb = kb_results[0]
    if isinstance(sample_kb, dict):
        print(f"   Title: {sample_kb.get('metadata', {}).get('title', 'N/A')}")
        print(f"   Category: {sample_kb.get('metadata', {}).get('category', 'N/A')}")
        print(f"   Content: {sample_kb.get('page_content', '')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test queries

# COMMAND ----------

roaming_device_queries = [
    "My iPhone 15 Pro shows 'Emergency Calls Only' in Mexico even though I have international roaming. How do I fix this?",
    "I'm traveling in Mexico and my phone won't connect to local networks. My plan includes international roaming.",
    "My iPhone works fine in the US but won't connect to data networks when traveling internationally. What should I try?",
    "Why does my Galaxy S24 work fine internationally but my iPhone 15 Pro Max doesn't on the same plan?",
    "How long does it take for international roaming features to activate after I add them to my plan?"
]

for i, query in enumerate(roaming_device_queries, 1):
    test_query(query, f"_roaming_{i}")

# COMMAND ----------

general_tech_queries = [
    "My phone won't connect to the network at all. What troubleshooting steps should I try?",
    "I can make calls but my data isn't working. How do I fix this?",
    "How do I set up my new iPhone with an eSIM?",
    "Why is my internet speed so slow on my phone?",
    "I'm not receiving text messages but can send them. What's wrong?",
    "My voicemail isn't working properly. How do I reset it?",
    "How do I manually select a network when traveling?",
    "What should I do if my phone keeps dropping calls in my area?"
]

for i, query in enumerate(general_tech_queries, 1):
    test_query(query, f"_general_{i}")

# COMMAND ----------

complex_queries = [
    "I upgraded to the Family Share Plus plan last month and was told international roaming in Mexico would be included, but my iPhone 15 Pro keeps showing Emergency Calls Only when I'm in Cancun. I've tried restarting, airplane mode, and manual network selection. My husband's Galaxy S24 on the same plan works fine. This has been going on for 3 days and I need it fixed immediately.",
    
    "I have multiple lines on my account and want to set different data limits for my teenagers' phones while keeping unlimited data for the adults. Also, one of the teen lines keeps having network connectivity issues with their iPhone 14. How do I manage both the parental controls and fix the connectivity problem?",
    
    "I just added a new line to my business account for an employee who's traveling internationally next week. The line was supposed to be activated within 2 hours but it's been 6 hours and still shows no service. They need international roaming capabilities and I need to know how to expedite this activation and ensure roaming will work."
]

for i, query in enumerate(complex_queries, 1):
    test_query(query, f"_complex_{i}")