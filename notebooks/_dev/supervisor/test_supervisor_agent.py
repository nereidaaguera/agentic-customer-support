# Databricks notebook source
# MAGIC %pip install -r ../../requirements.txt --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow.types.responses import ResponsesRequest

from telco_support_agent.agents.supervisor import SupervisorAgent

# COMMAND ----------

LLM_ENDPOINT = "databricks-claude-3-7-sonnet"

supervisor = SupervisorAgent(llm_endpoint=LLM_ENDPOINT)

# COMMAND ----------

test_queries = [
    "What plan am I currently on?",
    "Why is my bill higher this month?",
    "My phone won't connect to the network",
    "What's the difference between the Standard and Premium plans?"
]

def test_query(query):
    print(f"\n=== TESTING QUERY: \"{query}\" ===\n")
    
    request = ResponsesRequest(
        input=[{"role": "user", "content": query}]
    )
    
    response = supervisor.predict(request)
    
    for output_item in response.output:
        if output_item.get("type") == "message":
            for content_item in output_item.get("content", []):
                if content_item.get("type") == "output_text":
                    print(content_item.get("text"))
        elif output_item.get("type") == "function_call_output":
            print(output_item.get("output"))
    
    print("\n" + "="*80)

# COMMAND ----------

for query in test_queries:
    test_query(query)