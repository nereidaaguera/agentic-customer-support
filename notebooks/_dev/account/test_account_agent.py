# Databricks notebook source
# MAGIC %pip install -r ../../../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.library.restartPython()

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
    input=[{"role": "user", "content": "How many plans do I have on my account?"},
           {"role": "assistant", "content": "To provide you info about your plans, I need your customer ID. Could you please provide me with this information?"},
           {"role": "user", "content": "My customer ID is CUS-10601"},
           {"role": "assistant", "content": "Based on the information I've retrieved, you currently have 2 active plans on your account"},
           {"role": "user", "content": "Can you describe the cheapest plan?"}]
)

# COMMAND ----------

account_agent.predict(request)

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
        if hasattr(output_item, "type"):
            if output_item.type == "message" and hasattr(output_item, "content"):
                for content_item in output_item.content:
                    if hasattr(content_item, "type") and content_item.type == "output_text":
                        print(content_item.text)
            elif output_item.type == "function_call_output" and hasattr(output_item, "output"):
                print(output_item.output)
    
    print("\n" + "="*80)

# COMMAND ----------

for query in test_queries:
    test_query(query)
