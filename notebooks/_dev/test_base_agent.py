# Databricks notebook source
# MAGIC %md
# MAGIC # Base Agent Test

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
from typing import Any, Dict
from uuid import uuid4

from mlflow.types.responses import ResponsesRequest

from telco_support_agent.agents.base_agent import BaseAgent, ToolInfo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple calculator tool

# COMMAND ----------

def calculator_tool(**kwargs):
    operation = kwargs.get("operation")
    a = kwargs.get("a")
    b = kwargs.get("b")
    
    if not all([operation, a is not None, b is not None]):
        return "Error: Missing required parameters. Need 'operation', 'a', and 'b'."
    
    try:
        a = float(a)
        b = float(b)
    except (ValueError, TypeError):
        return "Error: Parameters 'a' and 'b' must be numbers."
    
    if operation == "add":
        return f"Result: {a + b}"
    elif operation == "subtract":
        return f"Result: {a - b}"
    elif operation == "multiply":
        return f"Result: {a * b}"
    elif operation == "divide":
        if b == 0:
            return "Error: Cannot divide by zero."
        return f"Result: {a / b}"
    else:
        return f"Error: Unknown operation '{operation}'. Supported operations are: add, subtract, multiply, divide."

# calculator tool spec
calculator_tool_spec = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "The first number"
                },
                "b": {
                    "type": "number",
                    "description": "The second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
}

calculator_tool_info = ToolInfo(
    name="calculator",
    spec=calculator_tool_spec,
    exec_fn=calculator_tool
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic test agent

# COMMAND ----------

class BasicTestAgent(BaseAgent):    
    def __init__(self, llm_endpoint: str):
        system_prompt = """
        You are a helpful assistant that can perform calculations.
        Use the calculator tool to help with arithmetic operations.
        """
        
        tools = [calculator_tool_info]
        
        super().__init__(
            llm_endpoint=llm_endpoint,
            tools=tools,
            system_prompt=system_prompt
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create / test agent

# COMMAND ----------

LLM_ENDPOINT = "databricks-claude-3-7-sonnet"

basic_agent = BasicTestAgent(llm_endpoint=LLM_ENDPOINT)

# COMMAND ----------

test_input = ResponsesRequest(
    input=[
        {"role": "user", "content": "What is 42 multiplied by 7?"}
    ]
)

response = basic_agent.predict(test_input)
print("\nAgent Response:")
print(response.output)
