# Databricks notebook source
# MAGIC %md
# MAGIC # Base Agent Test
# COMMAND ----------

# MAGIC %pip install -U backoff databricks-openai openai pydantic databricks-agents mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Import `BaseAgent`

# COMMAND ----------

from telco_support_agent.agents.base_agent import BaseAgent, ToolInfo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import other necessary libraries

# COMMAND ----------

import json
from typing import Any, Callable
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit
from mlflow.types.responses import (
    ResponsesRequest,
)
from unitycatalog.ai.core.base import get_uc_function_client

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a concrete TestAgent class

# COMMAND ----------

def create_tool_info(tool_spec, exec_fn=None):
    """Create ToolInfo object from tool spec."""
    # Remove strict parameter if using Claude model
    if "strict" in tool_spec.get("function", {}):
        del tool_spec["function"]["strict"]
    
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")
    
    # Use provided exec_fn or create default UC function executor
    if exec_fn:
        return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn)
    
    # Define wrapper for UC tool execution
    uc_function_client = get_uc_function_client()
    def default_exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=default_exec_fn)

# COMMAND ----------

class TestAgent(BaseAgent):
    """Simple test agent that extends BaseAgent."""
    
    def __init__(self, llm_endpoint: str):
        """Initialize the test agent with basic tools.
        
        Args:
            llm_endpoint: Name of the LLM endpoint to use
        """
        system_prompt = """
        You are a helpful assistant. You can use Python code to help answer questions.
        """
        
        # Set up Python executor tool
        uc_tool_names = ["system.ai.python_exec"]
        uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
        
        # Create tools
        tools = []
        for tool_spec in uc_toolkit.tools:
            tools.append(create_tool_info(tool_spec))
        
        super().__init__(
            llm_endpoint=llm_endpoint,
            tools=tools,
            system_prompt=system_prompt
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and test the agent

# COMMAND ----------

# Define LLM endpoint
LLM_ENDPOINT = "databricks-claude-3-7-sonnet"

# Create the test agent
test_agent = TestAgent(llm_endpoint=LLM_ENDPOINT)

# COMMAND ----------

# test with basic Python execution task
test_input = ResponsesRequest(
    input=[
        {"role": "user", "content": "Calculate the sum of numbers from 1 to 10 using Python."}
    ]
)

# Get prediction
try:
    response = test_agent.predict(test_input)
    print("Success! The agent is working correctly.")
    print("\nAgent Response:")
    print(json.dumps(response.output, indent=2))
except Exception as e:
    print(f"Error testing the agent: {e}")
    import traceback
    print("\nStacktrace:")
    traceback.print_exc()
