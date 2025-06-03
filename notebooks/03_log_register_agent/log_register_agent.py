# Databricks notebook source
# MAGIC %md
# MAGIC # Log & Register Agent
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
import mlflow
import yaml

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)

# COMMAND ----------

from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.utils.config import config_manager
from telco_support_agent.ops.logging import log_agent
from telco_support_agent.ops.registry import register_agent_to_uc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load log_register_agent config

# COMMAND ----------

CONFIG_PATH = "../../configs/log_register_agent.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

uc_config = config_manager.get_uc_config()
uc_model_name = f"{uc_config.agent['catalog']}.{uc_config.agent['schema']}.{uc_config.agent['model_name']}"

print("Configuration:")
print(f"  Name: {config['name']}")
print(f"  Description: {config['description']}")
print(f"  UC Model: {uc_model_name}")
print(f"  Input Example: {config['input_example']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC Functions

# COMMAND ----------

from telco_support_agent.tools import initialize_tools

print("Initializing required UC functions...")

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Supervisor Agent

# COMMAND ----------

print("Testing supervisor agent...")

supervisor = SupervisorAgent()
print(f"Created supervisor agent (LLM: {supervisor.llm_endpoint})")

from mlflow.types.responses import ResponsesAgentRequest

test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What plan am I currently on?"}],
    custom_inputs={"customer": "CUS-10001"}
)

response = supervisor.predict(test_request)
print("Test query completed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Agent to MLflow

# COMMAND ----------

print("Logging supervisor agent to MLflow...")

logged_model_info = log_agent(
    agent_class=SupervisorAgent,
    name=config["name"],
    input_example=config["input_example"],
    environment="prod",
)

print(f"Logged agent: {logged_model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Logged Model

# COMMAND ----------

print("Testing logged model...")

loaded_model = mlflow.pyfunc.load_model(logged_model_info.model_uri)

test_input = config["input_example"]
print(f"Testing with input: {test_input}")

response = loaded_model.predict(test_input)

print("✅ Logged model works correctly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog

# COMMAND ----------

print("Registering to Unity Catalog...")

model_version = register_agent_to_uc(
    model_uri=logged_model_info.model_uri,
    uc_model_name=uc_model_name,
)

print(f"✅ Registered: {uc_model_name} version {model_version.version}")
