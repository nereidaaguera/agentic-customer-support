# Databricks notebook source
# MAGIC %md
# MAGIC # Log & Register Agent
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import inspect
import os
import sys
import mlflow
import yaml

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)
print(f"Added {project_root} to Python path")

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
)

from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.ops.logging import log_agent
from telco_support_agent.ops.registry import register_agent_to_uc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load log_register_agent config

# COMMAND ----------

CONFIG_PATH = "../../configs/log_register_agent.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

print("Loaded config:")
print(yaml.dump(config, sort_keys=False, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Agent to MLflow

# COMMAND ----------

# collect resources (if any specified in config)
resources = []
if "additional_resources" in config:
    for resource_config in config["additional_resources"]:
        resource_type = resource_config.pop("type")
        if resource_type == "DatabricksServingEndpoint":
            resources.append(DatabricksServingEndpoint(**resource_config))
        elif resource_type == "DatabricksFunction":
            resources.append(DatabricksFunction(**resource_config))

logged_model_info = log_agent(
    agent_class=SupervisorAgent,
    name=config["name"],
    input_example=config["input_example"],
    resources=resources if resources else None,
)

print(f"Successfully logged agent to MLflow: {logged_model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Logged Model
# MAGIC Test that the logged model can be loaded and run in an isolated environment

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(logged_model_info.model_uri)
print(f"Successfully loaded the model from: {logged_model_info.model_uri}")

test_input = {
    "input": [{"role": "user", "content": "What plan am I currently on? My customer ID is CUS-10001."}]
}

print("\nSending test query to the loaded model...")
response = loaded_model.predict(test_input)

print("\nModel Response:")
for output in response.get("output", []):
    if output.get("type") == "message" and output.get("content"):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                print(content.get("text", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Agent to Unity Catalog

# COMMAND ----------

# uc_config = config["uc_registration"]
# uc_model_name = f"{uc_config['catalog']}.{uc_config['schema']}.{uc_config['model_name']}"

# model_version = register_agent_to_uc(
#     model_uri=logged_model_info.model_uri,
#     uc_model_name=uc_model_name,
# )

# print(f"Successfully registered agent to Unity Catalog: {uc_model_name} version {model_version.version}")
