# Databricks notebook source
# MAGIC %md
# MAGIC # Log / Register Agent
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
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
# MAGIC ## Load config

# COMMAND ----------

CONFIG_PATH = "../configs/log_register_agent.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

print("Loaded config:")
print(yaml.dump(config, sort_keys=False, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init Supervisor Agent

# COMMAND ----------

supervisor = SupervisorAgent()
print(f"Initialized supervisor agent using LLM endpoint: {supervisor.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Agent to MLflow

# COMMAND ----------

resources = []
if "additional_resources" in config:
    for resource_config in config["additional_resources"]:
        resource_type = resource_config.pop("type")
        if resource_type == "DatabricksServingEndpoint":
            resources.append(DatabricksServingEndpoint(**resource_config))
        elif resource_type == "DatabricksFunction":
            resources.append(DatabricksFunction(**resource_config))

logged_model_info = log_agent(
    agent=supervisor,
    artifact_path=config["artifact_path"],
    input_example=config["input_example"],
    resources=resources if resources else None,
)

print(f"Successfully logged agent to MLflow: {logged_model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Agent to Unity Catalog

# COMMAND ----------

uc_config = config["uc_registration"]
uc_model_name = f"{uc_config['catalog']}.{uc_config['schema']}.{uc_config['model_name']}"

model_version = register_agent_to_uc(
    model_uri=logged_model_info.model_uri,
    uc_model_name=uc_model_name,
)

print(f"Successfully registered agent to Unity Catalog: {uc_model_name} version {model_version.version}")
