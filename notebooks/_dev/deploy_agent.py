# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
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

from telco_support_agent.ops.deployment import deploy_agent
from telco_support_agent.ops.registry import get_latest_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Deployment Config

# COMMAND ----------

CONFIG_PATH = "../configs/deploy_agent.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

print("Loaded deployment configuration:")
print(yaml.dump(config, sort_keys=False, default_flow_style=False))

# COMMAND ----------

uc_config = config["uc_model"]
uc_model_name = f"{uc_config['catalog']}.{uc_config['schema']}.{uc_config['model_name']}"

if "version" in uc_config:
    model_version = uc_config["version"]
    print(f"Using specified model version: {model_version}")
else:
    model_version_obj = get_latest_model_version(uc_model_name)
    if model_version_obj is None:
        raise ValueError(f"No versions found for model: {uc_model_name}")
    model_version = model_version_obj.version
    print(f"Using latest model version: {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent

# COMMAND ----------

deployment_config = config.get("deployment", {})
environment_vars = config.get("environment_vars", {})

deployment_result = deploy_agent(
    uc_model_name=uc_model_name,
    model_version=model_version,
    deployment_name=deployment_config.get("endpoint_name"),
    scale_to_zero_enabled=deployment_config.get("scale_to_zero_enabled", False),
    environment_vars=environment_vars,
)

# COMMAND ----------

print("== Deployment Summary ==")
print(f"Endpoint Name: {deployment_result.endpoint_name}")
print(f"Status: {deployment_result.status}")
print(f"Query Endpoint: {deployment_result.query_endpoint}")
print(f"Workload Size: {deployment_config.get('workload_size', 'Default')}")
print(f"Scale-to-zero: {'Enabled' if deployment_config.get('scale_to_zero_enabled', False) else 'Disabled'}")
print(f"Model: {uc_model_name} (version {model_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Query

# COMMAND ----------

# from mlflow.deployments import get_deploy_client
#
# client = get_deploy_client()
# test_query = "What plan am I currently on? My customer ID is CUS-10001."
#
# response = client.predict(
#     endpoint=deployment_result.endpoint_name,
#     inputs={
#         "input": [{"role": "user", "content": test_query}],
#         "databricks_options": {"return_trace": True}
#     }
# )
#
# print("Test Query Response:")
# for output in response["output"]:
#     if "content" in output:
#         for content in output["content"]:
#             print(content.get("text", ""))