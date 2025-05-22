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
import mlflow

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

CONFIG_PATH = "../../configs/deploy_agent.yaml"

with open(CONFIG_PATH) as f:
    deploy_agent_config = yaml.safe_load(f)

print("Loaded deployment configuration:")
print(yaml.dump(deploy_agent_config, sort_keys=False, default_flow_style=False))

# COMMAND ----------

uc_config = deploy_agent_config["uc_model"]
uc_model_name = f"{uc_config['catalog']}.{uc_config['schema']}.{uc_config['model_name']}"

if "version" in uc_config:
    model_version = uc_config["version"]
    print(f"Using specified model version: {model_version}")
else:
    model_version = get_latest_model_version(uc_model_name)
    if model_version is None:
        raise ValueError(f"No versions found for model: {uc_model_name}")
    print(f"Using latest model version: {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation
# MAGIC 
# MAGIC Load and test the registered model before deployment.

# COMMAND ----------

model_uri = f"models:/{uc_model_name}/{model_version}"
print(f"Loading model: {model_uri}")

try:
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully")
    
    test_input = {
        "input": [{"role": "user", "content": "What plan am I currently on? My customer ID is CUS-10001."}]
    }
    
    print("Testing model prediction...")
    response = loaded_model.predict(test_input)
    
    if response and "output" in response and len(response["output"]) > 0:
        print("Model prediction successful")
        print("Proceeding with deployment...")
    else:
        raise ValueError("Model returned empty or invalid response")
        
except Exception as e:
    print(f"Pre-deployment validation failed: {str(e)}")
    raise RuntimeError("Model validation failed. Deployment aborted.") from e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent

# COMMAND ----------

deployment_config = deploy_agent_config.get("deployment", {})
environment_vars = deploy_agent_config.get("environment_vars", {})

print("Starting deployment...")

deployment_result = deploy_agent(
    uc_model_name=uc_model_name,
    model_version=model_version,
    deployment_name=deployment_config.get("endpoint_name"),
    scale_to_zero_enabled=deployment_config.get("scale_to_zero_enabled", False),
    environment_vars=environment_vars,
)

print("Deployment completed successfully!")

# COMMAND ----------

print("== Deployment Summary ==")
print(f"Endpoint Name: {deployment_result.endpoint_name}")
print(f"Model: {uc_model_name} (version {model_version})")
print(f"Workload Size: {deployment_config.get('workload_size', 'Default')}")
print(f"Scale-to-zero: {'Enabled' if deployment_config.get('scale_to_zero_enabled', False) else 'Disabled'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Deployed Endpoint
# MAGIC 
# MAGIC Verify the deployed endpoint works correctly.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

print("Testing deployed endpoint...")

client = get_deploy_client()
test_query = "What plan am I currently on? My customer ID is CUS-10001."

try:
    response = client.predict(
        endpoint=deployment_result.endpoint_name,
        inputs={
            "input": [{"role": "user", "content": test_query}],
            "databricks_options": {"return_trace": True}
        }
    )

    print("Endpoint test successful!")
    print("\nTest Query Response:")
    for output in response["output"]:
        if "content" in output:
            for content in output["content"]:
                print(content.get("text", ""))
                
except Exception as e:
    print(f"Endpoint test failed: {str(e)}")
