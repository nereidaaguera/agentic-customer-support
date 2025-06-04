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

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("git_commit", "")
dbutils.widgets.text("experiment_name", "/Workspace/telco_support_agent/dev/experiments")

env = dbutils.widgets.get("env")
git_commit = dbutils.widgets.get("git_commit")
experiment_name = dbutils.widgets.get("experiment_name")

# Setting env variable for telco support agent. In this way, the agent will deploy in the correct catalog and schema.
os.environ['TELCO_SUPPORT_AGENT_ENV'] = env

# COMMAND ----------

from telco_support_agent.ops.deployment import deploy_agent, AgentDeploymentError
from telco_support_agent.utils.config import config_manager
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

uc_config = config_manager.get_uc_config()
deployment_config = deploy_agent_config.get("deployment", {})
environment_vars = deploy_agent_config.get("environment_vars", {})
permissions = deploy_agent_config.get("permissions")
instructions = deploy_agent_config.get("instructions")

uc_model_name = f"{uc_config.agent['catalog']}.{uc_config.agent['schema']}.{uc_config.agent['model_name']}"

if "version" in uc_config.agent:
    model_version = uc_config.agent["version"]
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
    print("✅ Model loaded successfully")

    test_input = {
        "input": [{"role": "user", "content": "What plan am I currently on?"}],
        "custom_inputs": {"customer": "CUS-10001"}
    }

    print("Testing model prediction...")
    response = loaded_model.predict(test_input)

    if response and "output" in response and len(response["output"]) > 0:
        print("✅ Model prediction successful")
        print("Proceeding with deployment...")
    else:
        raise ValueError("Model returned empty or invalid response")

except Exception as e:
    print(f"❌ Pre-deployment validation failed: {str(e)}")
    raise RuntimeError("Model validation failed. Deployment aborted.") from e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent

# COMMAND ----------

print("Deploying agent..")
print(f"Model: {uc_model_name} version {model_version}")
print(f"Endpoint: {deployment_config.get('endpoint_name')}")
print(f"Workload Size: {deployment_config.get('workload_size', 'Small')}")
print(f"Scale-to-zero: {deployment_config.get('scale_to_zero_enabled', False)}")
print(f"Wait for endpoint to be ready: {deployment_config.get('wait_for_ready', True)}")

if environment_vars:
    print(f"Environment variables: {list(environment_vars.keys())}")

if permissions:
    print(f"Setting permissions for: {permissions.get('users', [])}")

if instructions:
    print("Setting review instructions")

try:
    deployment_result = deploy_agent(
        uc_model_name=uc_model_name,
        model_version=model_version,
        deployment_name=deployment_config.get("endpoint_name"),
        tags=deployment_config.get("tags"),
        scale_to_zero_enabled=deployment_config.get("scale_to_zero_enabled", False),
        environment_vars=environment_vars if environment_vars else None,
        workload_size=deployment_config.get("workload_size", "Small"),
        wait_for_ready=deployment_config.get("wait_for_ready", True),
        permissions=permissions,
        instructions=instructions,
        budget_policy_id=deployment_config.get("budget_policy_id"),
    )

    print("✅ Deployment completed successfully!")

except AgentDeploymentError as e:
    print(f"❌ Deployment failed: {str(e)}")
    raise
except Exception as e:
    print(f"❌ Unexpected deployment error: {str(e)}")
    raise

# COMMAND ----------

print("\n" + "="*50)
print("DEPLOYMENT SUMMARY")
print("="*50)
print(f"Endpoint Name: {deployment_result.endpoint_name}")
print(f"Model: {uc_model_name} (version {model_version})")
print(f"Workload Size: {deployment_config.get('workload_size', 'Small')}")
print(f"Scale-to-zero: {'Enabled' if deployment_config.get('scale_to_zero_enabled', False) else 'Disabled'}")
print(f"Query Endpoint: {deployment_result.query_endpoint}")

if hasattr(deployment_result, 'review_app_url'):
    print(f"Review App: {deployment_result.review_app_url}")

print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Deployed Endpoint
# MAGIC
# MAGIC Verify deployed endpoint works correctly

# COMMAND ----------

from mlflow.deployments import get_deploy_client

print("Testing deployed endpoint with custom inputs...")

client = get_deploy_client()

test_cases = [
    {
        "input": [{"role": "user", "content": "What plan am I currently on?"}],
        "custom_inputs": {"customer": "CUS-10001"},
        "description": "Account query with customer ID"
    },
    {
        "input": [{"role": "user", "content": "Show me my billing details for this month"}],
        "custom_inputs": {"customer": "CUS-10002"},
        "description": "Billing query with customer ID"
    },
    {
        "input": [{"role": "user", "content": "What devices do I have on my account?"}],
        "custom_inputs": {"customer": "CUS-10003"},
        "description": "Product query with customer ID"
    },
    {
        "input": [{"role": "user", "content": "My phone won't connect to WiFi"}],
        "description": "Tech support query (no custom inputs required)"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i}: {test_case['description']} ---")

    try:
        request_data = {
            "input": test_case["input"],
            "databricks_options": {"return_trace": True}
        }

        if "custom_inputs" in test_case:
            request_data["custom_inputs"] = test_case["custom_inputs"]
            print(f"Custom inputs: {test_case['custom_inputs']}")

        response = client.predict(
            endpoint=deployment_result.endpoint_name,
            inputs=request_data
        )

        print("✅ Query successful!")

        for output in response["output"]:
            if "content" in output:
                for content in output["content"]:
                    if "text" in content:
                        print(f"Response: {content['text'][:200]}...")
                        break

        if "custom_outputs" in response:
            print(f"Custom outputs: {response['custom_outputs']}")

    except Exception as e:
        print(f"❌ Query failed: {str(e)}")

print("\n🎉 Custom inputs endpoint testing completed!")
