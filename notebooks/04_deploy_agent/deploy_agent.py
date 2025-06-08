# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("git_commit", "")
dbutils.widgets.text("experiment_name", "/telco_support_agent/dev/experiments/dev_telco_support_agent")

# COMMAND ----------

import os
import sys
import yaml
import mlflow

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)

env = dbutils.widgets.get("env")
git_commit = dbutils.widgets.get("git_commit")
experiment_name = dbutils.widgets.get("experiment_name")

os.environ['TELCO_SUPPORT_AGENT_ENV'] = env

# COMMAND ----------

from telco_support_agent.ops.deployment import deploy_agent, cleanup_old_deployments, AgentDeploymentError
from telco_support_agent.utils.config import config_manager
from telco_support_agent.ops.registry import get_latest_model_version

# COMMAND ----------

mlflow.set_experiment(experiment_name)

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
environment_vars = deploy_agent_config.get("environment_vars", {}) | {"TELCO_SUPPORT_AGENT_ENV": env}
permissions = deploy_agent_config.get("permissions")
instructions = deploy_agent_config.get("instructions")

uc_model_name = uc_config.get_uc_model_name()

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
        "input": [{"role": "user", "content": "what was the customer's data usage last month?"}],
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

endpoint_name = f"{env}-{deployment_config.get('endpoint_name')}"

print("Deploying agent..")
print(f"Model: {uc_model_name} version {model_version}")
print(f"Endpoint: {endpoint_name}")
print(f"Workload Size: {deployment_config.get('workload_size', 'Small')}")
print(f"Scale-to-zero: {deployment_config.get('scale_to_zero_enabled', False)}")
print(f"Wait for endpoint to be ready: {deployment_config.get('wait_for_ready', True)}")
print(f"  Environment: {env}")
print(f"  Git Commit: {git_commit}")
print(f"  Experiment Name: {experiment_name}")

if environment_vars:
    print(f"Environment variables: {list(environment_vars.keys())}")

if permissions:
    if isinstance(permissions, list):
        print("Setting permissions for:")
        for perm_config in permissions:
            users = perm_config.get('users', [])
            permission_level = perm_config.get('permission_level', 'Unknown')
            print(f"  - {permission_level}: {', '.join(users)}")
    else:
        print(f"Setting permissions for: {permissions.get('users', [])}")

if instructions:
    print("Setting review instructions")

try:
    deployment_result = deploy_agent(
        uc_model_name=uc_model_name,
        model_version=model_version,
        deployment_name=endpoint_name,
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
# MAGIC ## Clean up old model serving endpoints

# COMMAND ----------

cleanup_enabled = deploy_agent_config.get("cleanup_old_versions", False)

if cleanup_enabled:
    cleanup_config = deploy_agent_config.get("cleanup", {})

    print("="*50)
    print("CLEANING UP OLD DEPLOYMENT VERSIONS")
    print("="*50)
    print(f"Model: {uc_model_name}")
    print(f"Current version: {model_version}")
    print(f"Endpoint: {deployment_result.endpoint_name}")
    print(f"Keep previous versions: {cleanup_config.get('keep_previous_count', 1)}")
    print()

    try:
        cleanup_result = cleanup_old_deployments(
            model_name=uc_model_name,
            current_version=str(model_version),
            endpoint_name=deployment_result.endpoint_name,
            keep_previous_count=cleanup_config.get("keep_previous_count", 1),
            max_deletion_attempts=cleanup_config.get("max_deletion_attempts", 3),
            wait_between_attempts=cleanup_config.get("wait_between_attempts", 60),
            wait_after_deletion=cleanup_config.get("wait_after_deletion", 180),
            raise_on_error=cleanup_config.get("raise_on_error", False),
        )

        print("✅ Cleanup completed!")
        print(f"Versions kept: {cleanup_result['versions_kept']}")
        print(f"Versions deleted: {cleanup_result['versions_deleted']}")

        if cleanup_result['versions_failed']:
            print(f"⚠️ Versions that failed to delete: {cleanup_result['versions_failed']}")
            print("These may need manual cleanup or will be retried in future deployments.")

        if not cleanup_result['versions_deleted'] and not cleanup_result['versions_failed']:
            print("No old versions found to clean up.")

    except AgentDeploymentError as e:
        print(f"❌ Cleanup failed with error: {str(e)}")
        if cleanup_config.get("raise_on_error", False):
            raise
        else:
            print("Continuing despite cleanup failure (raise_on_error=false)")
    except Exception as e:
        print(f"❌ Unexpected cleanup error: {str(e)}")
        if cleanup_config.get("raise_on_error", False):
            raise
        else:
            print("Continuing despite cleanup failure (raise_on_error=false)")

    print("="*50)
else:
    print("Cleanup of old versions is disabled in configuration")
    print("To enable, set 'cleanup_old_versions: true' in deploy_agent.yaml")

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
