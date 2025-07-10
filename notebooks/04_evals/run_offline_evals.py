# Databricks notebook source
# MAGIC %md
# MAGIC # Run Offline Evals
# MAGIC
# MAGIC offline evals using custom scorers

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("experiment_name", "/Shared/telco_support_agent/dev/dev_telco_support_agent")
dbutils.widgets.text("model_version", "")

# COMMAND ----------

import os
import sys
import mlflow
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)

experiment_name = dbutils.widgets.get("experiment_name")
model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

from telco_support_agent.evaluation import OFFLINE_SCORERS
from telco_support_agent.ops.registry import get_latest_model_version
from telco_support_agent.utils.config import config_manager

# COMMAND ----------

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Model Veriosn

# COMMAND ----------

uc_config = config_manager.get_uc_config()
uc_model_name = uc_config.get_uc_model_name()

if model_version:
    print(f"Using specified model version: {model_version}")
else:
    model_version = get_latest_model_version(uc_model_name)
    print(f"Using latest model version: {model_version}")

model_uri = f"models:/{uc_model_name}/{model_version}"
print(f"Model: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Evaluation Data

# COMMAND ----------

# Evaluation data with inputs matching the predict_fn parameters
eval_data = [
    {
        "inputs": {
            "messages": [{"role": "user", "content": "What plan am I currently on?"}],
            "customer_id": "CUS-10001"
        }
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "My internet is very slow today"}],
            "customer_id": "CUS-10002"
        }
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "Can you show me my last bill?"}],
            "customer_id": "CUS-10003"
        }
    }
]

print(f"Evaluation dataset: {len(eval_data)} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation

# COMMAND ----------

print(f"Available scorers: {len(OFFLINE_SCORERS)}")
for scorer in OFFLINE_SCORERS:
    print(f"  - {getattr(scorer, '__name__', 'unknown')}")

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)

def predict_fn(messages: list, customer_id: str) -> dict:
    """Wrapper to translate eval parameters to model input format."""
    model_input = {
        "input": messages,
        "custom_inputs": {"customer": customer_id}
    }
    return model.predict(model_input)

# COMMAND ----------

eval_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=OFFLINE_SCORERS
)

print("Evaluation complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# Display evaluation metrics
print("Evaluation Metrics:")
for metric_name, metric_value in eval_results.metrics.items():
    print(f"  {metric_name}: {metric_value:.3f}")

# COMMAND ----------

# Display detailed results table
results_df = eval_results.tables['eval_results_table']
print(f"Detailed results: {results_df.shape[0]} rows x {results_df.shape[1]} columns")
display(results_df)