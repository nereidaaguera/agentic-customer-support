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

eval_data = [
    {
        "inputs": {
            "input": [{"role": "user", "content": "What plan am I currently on?"}],
            "custom_inputs": {"customer": "CUS-10001"}
        }
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "My internet is very slow today"}],
            "custom_inputs": {"customer": "CUS-10002"}
        }
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "Can you show me my last bill?"}],
            "custom_inputs": {"customer": "CUS-10003"}
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

# Load the model for evaluation
model = mlflow.pyfunc.load_model(model_uri)

# Define predict function for evaluation
def predict_fn(inputs):
    """Predict function for agent evaluation."""
    return model.predict(inputs)

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

if hasattr(eval_results, 'metrics'):
    print("Metrics:")
    for name, value in eval_results.metrics.items():
        print(f"  {name}: {value}")

if hasattr(eval_results, 'tables'):
    results_df = eval_results.tables['eval_results_table']
    print(f"\nResults shape: {results_df.shape}")
    display(results_df)