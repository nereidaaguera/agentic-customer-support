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

scorer_names = [
    "brand_compliance_scorer",
    "data_privacy_scorer", 
    "query_resolution_scorer",
    "response_clarity_scorer",
    "routing_accuracy_scorer",
    "tool_accuracy_scorer"
]

print(f"Available scorers: {len(OFFLINE_SCORERS)}")
for i, scorer in enumerate(OFFLINE_SCORERS):
    name = scorer_names[i] if i < len(scorer_names) else f"scorer_{i}"
    print(f"  - {name}")

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

# evaluation metrics
print("Evaluation Metrics:")
for metric_name, metric_value in eval_results.metrics.items():
    print(f"  {metric_name}: {metric_value:.3f}")

# COMMAND ----------

# detailed results - get traces with assessments
eval_traces = mlflow.search_traces(run_id=eval_results.run_id)
print(f"Detailed results: {eval_traces.shape[0]} traces x {eval_traces.shape[1]} columns")

print("\nTrace columns:")
for col in eval_traces.columns:
    print(f"  - {col}")

# COMMAND ----------

eval_traces.iloc[0]['trace']

# COMMAND ----------

if len(eval_traces) > 0:
    print("Sample assessments from first trace:")
    sample_assessments = eval_traces.iloc[0]['assessments']
    for assessment in sample_assessments:
        print(f"  - {assessment.name}: {assessment.feedback.value}")
        # Check for rationale in feedback object
        if hasattr(assessment.feedback, 'rationale') and assessment.feedback.rationale:
            print(f"    Rationale: {assessment.feedback.rationale[:100]}...")
        print()
