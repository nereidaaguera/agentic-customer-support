## CLI setup

```bash
export DB_CLI_PROFILE="ml-models-prod" 
databricks auth login --profile $DB_CLI_PROFILE --host https://db-ml-models-prod-us-west.cloud.databricks.com
```

## Env setup
First, set up a Python 3.12 environment as described in the root directory README.md, then set environment 
variables and install dependencies:

```bash
export DATABRICKS_CONFIG_PROFILE="$DB_CLI_PROFILE"
export MLFLOW_TRACKING_URI="databricks://$DB_CLI_PROFILE"
export MLFLOW_EXPERIMENT_ID=2985175900678329

pip install -r telco_support_agent/agent/requirements.txt
pip install -r telco_support_agent/mcp_servers/outage_info_server/requirements.txt
```

## Test the agent

```bash
python telco_support_agent/agent/agent.py
```

## Log and deploy the agent

```bash
python telco_support_agent/agent/deploy.py
```
