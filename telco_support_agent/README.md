## CLI setup

databricks configure --profile ml-models-prod
databricks auth login --profile ml-models-prod

## Env setup

```bash
export DB_CLI_PROFILE="ml-models-prod" 
export DATABRICKS_CONFIG_PROFILE="$CLI_PROFILE"
export MLFLOW_TRACKING_URI="databricks://ml-models-prod"
```

## Run the server

```bash
uv run mcp-on-apps -p ml-models-prod
```