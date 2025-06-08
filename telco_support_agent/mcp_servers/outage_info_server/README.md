## Sync latest app code to Databricks
databricks sync . /Workspace/telco_outages_mcp_server

## One-time setup: create the app 
databricks apps create mcp-telco-outage-server

## Redeploy the app from the latest source code
databricks apps deploy mcp-telco-outage-server --source-code-path /Workspace/telco_outages_mcp_server


