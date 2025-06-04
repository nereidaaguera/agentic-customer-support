## Sync latest app code to Databricks
databricks sync . /Workspace/telco_outages_mcp_server

## Redeploy the app from the latest source code
databricks apps deploy telco-outage-server  

