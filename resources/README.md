# Databricks Resources

This directory contains all Databricks Asset Bundle resource definitions, organized by domain for clear separation of concerns.

## Organization

### `data_pipelines/`
Data engineering pipeline resources:
- `create_vector_indexes.yml` - Vector search index creation job for knowledge base and support tickets

### `agent/`
Machine learning agent resources:
- `log_register_deploy_agent.yml` - Complete ML lifecycle pipeline (log, register, deploy agent to model serving)
- `mcp_servers.yml` - MCP (Model Context Protocol) server applications that provide tools for agents

### `ui/`
User interface application resources:
- `customer_support_app.yml` - Customer support UI application

## Deployment

### CI/CD Workflows

- **PR Validation**: Runs on pull requests to validate bundle configuration and run tests
- **Dev Deployment**: Manual trigger only for developer testing
- **Staging Deployment**: Automatic on push to `main` branch for integration testing
- **Prod Deployment**: Manual trigger or release tag creation with validation

### Manual Deployment

Deploy all resources to an environment:
```bash
databricks bundle deploy -t <environment>
```

Run specific jobs:
```bash
databricks bundle run log_register_deploy_agent -t <environment>
databricks bundle run create_vector_indexes -t <environment>
```

Deploy app code to running apps:
```bash
databricks bundle run mcp_telco_outage_info -t <environment>
databricks bundle run customer_support_ui -t <environment>
```