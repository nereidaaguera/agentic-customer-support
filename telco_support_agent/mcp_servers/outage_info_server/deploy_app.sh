#!/bin/bash

# Deployment script for Telco Operations MCP Server
# This script automates the Databricks Apps deployment process

set -e  # Exit on any error

# Configuration
APP_NAME="telco-outage-server"
WORKSPACE_PATH="/Workspace/Users/sid.murching@databricks.com/telco_support_agent/outages_mcp_server"
databricks sync . "$WORKSPACE_PATH"
databricks apps deploy "$APP_NAME" --source-code-path "$WORKSPACE_PATH"
