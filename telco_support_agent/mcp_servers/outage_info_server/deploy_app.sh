#!/bin/bash

# Deployment script for Telco Operations MCP Server
# This script automates the Databricks Apps deployment process

set -e  # Exit on any error

# Configuration
APP_NAME="telco-operations-mcp-server"
WORKSPACE_PATH="/Workspace/Users/$(databricks current-user me --output json | jq -r '.userName')/telco_mcp_server"

echo "🚀 Deploying Telco Operations MCP Server"
echo "========================================"

# Step 1: Verify Databricks CLI is configured
echo "📋 Step 1: Verifying Databricks CLI configuration..."
if ! databricks current-user me > /dev/null 2>&1; then
    echo "❌ Databricks CLI is not configured. Please run:"
    echo "   databricks configure --token"
    exit 1
fi

CURRENT_USER=$(databricks current-user me --output json | jq -r '.userName')
echo "✅ Logged in as: $CURRENT_USER"

# Step 2: Create workspace directory if it doesn't exist
echo "📁 Step 2: Setting up workspace directory..."
databricks workspace mkdir -p "$WORKSPACE_PATH" || true
echo "✅ Workspace directory ready: $WORKSPACE_PATH"

# Step 3: Sync source code to workspace
echo "📦 Step 3: Syncing source code to workspace..."
echo "   This will upload all files except those in .gitignore"
databricks sync . "$WORKSPACE_PATH"
echo "✅ Source code synced successfully"

# Step 4: Create the app (or update if exists)
echo "🔧 Step 4: Creating/updating Databricks App..."
if databricks apps get "$APP_NAME" > /dev/null 2>&1; then
    echo "   App '$APP_NAME' already exists, updating..."
else
    echo "   Creating new app '$APP_NAME'..."
    databricks apps create "$APP_NAME"
fi

# Step 5: Deploy the application
echo "🚀 Step 5: Deploying the application..."
databricks apps deploy "$APP_NAME" --source-code-path "$WORKSPACE_PATH"

# Step 6: Get app status and URL
echo "📊 Step 6: Getting deployment status..."
sleep 5  # Give the app a moment to start
APP_INFO=$(databricks apps get "$APP_NAME" --output json)
APP_STATUS=$(echo "$APP_INFO" | jq -r '.status')
APP_URL=$(echo "$APP_INFO" | jq -r '.url // "Not available yet"')

echo ""
echo "🎉 Deployment completed!"
echo "======================="
echo "App Name: $APP_NAME"
echo "Status: $APP_STATUS"
echo "URL: $APP_URL"
echo "MCP Endpoint: $APP_URL/api/mcp"
echo ""

if [ "$APP_STATUS" = "RUNNING" ]; then
    echo "✅ Your Telco Operations MCP Server is now running!"
    echo ""
    echo "🔧 Available Tools:"
    echo "   • check-outage-status: Check network outage status" 
    echo "   • get-network-metrics: Get network performance metrics"
    echo "   • report-network-issue: Report network issues"
    echo ""
    echo "📖 Test your deployment:"
    echo "   Update demo_telco_integration.py with your app URL and run the demo"
else
    echo "⚠️  App is in status: $APP_STATUS"
    echo "   Check the logs in the Databricks UI for any issues"
    echo "   Run: databricks apps get $APP_NAME"
fi

echo ""
echo "🛠️  Useful commands:"
echo "   • View app status: databricks apps get $APP_NAME"
echo "   • View logs: databricks apps logs $APP_NAME"
echo "   • Stop app: databricks apps stop $APP_NAME"
echo "   • Start app: databricks apps start $APP_NAME"
echo "   • Delete app: databricks apps delete $APP_NAME" 