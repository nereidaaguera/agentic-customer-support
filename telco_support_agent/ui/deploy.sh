#!/bin/bash

# Deployment script for dev/staging/prod environments
# Usage: 
#   ./deploy.sh dev [databricks_profile]
#   ./deploy.sh staging [databricks_profile]
#   ./deploy.sh prod [databricks_profile]

set -e  # Exit on any error

# Function to display usage
show_usage() {
    echo "Usage:"
    echo "  ./deploy.sh dev [databricks_profile]        # Deploy to dev environment"
    echo "  ./deploy.sh staging [databricks_profile]    # Deploy to staging environment"
    echo "  ./deploy.sh prod [databricks_profile]       # Deploy to prod environment"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh dev"
    echo "  ./deploy.sh staging"
    echo "  ./deploy.sh prod PRODUCTION"
    echo "  ./deploy.sh dev dev-profile"
}

# Function to validate environment
validate_environment() {
    local env=$1
    if [[ "$env" != "dev" && "$env" != "staging" && "$env" != "prod" ]]; then
        echo "❌ Error: Environment must be 'dev', 'staging', or 'prod'"
        show_usage
        exit 1
    fi
}

# Function to check if config file exists
check_config_file() {
    local config_file=$1
    if [[ ! -f "$config_file" ]]; then
        echo "❌ Error: Configuration file $config_file not found"
        echo "Please ensure you have the environment-specific config file"
        exit 1
    fi
}

# Check arguments
if [[ $# -eq 0 ]]; then
    echo "❌ Error: No environment specified"
    show_usage
    exit 1
fi

# Environment-specific deployment
ENVIRONMENT=$1
DATABRICKS_PROFILE=${2:-"DEFAULT"}

validate_environment "$ENVIRONMENT"

# Set environment-specific variables
if [[ "$ENVIRONMENT" == "dev" ]]; then
    APP_FOLDER_IN_WORKSPACE="/Workspace/Shared/telco_support_agent/dev/databricks_app"
    LAKEHOUSE_APP_NAME="telco-support-agent-dev"
    CONFIG_FILE="app_dev.yaml"
elif [[ "$ENVIRONMENT" == "staging" ]]; then
    APP_FOLDER_IN_WORKSPACE="/Workspace/Shared/telco_support_agent/staging/databricks_app"
    LAKEHOUSE_APP_NAME="telco-support-agent-staging"
    CONFIG_FILE="app_staging.yaml"
elif [[ "$ENVIRONMENT" == "prod" ]]; then
    APP_FOLDER_IN_WORKSPACE="/Workspace/Shared/telco_support_agent/prod/databricks_app"
    LAKEHOUSE_APP_NAME="telco-support-agent-prod"
    CONFIG_FILE="app_prod.yaml"
fi

check_config_file "$CONFIG_FILE"

echo "🚀 Deploying Telco Support Agent UI - $ENVIRONMENT Environment"
echo "================================================"
echo "Environment: $ENVIRONMENT"
echo "Config file: $CONFIG_FILE"
echo "Workspace folder: $APP_FOLDER_IN_WORKSPACE"
echo "App name: $LAKEHOUSE_APP_NAME"
echo "Databricks profile: $DATABRICKS_PROFILE"
echo "================================================"

# Build frontend
echo "📦 Building frontend..."
if [ -d "frontend" ]; then
    cd frontend
    
    # Clean previous build
    rm -rf dist/
    echo "Installing frontend dependencies..."
    npm install
    echo "Building frontend..."
    npm run build
    
    if [ -d "dist" ]; then
        echo "✅ Frontend built successfully"
        cd ..
        
        # Move to static directory in root
        rm -rf static/
        mv frontend/dist static/
        echo "✅ Static files moved to root/static/"
    else
        echo "❌ Frontend build failed - no dist directory created"
        exit 1
    fi
else
    echo "⚠️  No frontend directory found, skipping frontend build"
fi

# Create deployment package
echo "📦 Creating deployment package..."
rm -rf .databricks_app_build/
mkdir -p .databricks_app_build/

# Copy all necessary files, excluding development files
rsync -av \
    --exclude='frontend/' \
    --exclude='node_modules/' \
    --exclude='**/__pycache__/' \
    --exclude='**/*.pyc' \
    --exclude='.*' \
    --exclude='tests/' \
    --exclude='test/' \
    --exclude='deploy.sh' \
    --exclude='app_local.yaml*' \
    --exclude='app_dev.yaml' \
    --exclude='app_staging.yaml' \
    --exclude='app_prod.yaml' \
    --exclude='**/.env*' \
    --exclude='**/venv/' \
    --exclude='**/.venv/' \
    --exclude='.databricks_app_build/' \
    ./ .databricks_app_build/

# Copy the appropriate config file as app.yaml
echo "📝 Using configuration file: $CONFIG_FILE"
cp "$CONFIG_FILE" .databricks_app_build/app.yaml

echo "✅ Deployment package created"

# Upload to workspace
echo "📁 Uploading to workspace..."
databricks workspace delete "$APP_FOLDER_IN_WORKSPACE" --recursive --profile $DATABRICKS_PROFILE 2>/dev/null || true
databricks workspace import-dir .databricks_app_build "$APP_FOLDER_IN_WORKSPACE" --overwrite --profile $DATABRICKS_PROFILE
echo "✅ Files uploaded to workspace"

# Create app if doesn't exist
echo "🚀 Creating/Deploying Databricks application..."
if ! databricks apps get "$LAKEHOUSE_APP_NAME" --profile $DATABRICKS_PROFILE >/dev/null 2>&1; then
  echo "📱 Creating new app: $LAKEHOUSE_APP_NAME"
  databricks apps create "$LAKEHOUSE_APP_NAME" --profile $DATABRICKS_PROFILE
fi

# Deploy the application
echo "🚀 Deploying to app: $LAKEHOUSE_APP_NAME"
databricks apps deploy "$LAKEHOUSE_APP_NAME" \
  --source-code-path "$APP_FOLDER_IN_WORKSPACE" \
  --profile $DATABRICKS_PROFILE

# Cleanup
rm -rf .databricks_app_build/

# Print success message
echo ""
echo "🎉 Deployment completed successfully!"
echo "================================================"
echo "Environment: $ENVIRONMENT"
echo "Endpoint: ${ENVIRONMENT}-telco-customer-support-agent"
echo "App name: $LAKEHOUSE_APP_NAME"
echo "Workspace folder: $APP_FOLDER_IN_WORKSPACE"
echo ""
echo "📱 Check app status:"
echo "   databricks apps get $LAKEHOUSE_APP_NAME --profile $DATABRICKS_PROFILE"
echo ""
echo "📋 View logs:"
echo "   databricks apps logs $LAKEHOUSE_APP_NAME --profile $DATABRICKS_PROFILE"
echo "================================================"