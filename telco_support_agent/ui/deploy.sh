#!/bin/bash

# Accept parameters
APP_FOLDER_IN_WORKSPACE=${1:-"/Workspace/Shared/telco_support_agent"}
LAKEHOUSE_APP_NAME=${2:-"telco-support-agent"}
DATABRICKS_PROFILE=${3:-"DEFAULT"}

echo "🚀 Deploying Telco Support Agent"
echo "================================================"
echo "Workspace folder: $APP_FOLDER_IN_WORKSPACE"
echo "App name: $LAKEHOUSE_APP_NAME"
echo "Databricks profile: $DATABRICKS_PROFILE"
echo "================================================"

echo "📦 Building frontend..."
(
  cd frontend
  rm -rf dist/
  npm install
  npm run build
  rm -rf ../static/
  mv dist ../static
  echo "✅ Frontend built successfully"
  
  # upload static files to workspace
  echo "📁 Uploading static files to workspace..."
  databricks workspace delete "$APP_FOLDER_IN_WORKSPACE/static" --recursive --profile $DATABRICKS_PROFILE 2>/dev/null || true
  databricks workspace import-dir ../static "$APP_FOLDER_IN_WORKSPACE/static" --overwrite --profile $DATABRICKS_PROFILE
  echo "✅ Static files uploaded"
) &

# Backend packaging
echo "🐍 Packaging backend..."
(
  rm -rf build
  mkdir -p build
  
  # Copy backend and configuration files
  rsync -av \
    --exclude='**/__pycache__/' \
    --exclude='**/*.pyc' \
    --exclude='.*' \
    --exclude='tests/' \
    --exclude='test/' \
    --exclude='frontend/' \
    --exclude='static/' \
    --exclude='build/' \
    --exclude='deploy.sh' \
    --exclude='app_local.yaml*' \
    --exclude='**/.env*' \
    --exclude='**/venv/' \
    --exclude='**/.venv/' \
    ./ build/
    
  echo "✅ Backend packaged successfully"
  
  # Upload backend to workspace
  echo "📁 Uploading backend to workspace..."
  databricks workspace delete "$APP_FOLDER_IN_WORKSPACE" --recursive --profile $DATABRICKS_PROFILE 2>/dev/null || true
  databricks workspace import-dir build "$APP_FOLDER_IN_WORKSPACE" --overwrite --profile $DATABRICKS_PROFILE
  echo "✅ Backend uploaded"
  
  # Cleanup
  rm -rf build
) &

# Wait for both background processes to finish
wait

# Deploy the application
echo "🚀 Deploying Databricks application..."
databricks apps deploy "$LAKEHOUSE_APP_NAME" --profile $DATABRICKS_PROFILE

echo ""
echo "🎉 Deployment completed successfully!"
echo "================================================"
echo "App name: $LAKEHOUSE_APP_NAME"
echo "Workspace folder: $APP_FOLDER_IN_WORKSPACE"
echo ""
echo "📱 Open the app:"
echo "   https://db-ml-models-prod-us-west.cloud.databricks.com/apps/$LAKEHOUSE_APP_NAME"
echo ""
echo "⚙️  Manage the app:"
echo "   Go to Compute > Apps in your Databricks workspace"
echo "================================================"