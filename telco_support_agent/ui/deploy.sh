#!/bin/bash

# Accept parameters
APP_FOLDER_IN_WORKSPACE=${1:-"/Workspace/telco-support-agent"}
LAKEHOUSE_APP_NAME=${2:-"telco-support-agent"}
DATABRICKS_PROFILE=${3:-"DEFAULT"}

echo "🚀 Deploying Telco Support Agent"
echo "================================================"
echo "Workspace folder: $APP_FOLDER_IN_WORKSPACE"
echo "App name: $LAKEHOUSE_APP_NAME"
echo "Databricks profile: $DATABRICKS_PROFILE"
echo "================================================"

# Frontend build and import
echo "📦 Building frontend..."
(
  if [ -d "frontend" ]; then
    cd frontend
    # Ensure clean build
    rm -rf dist/
    echo "Installing frontend dependencies..."
    npm install
    echo "Building frontend..."
    npm run build
    
    if [ -d "dist" ]; then
      echo "Moving dist to static..."
      rm -rf ../static/
      mv dist ../static
      echo "✅ Frontend built successfully"
      
      # Upload static files to workspace
      echo "📁 Uploading static files to workspace..."
      databricks workspace delete "$APP_FOLDER_IN_WORKSPACE/static" --recursive --profile $DATABRICKS_PROFILE 2>/dev/null || true
      databricks workspace import-dir ../static "$APP_FOLDER_IN_WORKSPACE/static" --overwrite --profile $DATABRICKS_PROFILE
      echo "✅ Static files uploaded"
    else
      echo "❌ Frontend build failed - no dist directory created"
      exit 1
    fi
  else
    echo "⚠️  No frontend directory found, skipping frontend build"
  fi
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

# Create app if it doesn't exist, then deploy
echo "🚀 Creating/Deploying Databricks application..."

# Check if app exists, create if not
if ! databricks apps get "$LAKEHOUSE_APP_NAME" --profile $DATABRICKS_PROFILE >/dev/null 2>&1; then
  echo "📱 Creating new app: $LAKEHOUSE_APP_NAME"
  databricks apps create "$LAKEHOUSE_APP_NAME" --profile $DATABRICKS_PROFILE
fi

# Deploy the application
echo "🚀 Deploying to app: $LAKEHOUSE_APP_NAME"
databricks apps deploy "$LAKEHOUSE_APP_NAME" \
  --source-code-path "$APP_FOLDER_IN_WORKSPACE" \
  --profile $DATABRICKS_PROFILE

# Print success message
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