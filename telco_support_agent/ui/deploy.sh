#!/bin/bash

APP_FOLDER_IN_WORKSPACE=${1:-"/Workspace/telco-support-agent-ui"}
LAKEHOUSE_APP_NAME=${2:-"telco-support-agent-ui"}
DATABRICKS_PROFILE=${3:-"DEFAULT"}

echo "🚀 Deploying Telco Support Agent UI"
echo "================================================"
echo "Workspace folder: $APP_FOLDER_IN_WORKSPACE"
echo "App name: $LAKEHOUSE_APP_NAME"
echo "Databricks profile: $DATABRICKS_PROFILE"
echo "================================================"

# Build frontend
echo "📦 Building frontend..."
if [ -d "frontend" ]; then
    cd frontend
    
    # clean previous build
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
    --exclude='**/.env*' \
    --exclude='**/venv/' \
    --exclude='**/.venv/' \
    --exclude='.databricks_app_build/' \
    ./ .databricks_app_build/

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
echo "App name: $LAKEHOUSE_APP_NAME"
echo "Workspace folder: $APP_FOLDER_IN_WORKSPACE"
echo ""
echo "📱 Check app status:"
echo "   databricks apps get $LAKEHOUSE_APP_NAME --profile $DATABRICKS_PROFILE"
echo ""
echo "📋 View logs:"
echo "   databricks apps logs $LAKEHOUSE_APP_NAME --profile $DATABRICKS_PROFILE"
echo "================================================"