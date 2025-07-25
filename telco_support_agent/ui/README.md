# Telco Support Agent - Databricks App

A production-grade AI-powered customer support system built entirely on Databricks using Mosaic AI Agent Framework.

## Architecture

- **Frontend**: Vue.js 3 with Vuetify for UI
- **Backend**: FastAPI with async support for agent integration  
- **Agent**: Multi-agent system deploye using Databricks Agent Framework
- **Deployment**: Databricks Lakehouse App

## Quick Start

### Prerequisites

**Databricks Requirements:**
- Databricks workspace with serverless compute enabled
- Deployed telco support agent endpoint: `telco-customer-support-agent`
- Network access to `*.databricksapps.com` domain

**Development Environment:**
- Python 3.11+ 
- Node.js 18+
- Databricks CLI v0.229.0+

### 1. Install dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Local Development

```bash
# copy example configuration
cp app_local.yaml.example app_local.yaml

# edit app_local.yaml with your Databricks credentials
# required values:
# - DATABRICKS_HOST: Your workspace URL
# - DATABRICKS_TOKEN: Your access token
# - DATABRICKS_ENDPOINT_NAME: Your agent endpoint name
```

**Example `app_local.yaml`:**
```yaml
command: ["uvicorn", "backend.app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
static_dir: static
env:
  - name: 'DATABRICKS_HOST'
    value: 'https://your-workspace.cloud.databricks.com'
  - name: 'DATABRICKS_TOKEN'
    value: 'your-token-here'
  - name: 'DATABRICKS_ENDPOINT_NAME'
    value: 'telco-customer-support-agent'
```

### 3. Setup Frontend

```bash
cd frontend

# install dependencies
npm install

# start development server (runs on http://localhost:5173)
npm run dev
```

### 4. Start Backend Locally

```bash
# From the root ui/ directory
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Using the Application

1. **Select Customer**: Choose a demo customer ID from the dropdown
2. **Ask Questions**: Type telecom-related questions like:
   - "What plan am I currently on?"
   - "Show me my billing for this month"
   - "What devices do I have?"
   - "My WiFi isn't working, can you help?"
3. **View Intelligence**: Toggle the intelligence panel to see agent reasoning and tool usage
4. **Provide Feedback**: Use thumbs up/down on responses to improve the system

## Deployment to Databricks

### Environment-Specific Deployment

The deployment system supports separate dev, staging, and prod environments with different endpoints.

### Prerequisites for Deployment

**Databricks CLI configured:**
```bash
databricks configure
# Or use OAuth: databricks auth login
```

### Environment Configuration

The system uses environment-specific configuration files:

- **Dev Environment**: `app_dev.yaml`
  - Endpoint: `dev-telco-customer-support-agent`
  - Workspace: `/Workspace/Shared/telco_support_agent/dev/databricks_app`
  - App Name: `telco-support-agent-dev`

- **Staging Environment**: `app_staging.yaml`
  - Endpoint: `staging-telco-customer-support-agent`
  - Workspace: `/Workspace/Shared/telco_support_agent/staging/databricks_app`
  - App Name: `telco-support-agent-staging`

- **Prod Environment**: `app_prod.yaml`
  - Endpoint: `prod-telco-customer-support-agent`
  - Workspace: `/Workspace/Shared/telco_support_agent/prod/databricks_app`
  - App Name: `telco-support-agent-prod`

### Deploy Commands

**Deploy to Development:**
```bash
# Deploy to dev environment with default profile
./deploy.sh dev

# Deploy to dev environment with specific profile
./deploy.sh dev dev-profile
```

**Deploy to Staging:**
```bash
# Deploy to staging environment with default profile
./deploy.sh staging

# Deploy to staging environment with specific profile
./deploy.sh staging staging-profile
```

**Deploy to Production:**
```bash
# Deploy to prod environment with default profile
./deploy.sh prod

# Deploy to prod environment with specific profile
./deploy.sh prod PRODUCTION
```

### Deployment Process

The deployment script will:

1. **Validate Environment**: Ensure environment is 'dev', 'staging', or 'prod'
2. **Build Frontend**: Compile Vue.js application
3. **Create Package**: Bundle application files (excluding dev files)
4. **Copy Configuration**: Use environment-specific config as `app.yaml`
5. **Upload to Workspace**: Deploy to environment-specific workspace folder
6. **Create/Update App**: Deploy as Databricks Lakehouse App

### Environment Variables

Each environment config file sets appropriate variables:

**Development (`app_dev.yaml`):**
- `ENV=dev`
- `DATABRICKS_ENDPOINT_NAME=dev-telco-customer-support-agent`

**Staging (`app_staging.yaml`):**
- `ENV=staging`
- `DATABRICKS_ENDPOINT_NAME=staging-telco-customer-support-agent`

**Production (`app_prod.yaml`):**
- `ENV=prod`
- `DATABRICKS_ENDPOINT_NAME=prod-telco-customer-support-agent`

### Monitoring Deployments

**Check app status:**
```bash
# Dev environment
databricks apps get telco-support-agent-dev --profile dev-profile

# Staging environment
databricks apps get telco-support-agent-staging --profile staging-profile

# Prod environment  
databricks apps get telco-support-agent-prod --profile PRODUCTION
```

**View logs:**
```bash
# Dev environment
databricks apps logs telco-support-agent-dev --profile dev-profile

# Staging environment
databricks apps logs telco-support-agent-staging --profile staging-profile

# Prod environment
databricks apps logs telco-support-agent-prod --profile PRODUCTION
```

### Best Practices

1. **Use Separate Profiles**: Configure different Databricks CLI profiles for dev, staging, and prod
2. **Follow Promotion Path**: Always test dev → staging → prod
3. **Environment Verification**: Always verify you're deploying to the correct environment
4. **Config Management**: Keep environment configs in version control
5. **Rollback Plan**: Know how to quickly rollback if issues occur

### Typical Workflow

```bash
# 1. Develop and test locally
npm run dev  # frontend
uvicorn backend.app.main:app --reload  # backend

# 2. Deploy to dev for integration testing
./deploy.sh dev

# 3. After dev testing, promote to staging
./deploy.sh staging

# 4. After staging validation, deploy to production
./deploy.sh prod PRODUCTION
```

### Troubleshooting

**Common Issues:**

1. **Config File Missing**: Ensure `app_dev.yaml`, `app_staging.yaml`, and `app_prod.yaml` exist
2. **Wrong Profile**: Verify Databricks CLI profile has access to target workspace
3. **Endpoint Access**: Confirm agent endpoints exist and are accessible
4. **Workspace Permissions**: Ensure profile has write access to workspace folders

## Configuration

### Environment Variables

**Production (`app.yaml`):**
- `ENV=prod`

**Development (`app_local.yaml`):**
- `DATABRICKS_HOST`: Your Databricks workspace URL
- `DATABRICKS_TOKEN`: Your personal access token or service principal token
- `DATABRICKS_ENDPOINT_NAME`: Name of your deployed agent endpoint
- `CORS_ORIGINS`: Frontend URLs for CORS (development only)
- `DEMO_CUSTOMER_IDS`: Comma-separated customer IDs for testing

### Databricks App Resources

The app requires these Databricks resources:
- **Model Serving Endpoint**: Databricks model serving endpoint deployed using Agent Framework
- **Unity Catalog Access**: For customer data and tools
- **SQL Warehouse**: For structured data queries (used by agent)

Configure these in the Databricks Apps UI under the "Resources" tab.

## Testing

TODO

## Monitoring and Logs

### View App Logs
In Databricks workspace:
1. Go to **Compute** > **Apps**
2. Click app name
3. Go to **Logs** tab

## Development Guide

### Project Structure
```
ui/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── config.py          # Configuration management
│   │   ├── routes/            # API endpoints
│   │   └── services/          # Business logic
│   └── requirements.txt       # Backend dependencies
├── frontend/                   # Vue.js frontend
│   ├── src/
│   │   ├── components/        # Vue components
│   │   ├── services/          # API integration
│   │   └── types/             # TypeScript definitions
│   ├── package.json
│   └── vite.config.ts         # Build configuration
├── app.yaml                   # Production config
├── app_local.yaml.example     # Development config template
├── deploy.sh                  # Deployment script
└── requirements.txt           # Root Python dependencies
```

### Adding New Features

**Frontend (Vue.js):**
1. Add components in `frontend/src/components/`
2. Update API calls in `frontend/src/services/api.ts`
3. Test locally with `npm run dev`

**Backend (FastAPI):**
1. Add routes in `backend/app/routes/`
2. Add business logic in `backend/app/services/`
3. Test with `uvicorn backend.app.main:app --reload`

### Code Quality
- **Frontend**: ESLint, Prettier, TypeScript
- **Backend**: Black, isort, flake8, mypy
- **Pre-commit hooks**: Automatic formatting and linting

## Demo Scenarios

### Customer Account Queries
- "What plan am I currently on?"
- "Is autopay enabled on my account?"
- "When does my contract expire?"

### Billing Questions  
- "Show me my bill for this month"
- "Why is my bill higher than usual?"
- "How much data did I use?"

### Technical Support
- "My phone won't connect to WiFi"
- "I can't make calls but data works"
- "How do I reset my voicemail?"

### Product Information
- "What plans are available?"
- "Compare iPhone vs Samsung devices"
- "Are there any current promotions?"