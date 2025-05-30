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
# Copy example configuration
cp app_local.yaml.example app_local.yaml

# Edit app_local.yaml with your Databricks credentials
# Required values:
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

# Install dependencies
npm install

# Start development server (runs on http://localhost:5173)
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

### Prerequisites for Deployment

**Databricks CLI configured:**
```bash
databricks configure
# Or use OAuth: databricks auth login
```

### Deploy

```bash
# Create the app in Databricks
databricks apps create telco-support-agent

# Deploy with default settings
./deploy.sh

# Deploy to specific workspace location
./deploy.sh "/Workspace/your-path/telco-support-agent" "my-telco-app"

# Deploy using specific Databricks profile
./deploy.sh "/Workspace/your-path/telco-support-agent" "my-telco-app" "PRODUCTION"
```

### Manual Deployment Steps

If you prefer manual deployment:

```bash
# 1. Build frontend
cd frontend
npm run build
cd ..

# 2. Move static files
rm -rf static/
mv frontend/dist static/

# 3. Create app in Databricks
databricks apps create telco-support-agent

# 4. Sync code to workspace
databricks workspace import-dir . "/Workspace/your-path/telco-support-agent" --overwrite

# 5. Deploy the app
databricks apps deploy telco-support-agent \
  --source-code-path "/Workspace/your-path/telco-support-agent"
```

## Configuration

### Environment Variables

**Production (`app.yaml`):**
- `ENV=prod`
- `TELCO_SUPPORT_AGENT_ENV=prod`

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
