# Telco Support Agent UI - Backend

FastAPI backend service that provides a web interface for the Databricks Telco Support Agent.

## Features

- **Simple REST API** for chat functionality
- **Databricks Integration** with your deployed agent endpoint
- **Customer Management** with demo customer IDs
- **Health Checks** for monitoring
- **CORS Support** for frontend integration
- **Environment-based Configuration**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy example env file and update with your settings:

```bash
cp .env.example .env
```

**Required environment variables:**
- `DATABRICKS_TOKEN`: Your Databricks workspace token
- `DATABRICKS_HOST`: Your Databricks workspace URL (default provided)
- `DATABRICKS_ENDPOINT_NAME`: Your deployed endpoint name (default: telco-customer-support-agent)

### 3. Run the Server

```bash
# development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# or using the main module
python -m app.main
```

API will be available at:
- **API Base**: http://localhost:8000/api
- **Health Check**: http://localhost:8000/health  
- **API Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Chat Endpoints

**POST `/api/chat`**
- Send a message to the telco support agent
- Returns structured response with agent type and tools used

**POST `/api/chat/stream`**  
- Send a message with streaming response (if supported)
- Returns server-sent events stream

### Utility Endpoints

**GET `/api/customers`**
- Get list of demo customer IDs for testing

**GET `/api/health`**
- Check if Databricks endpoint is accessible

**GET `/health`**
- General application health check

## Request Format

```json
{
  "message": "What plan am I currently on?",
  "customer_id": "CUS-10001", 
  "conversation_history": [
    {
      "role": "user",
      "content": "Previous message"
    },
    {
      "role": "assistant", 
      "content": "Previous response"
    }
  ]
}
```

## Response Format

```json
{
  "response": "You are currently on our Premium Family Plan...",
  "agent_type": "account",
  "custom_outputs": {
    "routing": {
      "agent_type": "account"
    },
    "customer": "CUS-10001"
  },
...
}
```

## Configuration

All configuration is handled through environment variables. See `.env.example` for all available options.

Key settings:
- **Databricks Authentication**: Set your workspace token
- **CORS Origins**: Configure allowed frontend URLs  
- **Timeouts**: Adjust request timeout for long-running queries
- **Demo Data**: Customize available customer IDs

## Error Handling

The service includes comprehensive error handling for:
- Databricks API failures
- Authentication errors  
- Network timeouts
- Invalid customer IDs
- Malformed requests

Errors are returned in a consistent format:

```json
{
  "detail": "Error description"
}
```

## Development

### Project Structure

```
app/
├── main.py              # FastAPI application setup
├── config.py            # Configuration management  
├── routes/
│   └── agent.py         # API route handlers
└── services/
    └── telco_agent_service.py  # Databricks integration
```