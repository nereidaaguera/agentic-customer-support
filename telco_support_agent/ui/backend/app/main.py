"""Main FastAPI application for Telco Support Agent UI."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from databricks.sdk import WorkspaceClient
from dbdemos_tracker import Tracker
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routes.agent import router as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"Starting Telco Support Agent UI on {settings.environment}")
    print(f"Databricks endpoint: {settings.databricks_endpoint}")
    print(f"App port: {settings.port}")
    print(f"Working directory: {os.getcwd()}")

    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        print(f"Static directory found: {static_dir}")
        print(f"Static files: {list(static_dir.iterdir())[:5]}...")  # First 5 files
    else:
        print(f"Static directory not found at: {static_dir}")

    yield

    print("Shutting down Telco Support Agent UI")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Telco Support Agent UI",
        description="Web interface for Databricks Telco Support Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # dbdemos tracking middleware
    @app.middleware("http")
    async def track_app_usage(request: Request, call_next):
        """Track app usage with dbdemos-tracker."""
        user_email = request.headers.get("X-Forwarded-Email")

        # track app view
        if user_email:
            try:
                org = WorkspaceClient().get_workspace_id()
                tracker = Tracker(org)
                tracker.track_app_view(
                    user_email=user_email,
                    app_name="agentic-customer-support",
                    path=str(request.url.path),
                )
            except Exception as e:
                print(f"Tracking error: {e}")

        response = await call_next(request)
        return response

    app.include_router(agent_router, prefix="/api")

    # Serve static files (frontend build)
    possible_static_dirs = [
        Path(__file__).parent.parent.parent / "static",  # Root level
        Path(__file__).parent.parent / "static",  # Backend level
        Path.cwd() / "static",  # Current working directory
    ]

    static_dir = None
    for dir_path in possible_static_dirs:
        if dir_path.exists() and dir_path.is_dir():
            static_dir = dir_path
            print(f"Using static directory: {static_dir}")
            break

    if static_dir:
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
        print(f"Static files mounted from: {static_dir}")
    else:
        print("Warning: No static directory found, serving API only")

    return app


# Create the app instance
app = create_app()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "telco-support-agent-ui"}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104
        port=settings.port,
        reload=False,
    )
