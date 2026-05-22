"""Mimo2API Python - Main entry point"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from app.routes import router
from app.config import config_manager

# Create FastAPI app
app = FastAPI(
    title="Mimo2API",
    description="Xiaomi MiMo AI to OpenAI-compatible API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)

# Static files directory
web_dir = Path(__file__).parent / "web"

# Serve admin interface
@app.get("/")
async def serve_admin():
    """Serve admin interface"""
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Admin interface not found"}


def main():
    """Main function"""
    # Get port config
    port = int(os.getenv("PORT", "8080"))

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    Mimo2API Python                       ║
║        Xiaomi MiMo AI → OpenAI-Compatible API           ║
╚══════════════════════════════════════════════════════════╝

🚀 Starting server...
📍 URL: http://localhost:{port}
📊 Admin UI: http://localhost:{port}
📡 API endpoint: http://localhost:{port}/v1/chat/completions
📖 API docs: http://localhost:{port}/docs

Config:
  - API Keys: {len(config_manager.config.api_keys.split(','))}
  - MiMo Accounts: {len(config_manager.config.mimo_accounts)}

Press Ctrl+C to stop
""")

    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
