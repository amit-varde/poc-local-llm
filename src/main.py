"""Main entry point for the Local LLM Pipeline."""

from pipeline.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    from pipeline.config import get_config

    config = get_config()
    settings = config.load_settings()

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload
    )