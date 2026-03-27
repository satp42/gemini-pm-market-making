"""FastAPI application entry point with bot lifecycle management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.db.database import close_db, init_db

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup:
    - Load settings
    - Initialize database engine
    - Start the bot loop as a background task (when engine is implemented)

    On shutdown:
    - Stop the bot loop
    - Close database connections
    """
    settings = get_settings()
    app.state.settings = settings

    # Initialize database
    await init_db(settings)
    logger.info(
        "Application started — env=%s, cycle=%ds",
        settings.gemini.env,
        settings.bot.bot_cycle_seconds,
    )

    # Bot loop will be started here once engine/loop.py is implemented
    # Example:
    #   from src.engine.loop import BotLoop
    #   bot_loop = BotLoop(settings)
    #   app.state.bot_loop = bot_loop
    #   await bot_loop.start()

    yield

    # Shutdown: stop bot loop and close DB
    # if hasattr(app.state, "bot_loop"):
    #     await app.state.bot_loop.stop()
    await close_db()
    logger.info("Application shut down cleanly")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Gemini PM Market Maker",
        description="Avellaneda-Stoikov market-making bot for Gemini Prediction Markets",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.app.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "env": settings.gemini.env}

    # Mount API router (placeholder import — router.py will be implemented)
    try:
        from src.api.router import router as api_router

        app.include_router(api_router, prefix="/api")
    except ImportError:
        logger.warning("API router not yet implemented, skipping mount")

    # Mount WebSocket endpoint (placeholder — ws.py will be implemented)
    try:
        from src.api.ws import router as ws_router

        app.include_router(ws_router)
    except ImportError:
        logger.warning("WebSocket router not yet implemented, skipping mount")

    return app


app = create_app()
