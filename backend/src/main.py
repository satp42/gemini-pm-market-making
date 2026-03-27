"""FastAPI application entry point with bot lifecycle management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router as api_router
from src.api.ws import ConnectionManager
from src.api.ws import router as ws_router
from src.config import get_settings
from src.db.database import close_db, get_session_factory, init_db
from src.gemini.client import GeminiClient

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
    - Create GeminiClient
    - Create ConnectionManager for WebSocket broadcasting
    - Create and start the BotLoop (when engine is implemented)

    On shutdown:
    - Stop the bot loop
    - Close the Gemini client
    - Close database connections
    """
    settings = get_settings()
    app.state.settings = settings
    app.state.config_overrides = {}

    # Initialize optimization progress (T028)
    from src.engine.optimizer import OptimizationProgress

    app.state.optimization_progress = OptimizationProgress()

    # Initialize database
    await init_db(settings)

    # Create Gemini API client
    gemini_client = GeminiClient(settings)
    app.state.gemini_client = gemini_client

    # Create WebSocket connection manager
    ws_manager = ConnectionManager()
    app.state.ws_manager = ws_manager

    # Create and start the bot loop when engine is implemented
    try:
        from src.engine.loop import BotLoop

        bot_loop = BotLoop(
            settings=settings,
            client=gemini_client,
            session_factory=get_session_factory(),
            config_overrides=app.state.config_overrides,
        )
        app.state.bot_loop = bot_loop

        # Set on_tick callback to broadcast to WebSocket clients
        bot_loop.on_tick = ws_manager.broadcast

        logger.info("Bot loop initialized and ready")
    except (ImportError, TypeError, AttributeError) as exc:
        logger.warning("Bot loop not available (engine not yet implemented): %s", exc)
        app.state.bot_loop = None

    logger.info(
        "Application started — env=%s, cycle=%ds",
        settings.gemini.env,
        settings.bot.bot_cycle_seconds,
    )

    yield

    # Shutdown: stop bot loop, close Gemini client, close DB
    bot_loop = getattr(app.state, "bot_loop", None)
    if bot_loop is not None and getattr(bot_loop, "running", False):
        try:
            await bot_loop.stop()
            logger.info("Bot loop stopped")
        except Exception:
            logger.exception("Error stopping bot loop during shutdown")

    try:
        await gemini_client.close()
        logger.info("Gemini client closed")
    except Exception:
        logger.exception("Error closing Gemini client during shutdown")

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

    # Mount API router
    app.include_router(api_router, prefix="/api")

    # Mount WebSocket router
    app.include_router(ws_router)

    return app


app = create_app()
