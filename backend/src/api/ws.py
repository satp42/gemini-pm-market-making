"""WebSocket streaming endpoint for real-time dashboard updates."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import desc, select
from starlette.websockets import WebSocketState

from src.db.database import get_session
from src.db.models import PnlSnapshot, PositionSnapshot, Quote

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts data to all clients.

    The bot loop calls ``broadcast()`` on each tick to push real-time updates
    to every connected dashboard client.
    """

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and add it to the active list."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "WebSocket client connected. Total connections: %d",
            len(self.active_connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the active list."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            "WebSocket client disconnected. Total connections: %d",
            len(self.active_connections),
        )

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients.

        Dead connections are silently removed from the active list.
        """
        dead: list[WebSocket] = []

        for connection in self.active_connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(data)
                else:
                    dead.append(connection)
            except Exception:
                logger.debug("Failed to send to WebSocket client, marking as dead")
                dead.append(connection)

        for connection in dead:
            self.disconnect(connection)

    async def send_personal(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        """Send a JSON message to a single client."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
        except Exception:
            logger.debug("Failed to send personal message to WebSocket client")

    @property
    def connection_count(self) -> int:
        """Return the number of active connections."""
        return len(self.active_connections)


# Module-level default manager instance.
# In production the lifespan handler creates and stores one on app.state;
# this default is used as a fallback in tests.
manager = ConnectionManager()


async def _build_initial_snapshot(app_state: Any) -> dict[str, Any]:
    """Build an initial state snapshot to send when a client first connects.

    Queries the database for the latest markets, positions, and P&L,
    and reads bot status from app.state.
    """
    markets: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    pnl: dict[str, Any] = {}
    status: dict[str, Any] = {
        "running": False,
        "uptime": 0.0,
        "activeMarkets": 0,
        "environment": "sandbox",
    }

    # Read bot status from app.state
    bot_loop = getattr(app_state, "bot_loop", None)
    settings = getattr(app_state, "settings", None)

    if bot_loop is not None:
        running = getattr(bot_loop, "running", False)
        started_at = getattr(bot_loop, "started_at", None)
        uptime = (time.time() - started_at.timestamp()) if (started_at and running) else 0.0
        active_symbols = getattr(bot_loop, "active_symbols", [])
        status = {
            "running": running,
            "uptime": uptime,
            "activeMarkets": len(active_symbols),
            "environment": settings.gemini.env if settings else "sandbox",
        }

    # Get symbol titles from bot loop
    titles: dict[str, str] = {}
    if bot_loop is not None:
        titles = getattr(bot_loop, "symbol_titles", {})

    # Query DB for latest snapshot data
    try:
        async for session in get_session():
            # Latest quote per symbol
            latest_quotes_subq = (
                select(Quote.symbol, Quote.timestamp)
                .distinct(Quote.symbol)
                .order_by(Quote.symbol, desc(Quote.timestamp))
                .subquery()
            )
            quote_stmt = (
                select(Quote)
                .join(
                    latest_quotes_subq,
                    (Quote.symbol == latest_quotes_subq.c.symbol)
                    & (Quote.timestamp == latest_quotes_subq.c.timestamp),
                )
                .order_by(Quote.symbol)
            )
            quote_result = await session.execute(quote_stmt)
            for q in quote_result.scalars().all():
                markets.append(
                    {
                        "symbol": q.symbol,
                        "eventTitle": titles.get(q.symbol, ""),
                        "midPrice": float(q.mid_price),
                        "reservationPrice": float(q.reservation_price),
                        "bidPrice": float(q.bid_price),
                        "askPrice": float(q.ask_price),
                        "spread": float(q.spread),
                        "inventory": float(q.inventory),
                        "sigmaSquared": float(q.sigma_sq),
                        "gamma": float(q.gamma),
                        "timeRemaining": float(q.t_minus_t),
                    }
                )

            # Latest position per symbol
            latest_pos_subq = (
                select(PositionSnapshot.symbol, PositionSnapshot.timestamp)
                .distinct(PositionSnapshot.symbol)
                .order_by(PositionSnapshot.symbol, desc(PositionSnapshot.timestamp))
                .subquery()
            )
            pos_stmt = (
                select(PositionSnapshot)
                .join(
                    latest_pos_subq,
                    (PositionSnapshot.symbol == latest_pos_subq.c.symbol)
                    & (PositionSnapshot.timestamp == latest_pos_subq.c.timestamp),
                )
                .order_by(PositionSnapshot.symbol)
            )
            pos_result = await session.execute(pos_stmt)
            for p in pos_result.scalars().all():
                positions.append(
                    {
                        "symbol": p.symbol,
                        "eventTitle": titles.get(p.symbol, ""),
                        "yesQuantity": float(p.yes_quantity),
                        "noQuantity": float(p.no_quantity),
                        "netInventory": float(p.net_inventory),
                        "unrealizedPnl": float(p.unrealized_pnl),
                        "timestamp": p.timestamp.isoformat(),
                    }
                )

            # Latest P&L snapshot
            pnl_stmt = (
                select(PnlSnapshot).order_by(desc(PnlSnapshot.timestamp)).limit(1)
            )
            pnl_result = await session.execute(pnl_stmt)
            pnl_row = pnl_result.scalar_one_or_none()
            if pnl_row is not None:
                pnl = {
                    "timestamp": pnl_row.timestamp.isoformat(),
                    "realizedPnl": float(pnl_row.total_realized_pnl),
                    "unrealizedPnl": float(pnl_row.total_unrealized_pnl),
                    "totalExposure": float(pnl_row.total_exposure),
                }

    except Exception:
        logger.exception("Failed to build initial WebSocket snapshot from DB")

    return {
        "type": "tick",
        "data": {
            "markets": markets,
            "positions": positions,
            "pnl": pnl,
            "status": status,
        },
    }


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard streaming.

    On connection:
      1. Accept and register the client.
      2. Send an initial state snapshot.
      3. Keep the connection open, listening for client messages
         (e.g., pings) until the client disconnects.

    The bot loop pushes updates by calling ``manager.broadcast()`` externally.
    """
    # Use the manager stored in app.state if available, else fall back to module-level
    ws_manager: ConnectionManager = getattr(
        websocket.app.state, "ws_manager", manager
    )

    await ws_manager.connect(websocket)

    try:
        # Send initial snapshot
        snapshot = await _build_initial_snapshot(websocket.app.state)
        await ws_manager.send_personal(websocket, snapshot)

        # Keep connection alive and listen for client messages
        while True:
            # Await messages from the client (pings, config changes, etc.)
            # This also keeps the connection alive and detects disconnection.
            data = await websocket.receive_text()

            # Clients can send a ping; respond with pong
            if data == "ping":
                await ws_manager.send_personal(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected normally")
    except Exception:
        ws_manager.disconnect(websocket)
        logger.exception("WebSocket connection error")
