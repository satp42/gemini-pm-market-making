"""REST API endpoints for the market-making bot dashboard."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session, get_session_factory
from src.db.models import OrderRecord, PnlSnapshot, PositionSnapshot, Quote
from src.engine.optimizer import run_theta_optimization
from src.gemini.client import GeminiAPIError

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic response/request schemas
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    running: bool
    uptime: float
    activeMarkets: int
    environment: str


class MarketSummary(BaseModel):
    symbol: str
    eventTitle: str = ""
    midPrice: float
    reservationPrice: float
    bidPrice: float
    askPrice: float
    spread: float
    inventory: float
    sigmaSquared: float
    gamma: float
    timeRemaining: float
    # Performative market-making fields (T024)
    xi: float | None = None
    theta0: float | None = None
    theta1: float | None = None
    theta2: float | None = None
    quotingMode: str | None = None


class QuoteHistoryEntry(BaseModel):
    id: int
    timestamp: str
    midPrice: float
    reservationPrice: float
    bidPrice: float
    askPrice: float
    spread: float
    inventory: float
    sigmaSquared: float
    gamma: float
    timeRemaining: float
    # Performative market-making fields (T024)
    xi: float | None = None
    theta0: float | None = None
    theta1: float | None = None
    theta2: float | None = None
    quotingMode: str | None = None


class OrderEntry(BaseModel):
    id: int
    timestamp: str
    symbol: str
    geminiOrderId: int
    side: str
    outcome: str
    price: float
    quantity: float
    status: str
    fillPrice: float | None


class TradeEntry(BaseModel):
    id: int
    timestamp: str
    symbol: str
    side: str
    outcome: str
    price: float
    quantity: float
    fillPrice: float | None


class MarketDetail(BaseModel):
    symbol: str
    quoteHistory: list[QuoteHistoryEntry]
    recentTrades: list[TradeEntry]
    currentQuote: MarketSummary | None


class PnlEntry(BaseModel):
    timestamp: str
    realizedPnl: float
    unrealizedPnl: float
    totalExposure: float


class PositionEntry(BaseModel):
    symbol: str
    eventTitle: str = ""
    yesQuantity: float
    noQuantity: float
    netInventory: float
    unrealizedPnl: float
    timestamp: str


class ConfigUpdateRequest(BaseModel):
    gamma: float | None = None
    k: float | None = None
    sigma_default: float | None = None
    variance_window: int | None = None
    bot_cycle_seconds: int | None = None
    scanner_cycle_seconds: int | None = None
    min_spread: float | None = None
    max_inventory_per_symbol: int | None = None
    max_total_exposure: int | None = None
    risk_widen_threshold: float | None = None
    # Performative market-making fields (T023)
    quoting_mode: str | None = None
    xi_default: float | None = None
    xi_min_trades: int | None = None
    xi_clamp_min: float | None = None
    xi_clamp_max: float | None = None
    q_ref: float | None = None

    @field_validator("quoting_mode")
    @classmethod
    def validate_quoting_mode(cls, v: str | None) -> str | None:
        if v is not None and v not in ("as", "performative", "theta"):
            raise ValueError("quoting_mode must be one of: as, performative, theta")
        return v


# ---------------------------------------------------------------------------
# Helper to get bot loop safely
# ---------------------------------------------------------------------------


def _performative_fields(q) -> dict:
    """Extract nullable-float performative fields from a quote record."""
    return dict(
        xi=float(q.xi) if q.xi is not None else None,
        theta0=float(q.theta0) if q.theta0 is not None else None,
        theta1=float(q.theta1) if q.theta1 is not None else None,
        theta2=float(q.theta2) if q.theta2 is not None else None,
        quotingMode=q.quoting_mode,
    )


def _get_bot_loop(request: Request) -> Any:
    """Return the bot loop from app.state, or None if not set."""
    return getattr(request.app.state, "bot_loop", None)


def _get_settings(request: Request) -> Any:
    """Return settings from app.state."""
    return getattr(request.app.state, "settings", None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request) -> StatusResponse:
    """Return current bot status including uptime and active market count."""
    bot_loop = _get_bot_loop(request)
    settings = _get_settings(request)

    running = False
    uptime = 0.0
    active_markets = 0

    if bot_loop is not None:
        running = getattr(bot_loop, "running", False)
        started_at = getattr(bot_loop, "started_at", None)
        if started_at is not None and running:
            uptime = time.time() - started_at.timestamp()
        active_symbols = getattr(bot_loop, "active_symbols", [])
        active_markets = len(active_symbols)

    environment = "sandbox"
    if settings is not None:
        environment = settings.gemini.env

    return StatusResponse(
        running=running,
        uptime=uptime,
        activeMarkets=active_markets,
        environment=environment,
    )


@router.get("/markets", response_model=list[MarketSummary])
async def get_markets(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> list[MarketSummary]:
    """Return the latest quote per symbol for all actively quoted markets."""
    # Subquery to get the latest quote timestamp per symbol
    latest_subq = (
        select(
            Quote.symbol,
            Quote.timestamp,
        )
        .distinct(Quote.symbol)
        .order_by(Quote.symbol, desc(Quote.timestamp))
        .subquery()
    )

    stmt = (
        select(Quote)
        .join(
            latest_subq,
            (Quote.symbol == latest_subq.c.symbol)
            & (Quote.timestamp == latest_subq.c.timestamp),
        )
        .order_by(Quote.symbol)
    )

    result = await session.execute(stmt)
    quotes = result.scalars().all()

    bot_loop = _get_bot_loop(request)
    titles = getattr(bot_loop, "symbol_titles", {}) if bot_loop else {}

    return [
        MarketSummary(
            symbol=q.symbol,
            eventTitle=titles.get(q.symbol, ""),
            midPrice=float(q.mid_price),
            reservationPrice=float(q.reservation_price),
            bidPrice=float(q.bid_price),
            askPrice=float(q.ask_price),
            spread=float(q.spread),
            inventory=float(q.inventory),
            sigmaSquared=float(q.sigma_sq),
            gamma=float(q.gamma),
            timeRemaining=float(q.t_minus_t),
            **_performative_fields(q),
        )
        for q in quotes
    ]


@router.get("/markets/{symbol}", response_model=MarketDetail)
async def get_market_detail(
    symbol: str,
    session: AsyncSession = Depends(get_session),
) -> MarketDetail:
    """Return detail view for a single market: quote history, recent trades."""
    # Quote history: last 100 entries for this symbol
    quote_stmt = (
        select(Quote)
        .where(Quote.symbol == symbol)
        .order_by(desc(Quote.timestamp))
        .limit(100)
    )
    quote_result = await session.execute(quote_stmt)
    quotes = quote_result.scalars().all()

    if not quotes:
        raise HTTPException(status_code=404, detail=f"No data found for symbol '{symbol}'")

    quote_history = [
        QuoteHistoryEntry(
            id=q.id,
            timestamp=q.timestamp.isoformat(),
            midPrice=float(q.mid_price),
            reservationPrice=float(q.reservation_price),
            bidPrice=float(q.bid_price),
            askPrice=float(q.ask_price),
            spread=float(q.spread),
            inventory=float(q.inventory),
            sigmaSquared=float(q.sigma_sq),
            gamma=float(q.gamma),
            timeRemaining=float(q.t_minus_t),
            **_performative_fields(q),
        )
        for q in quotes
    ]

    # Current quote is the most recent one
    latest_quote = quotes[0]
    current_quote = MarketSummary(
        symbol=latest_quote.symbol,
        midPrice=float(latest_quote.mid_price),
        reservationPrice=float(latest_quote.reservation_price),
        bidPrice=float(latest_quote.bid_price),
        askPrice=float(latest_quote.ask_price),
        spread=float(latest_quote.spread),
        inventory=float(latest_quote.inventory),
        sigmaSquared=float(latest_quote.sigma_sq),
        gamma=float(latest_quote.gamma),
        timeRemaining=float(latest_quote.t_minus_t),
        **_performative_fields(latest_quote),
    )

    # Recent trades: filled orders for this symbol (last 50)
    trades_stmt = (
        select(OrderRecord)
        .where(OrderRecord.symbol == symbol, OrderRecord.fill_price.is_not(None))
        .order_by(desc(OrderRecord.timestamp))
        .limit(50)
    )
    trades_result = await session.execute(trades_stmt)
    trades = trades_result.scalars().all()

    recent_trades = [
        TradeEntry(
            id=t.id,
            timestamp=t.timestamp.isoformat(),
            symbol=t.symbol,
            side=t.side,
            outcome=t.outcome,
            price=float(t.price),
            quantity=float(t.quantity),
            fillPrice=float(t.fill_price) if t.fill_price is not None else None,
        )
        for t in trades
    ]

    return MarketDetail(
        symbol=symbol,
        quoteHistory=quote_history,
        recentTrades=recent_trades,
        currentQuote=current_quote,
    )


_RANGE_DELTAS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}


@router.get("/pnl", response_model=list[PnlEntry])
async def get_pnl(
    range: str = Query(  # noqa: A002
        default="24h",
        description="Time range filter: 1h, 6h, 24h, 7d",
    ),
    session: AsyncSession = Depends(get_session),
) -> list[PnlEntry]:
    """Return P&L time series filtered by the requested time range."""
    delta = _RANGE_DELTAS.get(range)
    if delta is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid range '{range}'. Allowed values: {', '.join(_RANGE_DELTAS.keys())}",
        )

    cutoff = datetime.now(timezone.utc) - delta

    stmt = (
        select(PnlSnapshot)
        .where(PnlSnapshot.timestamp >= cutoff)
        .order_by(PnlSnapshot.timestamp)
    )
    result = await session.execute(stmt)
    snapshots = result.scalars().all()

    return [
        PnlEntry(
            timestamp=s.timestamp.isoformat(),
            realizedPnl=float(s.total_realized_pnl),
            unrealizedPnl=float(s.total_unrealized_pnl),
            totalExposure=float(s.total_exposure),
        )
        for s in snapshots
    ]


@router.get("/positions", response_model=list[PositionEntry])
async def get_positions(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> list[PositionEntry]:
    """Return current inventory per symbol from the latest position snapshots."""
    # Get the latest position snapshot per symbol using DISTINCT ON
    latest_subq = (
        select(
            PositionSnapshot.symbol,
            PositionSnapshot.timestamp,
        )
        .distinct(PositionSnapshot.symbol)
        .order_by(PositionSnapshot.symbol, desc(PositionSnapshot.timestamp))
        .subquery()
    )

    stmt = (
        select(PositionSnapshot)
        .join(
            latest_subq,
            (PositionSnapshot.symbol == latest_subq.c.symbol)
            & (PositionSnapshot.timestamp == latest_subq.c.timestamp),
        )
        .order_by(PositionSnapshot.symbol)
    )

    result = await session.execute(stmt)
    positions = result.scalars().all()

    bot_loop = _get_bot_loop(request)
    titles = getattr(bot_loop, "symbol_titles", {}) if bot_loop else {}

    return [
        PositionEntry(
            symbol=p.symbol,
            eventTitle=titles.get(p.symbol, ""),
            yesQuantity=float(p.yes_quantity),
            noQuantity=float(p.no_quantity),
            netInventory=float(p.net_inventory),
            unrealizedPnl=float(p.unrealized_pnl),
            timestamp=p.timestamp.isoformat(),
        )
        for p in positions
    ]


@router.post("/bot/start")
async def start_bot(request: Request) -> dict[str, str]:
    """Start the bot loop. Returns 409 if already running."""
    bot_loop = _get_bot_loop(request)

    if bot_loop is None:
        raise HTTPException(status_code=503, detail="Bot loop not initialized")

    if getattr(bot_loop, "running", False):
        raise HTTPException(status_code=409, detail="Bot is already running")

    gemini_client = getattr(request.app.state, "gemini_client", None)
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")

    try:
        # Fail fast on invalid API credentials before spawning a failing loop.
        await gemini_client.get_positions()
    except GeminiAPIError as exc:
        if exc.status_code in {401, 403} or exc.reason.lower() in {
            "invalidapikey",
            "invalid_api_key",
        }:
            raise HTTPException(
                status_code=400,
                detail="Invalid Gemini API credentials; update API key/secret before starting bot.",
            ) from exc
        raise

    try:
        await bot_loop.start()
    except Exception as exc:
        logger.exception("Failed to start bot loop")
        raise HTTPException(status_code=500, detail=f"Failed to start bot: {exc}") from exc

    return {"status": "started", "message": "Bot loop started successfully"}


@router.post("/bot/stop")
async def stop_bot(request: Request) -> dict[str, str]:
    """Stop the bot loop. Returns 409 if not running."""
    bot_loop = _get_bot_loop(request)

    if bot_loop is None:
        raise HTTPException(status_code=503, detail="Bot loop not initialized")

    if not getattr(bot_loop, "running", False):
        raise HTTPException(status_code=409, detail="Bot is not running")

    try:
        await bot_loop.stop()
    except Exception as exc:
        logger.exception("Failed to stop bot loop")
        raise HTTPException(status_code=500, detail=f"Failed to stop bot: {exc}") from exc

    return {"status": "stopped", "message": "Bot loop stopped successfully"}


@router.post("/config")
async def update_config(
    request: Request,
    body: ConfigUpdateRequest,
) -> dict[str, Any]:
    """Update runtime configuration overrides.

    Accepts a partial config object and merges non-null fields into
    the app-level config_overrides dict. Returns the new effective config.
    """
    settings = _get_settings(request)
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialized")

    # Ensure config_overrides dict exists on app.state
    if not hasattr(request.app.state, "config_overrides"):
        request.app.state.config_overrides = {}

    overrides: dict[str, Any] = request.app.state.config_overrides

    # Merge non-null values from the request body
    update_data = body.model_dump(exclude_none=True)
    overrides.update(update_data)

    # Build effective config by combining defaults with overrides
    effective: dict[str, Any] = {
        # Avellaneda-Stoikov params
        "gamma": overrides.get("gamma", settings.avellaneda_stoikov.gamma),
        "k": overrides.get("k", settings.avellaneda_stoikov.k),
        "sigma_default": overrides.get(
            "sigma_default", settings.avellaneda_stoikov.sigma_default
        ),
        "variance_window": overrides.get(
            "variance_window", settings.avellaneda_stoikov.variance_window
        ),
        # Bot params
        "bot_cycle_seconds": overrides.get(
            "bot_cycle_seconds", settings.bot.bot_cycle_seconds
        ),
        "scanner_cycle_seconds": overrides.get(
            "scanner_cycle_seconds", settings.bot.scanner_cycle_seconds
        ),
        "min_spread": overrides.get("min_spread", settings.bot.min_spread),
        # Risk params
        "max_inventory_per_symbol": overrides.get(
            "max_inventory_per_symbol", settings.risk.max_inventory_per_symbol
        ),
        "max_total_exposure": overrides.get(
            "max_total_exposure", settings.risk.max_total_exposure
        ),
        "risk_widen_threshold": overrides.get(
            "risk_widen_threshold", settings.risk.risk_widen_threshold
        ),
        # Performative params (T025)
        "quoting_mode": overrides.get(
            "quoting_mode", settings.performative.quoting_mode
        ),
        "xi_default": overrides.get(
            "xi_default", settings.performative.xi_default
        ),
        "xi_min_trades": overrides.get(
            "xi_min_trades", settings.performative.xi_min_trades
        ),
        "xi_clamp_min": overrides.get(
            "xi_clamp_min", settings.performative.xi_clamp_min
        ),
        "xi_clamp_max": overrides.get(
            "xi_clamp_max", settings.performative.xi_clamp_max
        ),
        "q_ref": overrides.get(
            "q_ref", settings.performative.q_ref
        ),
    }

    logger.info("Config overrides updated: %s", update_data)
    return {"overrides": overrides, "effective": effective}


# ---------------------------------------------------------------------------
# Theta optimization endpoints (T026, T027)
# ---------------------------------------------------------------------------


@router.post("/optimize/theta", status_code=202)
async def start_theta_optimization(
    request: Request,
    category: str | None = Query(default=None, description="Specific category to optimize"),
) -> dict[str, Any]:
    """Trigger theta optimization for all (or a specific) prediction market category.

    Returns 202 Accepted immediately; optimization runs in background.
    Returns 409 if optimization is already running.
    Returns 503 if bot loop is not initialized.
    """
    # Check bot loop availability
    bot_loop = _get_bot_loop(request)
    if bot_loop is None:
        raise HTTPException(status_code=503, detail="Bot loop not initialized")

    # Check if optimization is already running
    progress = getattr(request.app.state, "optimization_progress", None)
    if progress is None:
        raise HTTPException(status_code=503, detail="Optimization progress not initialized")
    if progress.running:
        raise HTTPException(status_code=409, detail="Optimization is already running")

    # Derive categories from bot_loop.symbol_categories
    symbol_categories: dict[str, str] = bot_loop.symbol_categories
    # Build reverse map: category -> list of symbols
    category_symbols: dict[str, list[str]] = {}
    for sym, cat in symbol_categories.items():
        if cat:
            category_symbols.setdefault(cat, []).append(sym)

    # Filter to specific category if requested
    if category is not None:
        if category in category_symbols:
            category_symbols = {category: category_symbols[category]}
        else:
            category_symbols = {category: []}

    target_categories = list(category_symbols.keys())

    # Fetch trade data for each symbol in target categories
    client = bot_loop.client
    categories_data: dict[str, list[list[float]]] = {}
    for cat, symbols in category_symbols.items():
        price_series_list: list[list[float]] = []
        for sym in symbols:
            try:
                trades = await client.get_trades(sym, limit=200)
                if trades:
                    prices = [float(t.price) for t in trades]
                    price_series_list.append(prices)
            except Exception:
                logger.warning("Failed to fetch trades for %s, skipping", sym)
        categories_data[cat] = price_series_list

    settings = _get_settings(request)

    # Launch background optimization -- store task reference on app state
    # so it isn't garbage-collected and exceptions surface properly.
    request.app.state.optimization_task = asyncio.create_task(
        run_theta_optimization(
            categories=categories_data,
            progress=progress,
            session_factory=get_session_factory(),
            settings=settings.performative,
            gamma=settings.avellaneda_stoikov.gamma,
            k=settings.avellaneda_stoikov.k,
            sigma_default=settings.avellaneda_stoikov.sigma_default,
        )
    )

    return {"status": "started", "categories": target_categories}


@router.get("/optimize/theta/status")
async def get_theta_optimization_status(request: Request) -> dict[str, Any]:
    """Get current and last theta optimization status."""
    progress = getattr(request.app.state, "optimization_progress", None)
    if progress is None:
        return {
            "running": False,
            "currentTrial": 0,
            "totalTrials": 0,
            "bestValue": None,
            "currentCategory": "",
            "categoriesCompleted": [],
            "startedAt": None,
            "completedAt": None,
            "failed": False,
            "errorMessage": "",
        }

    return {
        "running": progress.running,
        "currentTrial": progress.current_trial,
        "totalTrials": progress.total_trials,
        "bestValue": progress.best_value,
        "currentCategory": progress.current_category,
        "categoriesCompleted": list(progress.categories_completed),
        "startedAt": progress.started_at.isoformat() if progress.started_at else None,
        "completedAt": progress.completed_at.isoformat() if progress.completed_at else None,
        "failed": progress.failed,
        "errorMessage": progress.error_message,
    }
