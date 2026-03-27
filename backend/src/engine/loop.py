"""Main bot cycle orchestrator -- the heart of the market-making engine."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings
from src.db.models import OrderRecord, PnlSnapshot, PositionSnapshot, ThetaParameter, XiEstimateRecord
from src.db.models import Quote as QuoteRecord
from src.engine.book import OrderBookMonitor
from src.engine.orders import OrderManager
from src.engine.performative import compute_performative_quote
from src.engine.positions import PositionTracker
from src.engine.quoting import Quote, compute_quote, estimate_variance
from src.engine.risk import RiskAction, RiskManager
from src.engine.scanner import MarketScanner
from src.engine.xi import XiEstimate, estimate_xi
from src.gemini.client import GeminiClient

logger = logging.getLogger(__name__)


class BotLoop:
    """Composes all engine components and runs the main market-making loop.

    Launched as an ``asyncio`` background task during application startup.
    """

    def __init__(
        self,
        settings: Settings,
        client: GeminiClient,
        session_factory: async_sessionmaker[AsyncSession],
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._session_factory = session_factory
        self._config_overrides: dict[str, Any] = (
            config_overrides if config_overrides is not None else {}
        )

        # Engine components
        self._scanner = MarketScanner(client, settings.bot)
        self._book = OrderBookMonitor(client)
        self._positions = PositionTracker(client)
        self._orders = OrderManager(client)
        self._risk = RiskManager(settings.risk)

        # State
        self._running = False
        self._started_at: datetime | None = None
        self._active_symbols: list[str] = []
        self._symbol_titles: dict[str, str] = {}
        self._symbol_prices: dict[str, dict] = {}
        self._symbol_categories: dict[str, str] = {}
        self._theta_cache: dict[str, tuple[float, float, float]] = {}
        self._task: asyncio.Task[None] | None = None
        self._last_scan_time: float = 0.0

        # WebSocket broadcast callback -- set by the API layer
        self.on_tick: Callable[[dict[str, Any]], Any] | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def active_symbols(self) -> list[str]:
        return list(self._active_symbols)

    @property
    def symbol_titles(self) -> dict[str, str]:
        return dict(self._symbol_titles)

    @property
    def client(self) -> GeminiClient:
        return self._client

    @property
    def symbol_categories(self) -> dict[str, str]:
        return dict(self._symbol_categories)

    @property
    def risk_manager(self) -> RiskManager:
        """Expose risk manager so the API can toggle the kill switch."""
        return self._risk

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the main loop as a background task."""
        if self._running:
            logger.warning("Bot loop is already running")
            return
        self._running = True
        self._started_at = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._run())
        logger.info("Bot loop started")

    async def stop(self) -> None:
        """Stop the loop and cancel all outstanding orders."""
        self._running = False
        logger.info("Bot loop stopping -- cancelling all orders")

        # Cancel all outstanding orders across all active symbols
        for symbol in self._active_symbols:
            try:
                order_ids = await self._positions.get_active_order_ids(symbol)
                if order_ids:
                    await self._orders.cancel_stale_orders(symbol, order_ids)
            except Exception:
                logger.exception(
                    "Error cancelling orders for %s during shutdown", symbol
                )

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._started_at = None
        logger.info("Bot loop stopped")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current bot state for the API."""
        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "active_symbols": list(self._active_symbols),
            "num_active_symbols": len(self._active_symbols),
            "kill_switch": self._risk.should_kill_all(),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main async loop -- runs until ``stop()`` is called."""
        logger.info("Bot loop _run() entered")

        while self._running:
            cycle_start = time.monotonic()

            try:
                # 1. Scanner: refresh active symbols on its own cadence
                await self._maybe_scan()

                # 2. If kill switch is active, cancel everything and wait
                if self._risk.should_kill_all():
                    await self._cancel_all_symbols()
                    await asyncio.sleep(self._settings.bot.bot_cycle_seconds)
                    continue

                # 3. Gather inventory for all symbols at once
                inventories = await self._positions.get_inventory()

                # 4. Check total exposure
                if not self._risk.check_total_exposure(inventories):
                    logger.warning("Total exposure exceeded -- skipping this cycle")
                    await asyncio.sleep(self._settings.bot.bot_cycle_seconds)
                    continue

                # 5. Process each symbol independently
                tick_data: dict[str, Any] = {
                    "symbols": {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                for symbol in self._active_symbols:
                    try:
                        symbol_data = await self._process_symbol(symbol, inventories)
                        if symbol_data:
                            tick_data["symbols"][symbol] = symbol_data
                    except Exception:
                        logger.exception(
                            "Error processing symbol %s -- continuing", symbol
                        )

                # 6. Persist aggregated snapshot
                await self._persist_tick(tick_data, inventories)

                # 7. Broadcast via callback (format for frontend)
                if self.on_tick is not None:
                    try:
                        uptime = (
                            time.time() - self._started_at.timestamp()
                            if self._started_at
                            else 0.0
                        )
                        broadcast_msg = {
                            "type": "tick",
                            "data": {
                                "status": {
                                    "running": self._running,
                                    "uptime": uptime,
                                    "activeMarkets": len(self._active_symbols),
                                    "environment": self._settings.gemini.env,
                                },
                                "markets": [
                                    sd.get("quote_summary", {})
                                    for sd in tick_data.get("symbols", {}).values()
                                    if "quote_summary" in sd
                                ],
                                "positions": [],
                                "pnl": {},
                            },
                        }
                        result = self.on_tick(broadcast_msg)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.warning("on_tick callback failed", exc_info=True)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unhandled error in bot loop cycle")

            # Sleep remaining time in the cycle
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, self._settings.bot.bot_cycle_seconds - elapsed)
            await asyncio.sleep(sleep_time)

        logger.info("Bot loop _run() exited")

    # ------------------------------------------------------------------
    # Symbol processing
    # ------------------------------------------------------------------

    async def _process_symbol(
        self,
        symbol: str,
        inventories: dict[str, float],
    ) -> dict[str, Any] | None:
        """Run one full cycle for a single symbol. Returns tick data dict or None."""
        # a. Market state (use prefetched prices from the events API)
        prefetched = self._symbol_prices.get(symbol)
        state = await self._book.get_market_state(symbol, prefetched_prices=prefetched)
        if state is None:
            logger.debug("No market state for %s -- skipping", symbol)
            return None

        # b. Inventory
        inventory = inventories.get(symbol, 0.0)

        # c. Risk checks
        symbol_risk = self._risk.check_symbol(symbol, inventory)

        # d. Time-to-expiry (normalised as fraction of a day: 24h = 1.0)
        #    Default to 0.5 (mid-life); the scanner already filters out
        #    contracts that are too close to expiry.
        t_minus_t = 0.5

        time_risk = self._risk.check_time_safety(t_minus_t)

        # Take the most restrictive risk action
        risk_action = self._worst_risk(symbol_risk, time_risk)

        # e. Act on risk decision
        active_ids = await self._positions.get_active_order_ids(symbol)

        if risk_action == RiskAction.STOP_QUOTING:
            if active_ids:
                await self._orders.cancel_stale_orders(symbol, active_ids)
            return {
                "action": "stop",
                "inventory": inventory,
                "mid_price": state.mid_price,
            }

        # f. Compute quote
        sigma_sq = estimate_variance(
            state.trade_prices,
            default=self._settings.avellaneda_stoikov.sigma_default,
        )
        gamma = self._settings.avellaneda_stoikov.gamma
        k = self._settings.avellaneda_stoikov.k

        quoting_mode = self._config_overrides.get(
            "quoting_mode", self._settings.performative.quoting_mode
        )

        if quoting_mode == "as":
            quote = compute_quote(
                mid_price=state.mid_price,
                inventory=inventory,
                gamma=gamma,
                sigma_sq=sigma_sq,
                t_minus_t=t_minus_t,
                k=k,
            )
        elif quoting_mode in ("performative", "theta"):
            # (a) Estimate xi from trade data
            perf = self._settings.performative
            xi_est = estimate_xi(
                trade_prices=state.trade_prices,
                trade_timestamps=state.trade_timestamps,
                xi_default=perf.xi_default,
                xi_min_trades=perf.xi_min_trades,
                xi_clamp_min=perf.xi_clamp_min,
                xi_clamp_max=perf.xi_clamp_max,
                r_squared_threshold=perf.r_squared_threshold,
            )

            # (b) Lookup theta from cache
            category = self._symbol_categories.get(symbol, "")
            theta_tuple = self._theta_cache.get(category, None)

            # (c) If mode=="theta" and no theta found, fall back to performative
            if theta_tuple is not None:
                theta0, theta1, theta2 = theta_tuple
            else:
                if quoting_mode == "theta":
                    logger.warning(
                        "No theta parameters for category %r (symbol %s), "
                        "falling back to performative defaults (1,1,1)",
                        category,
                        symbol,
                    )
                theta0, theta1, theta2 = 1.0, 1.0, 1.0

            # (d) Compute performative quote
            quote = compute_performative_quote(
                mid_price=state.mid_price,
                inventory=inventory,
                gamma=gamma,
                sigma_sq=sigma_sq,
                t_minus_t=t_minus_t,
                k=k,
                xi=xi_est.xi,
                theta0=theta0,
                theta1=theta1,
                theta2=theta2,
                q_ref=perf.q_ref,
            )
            # Tag with actual mode used
            quote.quoting_mode = quoting_mode

            # Persist xi estimate (fire-and-forget)
            asyncio.create_task(self._persist_xi_estimate(symbol, xi_est))
        else:
            # Unknown mode -- default to A&S with warning
            logger.warning("Unknown quoting_mode %r, defaulting to A&S", quoting_mode)
            quote = compute_quote(
                mid_price=state.mid_price,
                inventory=inventory,
                gamma=gamma,
                sigma_sq=sigma_sq,
                t_minus_t=t_minus_t,
                k=k,
            )

        # Widen spread if risk says so
        if risk_action == RiskAction.WIDEN_SPREAD:
            widened_spread = quote.spread * 2.0
            bid = quote.reservation_price - widened_spread / 2.0
            ask = quote.reservation_price + widened_spread / 2.0
            bid = max(0.01, min(0.99, bid))
            ask = max(0.01, min(0.99, ask))
            quote = Quote(
                bid_price=bid,
                ask_price=ask,
                reservation_price=quote.reservation_price,
                spread=widened_spread,
                mid_price=quote.mid_price,
                inventory=quote.inventory,
                sigma_sq=quote.sigma_sq,
                gamma=quote.gamma,
                t_minus_t=quote.t_minus_t,
                k=quote.k,
                xi=quote.xi,
                theta0=quote.theta0,
                theta1=quote.theta1,
                theta2=quote.theta2,
                quoting_mode=quote.quoting_mode,
            )

        # g. Cancel stale, place new
        if active_ids:
            await self._orders.cancel_stale_orders(symbol, active_ids)

        bid_order, ask_order = await self._orders.place_quotes(symbol, quote, inventory=inventory)

        # h. Persist quote to DB
        await self._persist_quote(
            symbol, quote, bid_order, ask_order, inventory, state.mid_price
        )

        return {
            "action": risk_action.value,
            "quote": {
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "reservation": quote.reservation_price,
                "spread": quote.spread,
            },
            "inventory": inventory,
            "mid_price": state.mid_price,
            "sigma_sq": sigma_sq,
            "xi": quote.xi,
            "theta0": quote.theta0,
            "theta1": quote.theta1,
            "theta2": quote.theta2,
            "quoting_mode": quote.quoting_mode,
            "quote_summary": {
                "symbol": symbol,
                "title": self._symbol_titles.get(symbol, symbol),
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "reservation": quote.reservation_price,
                "spread": quote.spread,
                "midPrice": state.mid_price,
                "inventory": inventory,
                "xi": quote.xi,
                "theta0": quote.theta0,
                "theta1": quote.theta1,
                "theta2": quote.theta2,
                "quotingMode": quote.quoting_mode,
            },
        }

    # ------------------------------------------------------------------
    # Scanner
    # ------------------------------------------------------------------

    async def _maybe_scan(self) -> None:
        """Run the scanner if enough time has passed since the last scan."""
        now = time.monotonic()
        if now - self._last_scan_time >= self._settings.bot.scanner_cycle_seconds:
            try:
                symbols, titles, prices, categories = await self._scanner.scan()
                self._active_symbols = symbols
                self._symbol_titles.update(titles)
                self._symbol_prices = prices
                self._symbol_categories = categories
                self._theta_cache = await self._load_theta_cache()
                self._last_scan_time = now
            except Exception:
                logger.exception("Scanner failed -- keeping previous symbol list")

    # ------------------------------------------------------------------
    # Risk helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _worst_risk(a: RiskAction, b: RiskAction) -> RiskAction:
        """Return the most restrictive of two risk actions."""
        order = {
            RiskAction.QUOTE_NORMAL: 0,
            RiskAction.WIDEN_SPREAD: 1,
            RiskAction.STOP_QUOTING: 2,
        }
        return a if order[a] >= order[b] else b

    async def _cancel_all_symbols(self) -> None:
        """Cancel all orders across all active symbols (kill-switch path)."""
        for symbol in self._active_symbols:
            try:
                ids = await self._positions.get_active_order_ids(symbol)
                if ids:
                    await self._orders.cancel_stale_orders(symbol, ids)
            except Exception:
                logger.exception(
                    "Error cancelling orders for %s during kill-all", symbol
                )

    # ------------------------------------------------------------------
    # Theta cache
    # ------------------------------------------------------------------

    async def _load_theta_cache(self) -> dict[str, tuple[float, float, float]]:
        """Query ThetaParameter table and return {category: (theta0, theta1, theta2)}."""
        cache: dict[str, tuple[float, float, float]] = {}
        try:
            async with self._session_factory() as session:
                result = await session.execute(select(ThetaParameter))
                for row in result.scalars().all():
                    cache[row.category] = (
                        float(row.theta0),
                        float(row.theta1),
                        float(row.theta2),
                    )
        except Exception:
            logger.exception("Failed to load theta cache from DB")
        return cache

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_xi_estimate(self, symbol: str, xi_est: XiEstimate) -> None:
        """Fire-and-forget: write XiEstimateRecord to the database."""
        try:
            async with self._session_factory() as session:
                session.add(
                    XiEstimateRecord(
                        symbol=symbol,
                        xi=xi_est.xi,
                        r_squared=xi_est.r_squared,
                        num_trades=xi_est.num_trades,
                        used_default=xi_est.used_default,
                    )
                )
                await session.commit()
        except Exception:
            logger.exception("Failed to persist xi estimate for %s", symbol)

    async def _persist_quote(
        self,
        symbol: str,
        quote: Quote,
        bid_order: Any,
        ask_order: Any,
        inventory: float,
        mid_price: float,
    ) -> None:
        """Write quote and order records to the database."""
        try:
            async with self._session_factory() as session:
                session.add(
                    QuoteRecord(
                        symbol=symbol,
                        mid_price=mid_price,
                        reservation_price=quote.reservation_price,
                        bid_price=quote.bid_price,
                        ask_price=quote.ask_price,
                        spread=quote.spread,
                        inventory=inventory,
                        sigma_sq=quote.sigma_sq,
                        gamma=quote.gamma,
                        t_minus_t=quote.t_minus_t,
                        xi=quote.xi,
                        theta0=quote.theta0,
                        theta1=quote.theta1,
                        theta2=quote.theta2,
                        quoting_mode=quote.quoting_mode,
                    )
                )

                for order, side in [(bid_order, "buy"), (ask_order, "sell")]:
                    if order is not None:
                        session.add(
                            OrderRecord(
                                symbol=symbol,
                                gemini_order_id=order.order_id,
                                side=side,
                                outcome=order.outcome or "yes",
                                price=float(order.price),
                                quantity=float(order.quantity),
                                status=order.status,
                            )
                        )

                await session.commit()
        except Exception:
            logger.exception("Failed to persist quote data for %s", symbol)

    async def _persist_tick(
        self,
        tick_data: dict[str, Any],
        inventories: dict[str, float],
    ) -> None:
        """Write position and PnL snapshots to the database."""
        try:
            async with self._session_factory() as session:
                # Position snapshots
                all_positions = await self._client.get_positions()
                per_symbol: dict[str, dict[str, float]] = {}
                for pos in all_positions:
                    s = pos.symbol
                    if s not in per_symbol:
                        per_symbol[s] = {"yes": 0.0, "no": 0.0}
                    if pos.outcome.lower() == "yes":
                        per_symbol[s]["yes"] += float(pos.quantity)
                    else:
                        per_symbol[s]["no"] += float(pos.quantity)

                total_unrealized = 0.0
                for s, qtys in per_symbol.items():
                    net = qtys["yes"] - qtys["no"]
                    sym_data = tick_data.get("symbols", {}).get(s, {})
                    mid = sym_data.get("mid_price", 0.0)
                    upnl = net * mid
                    total_unrealized += upnl

                    session.add(
                        PositionSnapshot(
                            symbol=s,
                            yes_quantity=qtys["yes"],
                            no_quantity=qtys["no"],
                            net_inventory=net,
                            unrealized_pnl=upnl,
                        )
                    )

                # PnL snapshot
                total_exposure = sum(abs(v) for v in inventories.values())
                session.add(
                    PnlSnapshot(
                        total_realized_pnl=0.0,
                        total_unrealized_pnl=total_unrealized,
                        total_exposure=total_exposure,
                        num_active_markets=len(self._active_symbols),
                    )
                )

                await session.commit()
        except Exception:
            logger.exception("Failed to persist tick data")
