"""Order book monitor -- reads and processes order book data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.gemini.client import GeminiClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketState:
    """Snapshot of the current market for a single symbol."""

    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    trade_prices: list[float] = field(default_factory=list)
    trade_timestamps: list[float] = field(default_factory=list)


class OrderBookMonitor:
    """Fetches order book and recent trades to build a MarketState."""

    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    async def get_market_state(
        self,
        symbol: str,
        prefetched_prices: dict | None = None,
    ) -> MarketState | None:
        """Build a MarketState for *symbol*.

        When *prefetched_prices* is provided (keys: best_bid, best_ask,
        last_trade_price), those values are used directly instead of
        calling the ``/v1/book/`` endpoint. This is required for
        prediction-market instruments which are not available on the
        standard order-book endpoint.

        Returns ``None`` when pricing data is unavailable.
        """
        best_bid: float | None = None
        best_ask: float | None = None
        trade_prices: list[float] = []
        trade_timestamps: list[float] = []

        if prefetched_prices is not None:
            best_bid = prefetched_prices.get("best_bid")
            best_ask = prefetched_prices.get("best_ask")
            ltp = prefetched_prices.get("last_trade_price")
            if ltp is not None:
                trade_prices = [ltp]
            if best_bid is None and ltp is not None:
                best_bid = ltp
            if best_ask is None and ltp is not None:
                best_ask = ltp
        else:
            try:
                order_book = await self._client.get_order_book(symbol)
            except Exception:
                logger.exception("Failed to fetch order book for %s", symbol)
                return None

            if order_book.bids:
                best_bid = float(order_book.bids[0].price)
            if order_book.asks:
                best_ask = float(order_book.asks[0].price)

            try:
                trades = await self._client.get_trades(symbol, limit=100)
                trade_prices = [float(t.price) for t in trades]
                trade_timestamps = [float(t.timestamp) for t in trades]
            except Exception:
                logger.warning("Failed to fetch trades for %s, continuing without", symbol)

            last_trade = trade_prices[-1] if trade_prices else None
            if best_bid is None and last_trade is not None:
                best_bid = last_trade
            if best_ask is None and last_trade is not None:
                best_ask = last_trade

        if best_bid is None or best_ask is None:
            logger.warning(
                "No pricing data for %s -- both sides empty and no trades", symbol
            )
            return None

        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        return MarketState(
            mid_price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            trade_prices=trade_prices,
            trade_timestamps=trade_timestamps,
        )
