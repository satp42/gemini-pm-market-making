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


class OrderBookMonitor:
    """Fetches order book and recent trades to build a MarketState."""

    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    async def get_market_state(self, symbol: str) -> MarketState | None:
        """Build a MarketState for *symbol*.

        Returns ``None`` when the book is completely empty and there are no
        recent trades to fall back on.
        """
        try:
            order_book = await self._client.get_order_book(symbol)
        except Exception:
            logger.exception("Failed to fetch order book for %s", symbol)
            return None

        best_bid: float | None = None
        best_ask: float | None = None

        if order_book.bids:
            best_bid = float(order_book.bids[0].price)
        if order_book.asks:
            best_ask = float(order_book.asks[0].price)

        # Fetch recent trades for variance estimation and fallback pricing
        trade_prices: list[float] = []
        try:
            trades = await self._client.get_trades(symbol, limit=100)
            trade_prices = [float(t.price) for t in trades]
        except Exception:
            logger.warning("Failed to fetch trades for %s, continuing without", symbol)

        # Fall back to last trade price when one or both sides of the book are empty
        last_trade = trade_prices[-1] if trade_prices else None

        if best_bid is None and last_trade is not None:
            best_bid = last_trade
        if best_ask is None and last_trade is not None:
            best_ask = last_trade

        # If still nothing, we cannot construct a state
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
        )
