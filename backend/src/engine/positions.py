"""Position tracker -- inventory management and P&L computation."""

from __future__ import annotations

import logging

from src.engine.book import MarketState
from src.gemini.client import GeminiClient

logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks inventory per symbol and computes unrealised P&L."""

    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    async def get_inventory(self) -> dict[str, float]:
        """Return net inventory per symbol: ``yes_qty - no_qty``.

        Positive values mean long Yes, negative means long No.
        """
        positions = await self._client.get_positions()
        inventory: dict[str, float] = {}

        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.quantity)
            if pos.outcome.lower() == "yes":
                inventory[symbol] = inventory.get(symbol, 0.0) + qty
            elif pos.outcome.lower() == "no":
                inventory[symbol] = inventory.get(symbol, 0.0) - qty
            else:
                logger.warning(
                    "Unknown outcome '%s' for position on %s", pos.outcome, symbol
                )

        return inventory

    async def get_active_order_ids(self, symbol: str | None = None) -> list[int]:
        """Return Gemini order IDs for all active orders, optionally filtered by symbol."""
        orders = await self._client.get_active_orders(symbol=symbol)
        return [o.order_id for o in orders]

    @staticmethod
    def compute_unrealized_pnl(
        positions: dict[str, float],
        market_states: dict[str, MarketState],
    ) -> dict[str, float]:
        """Mark each position to mid-price and return per-symbol unrealised P&L.

        For a prediction market contract valued 0-1:
        - Long Yes inventory at mid p has value: qty * p
        - Long No  inventory at mid p has value: |qty| * (1 - p)

        We approximate unrealised PnL as ``inventory * mid_price`` for net
        inventory.  (Actual PnL would require avg entry price; this is a
        mark-to-market proxy.)
        """
        pnl: dict[str, float] = {}
        for symbol, inv in positions.items():
            state = market_states.get(symbol)
            if state is None:
                pnl[symbol] = 0.0
                continue
            pnl[symbol] = inv * state.mid_price
        return pnl
