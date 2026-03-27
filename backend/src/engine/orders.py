"""Order manager -- translates quotes into Gemini API calls."""

from __future__ import annotations

import logging
from decimal import Decimal

from src.engine.quoting import Quote
from src.gemini.client import GeminiClient
from src.gemini.models import Order

logger = logging.getLogger(__name__)


class OrderManager:
    """Places and cancels orders on the Gemini prediction market."""

    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    async def cancel_stale_orders(
        self, symbol: str, active_order_ids: list[int]
    ) -> int:
        """Cancel all orders in *active_order_ids*. Returns the count successfully cancelled."""
        cancelled = 0
        for order_id in active_order_ids:
            try:
                await self._client.cancel_order(order_id)
                cancelled += 1
                logger.debug("Cancelled order %d on %s", order_id, symbol)
            except Exception:
                logger.warning(
                    "Failed to cancel order %d on %s", order_id, symbol, exc_info=True
                )
        if cancelled:
            logger.info(
                "Cancelled %d/%d orders on %s",
                cancelled,
                len(active_order_ids),
                symbol,
            )
        return cancelled

    async def place_quotes(
        self,
        symbol: str,
        quote: Quote,
        quantity: int = 1,
    ) -> tuple[Order | None, Order | None]:
        """Place a bid and an ask order for *symbol* based on *quote*.

        Returns ``(bid_order, ask_order)``.  Either may be ``None`` if the
        placement failed (errors are logged, not raised).
        """
        bid_order: Order | None = None
        ask_order: Order | None = None

        # Place bid (buy Yes at bid price)
        try:
            bid_order = await self._client.place_order(
                symbol=symbol,
                side="buy",
                outcome="yes",
                price=Decimal(str(round(quote.bid_price, 2))),
                quantity=quantity,
            )
            logger.info(
                "Placed bid on %s: price=%.4f qty=%d orderId=%d",
                symbol,
                quote.bid_price,
                quantity,
                bid_order.order_id,
            )
        except Exception:
            logger.warning(
                "Failed to place bid on %s at %.4f",
                symbol,
                quote.bid_price,
                exc_info=True,
            )

        # Place ask (sell Yes at ask price)
        try:
            ask_order = await self._client.place_order(
                symbol=symbol,
                side="sell",
                outcome="yes",
                price=Decimal(str(round(quote.ask_price, 2))),
                quantity=quantity,
            )
            logger.info(
                "Placed ask on %s: price=%.4f qty=%d orderId=%d",
                symbol,
                quote.ask_price,
                quantity,
                ask_order.order_id,
            )
        except Exception:
            logger.warning(
                "Failed to place ask on %s at %.4f",
                symbol,
                quote.ask_price,
                exc_info=True,
            )

        return bid_order, ask_order
