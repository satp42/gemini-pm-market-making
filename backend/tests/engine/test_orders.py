"""Tests for the order manager."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, call

import pytest

from src.engine.orders import OrderManager
from src.engine.quoting import Quote
from src.gemini.models import Order

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _quote(bid: float = 0.40, ask: float = 0.60) -> Quote:
    return Quote(
        bid_price=bid,
        ask_price=ask,
        reservation_price=0.50,
        spread=ask - bid,
        mid_price=0.50,
        inventory=0,
        sigma_sq=0.01,
        gamma=0.1,
        t_minus_t=0.5,
        k=1.5,
    )


def _order(order_id: int = 1, **kwargs) -> Order:
    defaults = dict(
        order_id=order_id,
        status="active",
        symbol="SYM",
        side="buy",
        outcome="yes",
        quantity="1",
        price=Decimal("0.50"),
    )
    defaults.update(kwargs)
    return Order(**defaults)


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


# -----------------------------------------------------------------------
# cancel_stale_orders
# -----------------------------------------------------------------------


class TestCancelStaleOrders:
    async def test_cancels_all(self, mock_client: AsyncMock) -> None:
        mock_client.cancel_order.return_value = _order()
        mgr = OrderManager(mock_client)
        count = await mgr.cancel_stale_orders("SYM", [101, 102, 103])
        assert count == 3
        assert mock_client.cancel_order.call_count == 3

    async def test_returns_zero_for_empty_list(self, mock_client: AsyncMock) -> None:
        mgr = OrderManager(mock_client)
        count = await mgr.cancel_stale_orders("SYM", [])
        assert count == 0
        mock_client.cancel_order.assert_not_called()

    async def test_partial_failure(self, mock_client: AsyncMock) -> None:
        mock_client.cancel_order.side_effect = [
            _order(101),
            RuntimeError("API error"),
            _order(103),
        ]
        mgr = OrderManager(mock_client)
        count = await mgr.cancel_stale_orders("SYM", [101, 102, 103])
        assert count == 2


# -----------------------------------------------------------------------
# place_quotes
# -----------------------------------------------------------------------


class TestPlaceQuotes:
    async def test_places_bid_and_ask(self, mock_client: AsyncMock) -> None:
        bid_resp = _order(1, side="buy")
        ask_resp = _order(2, side="sell")
        mock_client.place_order.side_effect = [bid_resp, ask_resp]

        mgr = OrderManager(mock_client)
        bid, ask = await mgr.place_quotes("SYM", _quote())

        assert bid is not None
        assert ask is not None
        assert bid.order_id == 1
        assert ask.order_id == 2
        assert mock_client.place_order.call_count == 2

    async def test_bid_failure_still_places_ask(self, mock_client: AsyncMock) -> None:
        ask_resp = _order(2, side="sell")
        mock_client.place_order.side_effect = [RuntimeError("bid fail"), ask_resp]

        mgr = OrderManager(mock_client)
        bid, ask = await mgr.place_quotes("SYM", _quote())

        assert bid is None
        assert ask is not None

    async def test_ask_failure_still_returns_bid(self, mock_client: AsyncMock) -> None:
        bid_resp = _order(1, side="buy")
        mock_client.place_order.side_effect = [bid_resp, RuntimeError("ask fail")]

        mgr = OrderManager(mock_client)
        bid, ask = await mgr.place_quotes("SYM", _quote())

        assert bid is not None
        assert ask is None

    async def test_both_fail_returns_none_none(self, mock_client: AsyncMock) -> None:
        mock_client.place_order.side_effect = RuntimeError("all fail")

        mgr = OrderManager(mock_client)
        bid, ask = await mgr.place_quotes("SYM", _quote())

        assert bid is None
        assert ask is None

    async def test_correct_order_params(self, mock_client: AsyncMock) -> None:
        mock_client.place_order.return_value = _order()
        mgr = OrderManager(mock_client)
        q = _quote(bid=0.40, ask=0.60)
        await mgr.place_quotes("MY-SYM", q, quantity=5)

        calls = mock_client.place_order.call_args_list
        # Bid call
        assert calls[0] == call(
            symbol="MY-SYM",
            side="buy",
            outcome="yes",
            price=Decimal("0.4"),
            quantity=5,
        )
        # Ask call
        assert calls[1] == call(
            symbol="MY-SYM",
            side="sell",
            outcome="yes",
            price=Decimal("0.6"),
            quantity=5,
        )
