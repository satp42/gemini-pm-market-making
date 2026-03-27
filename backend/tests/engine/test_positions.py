"""Tests for the position tracker."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.engine.book import MarketState
from src.engine.positions import PositionTracker
from src.gemini.models import Order, Position

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _pos(symbol: str, outcome: str, qty: str) -> Position:
    return Position(symbol=symbol, outcome=outcome, quantity=Decimal(qty))


def _order(order_id: int) -> Order:
    return Order(order_id=order_id, status="active", symbol="SYM")


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


# -----------------------------------------------------------------------
# get_inventory
# -----------------------------------------------------------------------


class TestGetInventory:
    async def test_yes_only(self, mock_client: AsyncMock) -> None:
        mock_client.get_positions.return_value = [_pos("SYM-A", "yes", "10")]
        tracker = PositionTracker(mock_client)
        inv = await tracker.get_inventory()
        assert inv == {"SYM-A": 10.0}

    async def test_no_only(self, mock_client: AsyncMock) -> None:
        mock_client.get_positions.return_value = [_pos("SYM-A", "no", "5")]
        tracker = PositionTracker(mock_client)
        inv = await tracker.get_inventory()
        assert inv == {"SYM-A": -5.0}

    async def test_yes_and_no_same_symbol(self, mock_client: AsyncMock) -> None:
        mock_client.get_positions.return_value = [
            _pos("SYM-A", "yes", "10"),
            _pos("SYM-A", "no", "3"),
        ]
        tracker = PositionTracker(mock_client)
        inv = await tracker.get_inventory()
        assert inv == {"SYM-A": 7.0}

    async def test_multiple_symbols(self, mock_client: AsyncMock) -> None:
        mock_client.get_positions.return_value = [
            _pos("SYM-A", "yes", "10"),
            _pos("SYM-B", "no", "5"),
        ]
        tracker = PositionTracker(mock_client)
        inv = await tracker.get_inventory()
        assert inv == {"SYM-A": 10.0, "SYM-B": -5.0}

    async def test_empty_positions(self, mock_client: AsyncMock) -> None:
        mock_client.get_positions.return_value = []
        tracker = PositionTracker(mock_client)
        inv = await tracker.get_inventory()
        assert inv == {}


# -----------------------------------------------------------------------
# get_active_order_ids
# -----------------------------------------------------------------------


class TestGetActiveOrderIds:
    async def test_returns_order_ids(self, mock_client: AsyncMock) -> None:
        mock_client.get_active_orders.return_value = [_order(101), _order(102)]
        tracker = PositionTracker(mock_client)
        ids = await tracker.get_active_order_ids("SYM")
        assert ids == [101, 102]

    async def test_returns_empty_for_no_orders(self, mock_client: AsyncMock) -> None:
        mock_client.get_active_orders.return_value = []
        tracker = PositionTracker(mock_client)
        ids = await tracker.get_active_order_ids("SYM")
        assert ids == []

    async def test_passes_symbol_to_client(self, mock_client: AsyncMock) -> None:
        mock_client.get_active_orders.return_value = []
        tracker = PositionTracker(mock_client)
        await tracker.get_active_order_ids("MY-SYM")
        mock_client.get_active_orders.assert_called_once_with(symbol="MY-SYM")

    async def test_passes_none_symbol(self, mock_client: AsyncMock) -> None:
        mock_client.get_active_orders.return_value = []
        tracker = PositionTracker(mock_client)
        await tracker.get_active_order_ids(None)
        mock_client.get_active_orders.assert_called_once_with(symbol=None)


# -----------------------------------------------------------------------
# compute_unrealized_pnl
# -----------------------------------------------------------------------


class TestComputeUnrealizedPnl:
    def test_basic_pnl(self) -> None:
        positions = {"SYM-A": 10.0}
        states = {"SYM-A": MarketState(mid_price=0.60, best_bid=0.55, best_ask=0.65, spread=0.10)}
        pnl = PositionTracker.compute_unrealized_pnl(positions, states)
        assert pnl == {"SYM-A": pytest.approx(6.0)}

    def test_negative_inventory(self) -> None:
        positions = {"SYM-A": -5.0}
        states = {"SYM-A": MarketState(mid_price=0.40, best_bid=0.35, best_ask=0.45, spread=0.10)}
        pnl = PositionTracker.compute_unrealized_pnl(positions, states)
        assert pnl == {"SYM-A": pytest.approx(-2.0)}

    def test_missing_market_state_returns_zero(self) -> None:
        positions = {"SYM-A": 10.0}
        pnl = PositionTracker.compute_unrealized_pnl(positions, {})
        assert pnl == {"SYM-A": 0.0}

    def test_empty_positions(self) -> None:
        pnl = PositionTracker.compute_unrealized_pnl({}, {})
        assert pnl == {}
