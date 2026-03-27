"""Tests for the order book monitor."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.engine.book import OrderBookMonitor
from src.gemini.models import OrderBook, OrderBookEntry, Trade

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _entry(price: str, amount: str = "10") -> OrderBookEntry:
    return OrderBookEntry(price=Decimal(price), amount=Decimal(amount))


def _trade(price: str, ts: int = 1000) -> Trade:
    return Trade(timestamp=ts, price=Decimal(price), amount=Decimal("1"))


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestOrderBookMonitor:
    async def test_basic_market_state(self, mock_client: AsyncMock) -> None:
        mock_client.get_order_book.return_value = OrderBook(
            bids=[_entry("0.40")],
            asks=[_entry("0.60")],
        )
        mock_client.get_trades.return_value = [_trade("0.50")]

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM")

        assert state is not None
        assert state.best_bid == pytest.approx(0.40)
        assert state.best_ask == pytest.approx(0.60)
        assert state.mid_price == pytest.approx(0.50)
        assert state.spread == pytest.approx(0.20)
        assert state.trade_prices == [0.50]

    async def test_fallback_to_last_trade_when_bid_missing(
        self, mock_client: AsyncMock
    ) -> None:
        mock_client.get_order_book.return_value = OrderBook(bids=[], asks=[_entry("0.60")])
        mock_client.get_trades.return_value = [_trade("0.55")]

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM")

        assert state is not None
        assert state.best_bid == pytest.approx(0.55)  # from last trade
        assert state.best_ask == pytest.approx(0.60)

    async def test_fallback_to_last_trade_when_ask_missing(
        self, mock_client: AsyncMock
    ) -> None:
        mock_client.get_order_book.return_value = OrderBook(bids=[_entry("0.40")], asks=[])
        mock_client.get_trades.return_value = [_trade("0.45")]

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM")

        assert state is not None
        assert state.best_bid == pytest.approx(0.40)
        assert state.best_ask == pytest.approx(0.45)  # from last trade

    async def test_returns_none_when_completely_empty(
        self, mock_client: AsyncMock
    ) -> None:
        mock_client.get_order_book.return_value = OrderBook(bids=[], asks=[])
        mock_client.get_trades.return_value = []

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM")

        assert state is None

    async def test_returns_none_on_order_book_error(
        self, mock_client: AsyncMock
    ) -> None:
        mock_client.get_order_book.side_effect = RuntimeError("API down")

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM")

        assert state is None

    async def test_continues_without_trades_on_trade_error(
        self, mock_client: AsyncMock
    ) -> None:
        mock_client.get_order_book.return_value = OrderBook(
            bids=[_entry("0.40")], asks=[_entry("0.60")]
        )
        mock_client.get_trades.side_effect = RuntimeError("trades API down")

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM")

        assert state is not None
        assert state.trade_prices == []
        assert state.mid_price == pytest.approx(0.50)

    async def test_market_state_is_frozen(self, mock_client: AsyncMock) -> None:
        mock_client.get_order_book.return_value = OrderBook(
            bids=[_entry("0.40")], asks=[_entry("0.60")]
        )
        mock_client.get_trades.return_value = []

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM")

        assert state is not None
        with pytest.raises(AttributeError):
            state.mid_price = 0.99  # type: ignore[misc]

    async def test_prefetched_prices_still_fetch_trade_history(
        self, mock_client: AsyncMock
    ) -> None:
        prefetched_prices = {
            "best_bid": 0.49,
            "best_ask": 0.51,
            "last_trade_price": 0.50,
        }
        mock_client.get_trades.return_value = [
            _trade("0.48", ts=1700),
            _trade("0.49", ts=1701),
            _trade("0.50", ts=1702),
        ]

        monitor = OrderBookMonitor(mock_client)
        state = await monitor.get_market_state("SYM", prefetched_prices=prefetched_prices)

        assert state is not None
        assert state.best_bid == pytest.approx(0.49)
        assert state.best_ask == pytest.approx(0.51)
        assert state.trade_prices == [0.48, 0.49, 0.50]
        assert state.trade_timestamps == [1700.0, 1701.0, 1702.0]
        mock_client.get_trades.assert_awaited_once_with("SYM", limit=100)
