"""Tests for the market scanner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.config import BotSettings
from src.engine.scanner import MarketScanner
from src.gemini.models import Contract, Event, EventsResponse

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_contract(
    symbol: str = "SYM-A",
    best_bid: Decimal | None = Decimal("0.40"),
    best_ask: Decimal | None = Decimal("0.60"),
) -> Contract:
    return Contract(
        instrument_symbol=symbol,
        outcome="yes",
        best_bid=best_bid,
        best_ask=best_ask,
    )


def _make_event(
    ticker: str = "EVT-1",
    contracts: list[Contract] | None = None,
    hours_from_now: float = 48.0,
) -> Event:
    expiry = (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()
    return Event(
        event_ticker=ticker,
        title="Test event",
        status="active",
        expiration_date=expiry,
        contracts=contracts or [_make_contract()],
    )


@pytest.fixture
def bot_settings() -> BotSettings:
    return BotSettings(
        BOT_CYCLE_SECONDS=10,
        SCANNER_CYCLE_SECONDS=300,
        MIN_SPREAD=0.03,
        MIN_TIME_TO_EXPIRY_HOURS=1,
        EXCLUDED_SYMBOLS=[],
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestMarketScanner:
    async def test_scan_returns_eligible_symbols(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        mock_client.get_events.return_value = EventsResponse(
            events=[_make_event(contracts=[_make_contract("SYM-A")])],
            pagination={},
        )
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan()
        assert result == ["SYM-A"]

    async def test_scan_rejects_narrow_spread(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        contract = _make_contract("SYM-B", best_bid=Decimal("0.50"), best_ask=Decimal("0.51"))
        mock_client.get_events.return_value = EventsResponse(
            events=[_make_event(contracts=[contract])],
            pagination={},
        )
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan()
        assert result == []

    async def test_scan_rejects_missing_bid(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        contract = _make_contract("SYM-C", best_bid=None, best_ask=Decimal("0.60"))
        mock_client.get_events.return_value = EventsResponse(
            events=[_make_event(contracts=[contract])],
            pagination={},
        )
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan()
        assert result == []

    async def test_scan_rejects_missing_ask(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        contract = _make_contract("SYM-D", best_bid=Decimal("0.40"), best_ask=None)
        mock_client.get_events.return_value = EventsResponse(
            events=[_make_event(contracts=[contract])],
            pagination={},
        )
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan()
        assert result == []

    async def test_scan_rejects_excluded_symbol(
        self, mock_client: AsyncMock,
    ) -> None:
        settings = BotSettings(
            BOT_CYCLE_SECONDS=10,
            SCANNER_CYCLE_SECONDS=300,
            MIN_SPREAD=0.03,
            MIN_TIME_TO_EXPIRY_HOURS=1,
            EXCLUDED_SYMBOLS=["SYM-EXCLUDED"],
        )
        mock_client.get_events.return_value = EventsResponse(
            events=[_make_event(contracts=[_make_contract("SYM-EXCLUDED")])],
            pagination={},
        )
        scanner = MarketScanner(mock_client, settings)
        result = await scanner.scan()
        assert result == []

    async def test_scan_rejects_expiring_soon(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        mock_client.get_events.return_value = EventsResponse(
            events=[_make_event(hours_from_now=0.5)],  # 30 min, below 1h threshold
            pagination={},
        )
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan()
        assert result == []

    async def test_scan_newly_listed(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        mock_client.get_newly_listed.return_value = [
            _make_event(contracts=[_make_contract("NEW-1")]),
        ]
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan_newly_listed()
        assert result == ["NEW-1"]

    async def test_scan_handles_no_expiration_date(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        event = Event(
            event_ticker="NO-EXPIRY",
            title="No expiry",
            status="active",
            expiration_date=None,
            contracts=[_make_contract("SYM-NOEXP")],
        )
        mock_client.get_events.return_value = EventsResponse(events=[event], pagination={})
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan()
        # No expiry means we cannot reject on time grounds, so it passes
        assert result == ["SYM-NOEXP"]

    async def test_scan_multiple_contracts_in_event(
        self, mock_client: AsyncMock, bot_settings: BotSettings
    ) -> None:
        contracts = [
            _make_contract("SYM-X"),
            # Too narrow spread -- should be rejected
            _make_contract("SYM-Y", Decimal("0.50"), Decimal("0.51")),
        ]
        mock_client.get_events.return_value = EventsResponse(
            events=[_make_event(contracts=contracts)],
            pagination={},
        )
        scanner = MarketScanner(mock_client, bot_settings)
        result = await scanner.scan()
        assert result == ["SYM-X"]
