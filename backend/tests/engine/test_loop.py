"""Tests for the bot loop orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import ASSettings, BotSettings, RiskSettings, Settings
from src.engine.loop import BotLoop
from src.engine.risk import RiskAction
from src.gemini.models import EventsResponse

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_settings() -> MagicMock:
    """Create a mock Settings object with fast cycle times for testing."""
    settings = MagicMock(spec=Settings)

    # BotSettings fields use aliases for init
    bot = MagicMock(spec=BotSettings)
    bot.bot_cycle_seconds = 1
    bot.scanner_cycle_seconds = 1
    bot.min_spread = 0.03
    bot.min_time_to_expiry_hours = 1
    bot.excluded_symbols = []
    settings.bot = bot

    # RiskSettings fields use aliases for init
    risk = MagicMock(spec=RiskSettings)
    risk.max_inventory_per_symbol = 200
    risk.max_total_exposure = 1000
    risk.risk_widen_threshold = 0.8
    settings.risk = risk

    # ASSettings uses env_prefix, not aliases
    a_s = MagicMock(spec=ASSettings)
    a_s.gamma = 0.1
    a_s.k = 1.5
    a_s.sigma_default = 0.01
    a_s.variance_window = 100
    settings.avellaneda_stoikov = a_s

    return settings


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    # Default: scanner returns no events
    client.get_events.return_value = EventsResponse(events=[], pagination={})
    client.get_positions.return_value = []
    client.get_active_orders.return_value = []
    return client


@pytest.fixture
def mock_session_factory() -> AsyncMock:
    """Create a mock async session factory."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session
    return factory


@pytest.fixture
def bot_loop(
    mock_client: AsyncMock, mock_session_factory: AsyncMock
) -> BotLoop:
    return BotLoop(_make_settings(), mock_client, mock_session_factory)


# -----------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------


class TestBotLoopLifecycle:
    def test_initial_state(self, bot_loop: BotLoop) -> None:
        assert bot_loop.running is False
        assert bot_loop.started_at is None
        assert bot_loop.active_symbols == []

    async def test_start_sets_running(self, bot_loop: BotLoop) -> None:
        await bot_loop.start()
        assert bot_loop.running is True
        assert bot_loop.started_at is not None
        # Clean up
        await bot_loop.stop()

    async def test_stop_clears_state(self, bot_loop: BotLoop) -> None:
        await bot_loop.start()
        await bot_loop.stop()
        assert bot_loop.running is False
        assert bot_loop.started_at is None

    async def test_double_start_is_safe(self, bot_loop: BotLoop) -> None:
        await bot_loop.start()
        await bot_loop.start()  # Should not raise
        assert bot_loop.running is True
        await bot_loop.stop()


# -----------------------------------------------------------------------
# get_status
# -----------------------------------------------------------------------


class TestGetStatus:
    def test_status_when_stopped(self, bot_loop: BotLoop) -> None:
        status = bot_loop.get_status()
        assert status["running"] is False
        assert status["started_at"] is None
        assert status["active_symbols"] == []
        assert status["kill_switch"] is False

    async def test_status_when_running(self, bot_loop: BotLoop) -> None:
        await bot_loop.start()
        status = bot_loop.get_status()
        assert status["running"] is True
        assert status["started_at"] is not None
        await bot_loop.stop()


# -----------------------------------------------------------------------
# _worst_risk
# -----------------------------------------------------------------------


class TestWorstRisk:
    def test_normal_vs_normal(self) -> None:
        result = BotLoop._worst_risk(
            RiskAction.QUOTE_NORMAL, RiskAction.QUOTE_NORMAL
        )
        assert result == RiskAction.QUOTE_NORMAL

    def test_normal_vs_widen(self) -> None:
        result = BotLoop._worst_risk(
            RiskAction.QUOTE_NORMAL, RiskAction.WIDEN_SPREAD
        )
        assert result == RiskAction.WIDEN_SPREAD

    def test_widen_vs_stop(self) -> None:
        result = BotLoop._worst_risk(
            RiskAction.WIDEN_SPREAD, RiskAction.STOP_QUOTING
        )
        assert result == RiskAction.STOP_QUOTING

    def test_stop_vs_normal(self) -> None:
        result = BotLoop._worst_risk(
            RiskAction.STOP_QUOTING, RiskAction.QUOTE_NORMAL
        )
        assert result == RiskAction.STOP_QUOTING

    def test_symmetric(self) -> None:
        result = BotLoop._worst_risk(
            RiskAction.WIDEN_SPREAD, RiskAction.QUOTE_NORMAL
        )
        assert result == RiskAction.WIDEN_SPREAD


# -----------------------------------------------------------------------
# Kill switch integration
# -----------------------------------------------------------------------


class TestKillSwitch:
    def test_risk_manager_accessible(self, bot_loop: BotLoop) -> None:
        assert bot_loop.risk_manager is not None
        assert bot_loop.risk_manager.should_kill_all() is False

    def test_kill_switch_toggle(self, bot_loop: BotLoop) -> None:
        bot_loop.risk_manager.set_kill_switch(True)
        assert bot_loop.get_status()["kill_switch"] is True
        bot_loop.risk_manager.set_kill_switch(False)
        assert bot_loop.get_status()["kill_switch"] is False
