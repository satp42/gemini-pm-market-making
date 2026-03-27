"""Tests for the risk manager."""

from __future__ import annotations

import pytest

from src.config import RiskSettings
from src.engine.risk import RiskAction, RiskManager

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def risk_settings() -> RiskSettings:
    return RiskSettings(
        MAX_INVENTORY_PER_SYMBOL=200,
        MAX_TOTAL_EXPOSURE=1000,
        RISK_WIDEN_THRESHOLD=0.8,
    )


@pytest.fixture
def risk_mgr(risk_settings: RiskSettings) -> RiskManager:
    return RiskManager(risk_settings)


# -----------------------------------------------------------------------
# check_symbol
# -----------------------------------------------------------------------


class TestCheckSymbol:
    def test_normal_low_inventory(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_symbol("SYM", 50) == RiskAction.QUOTE_NORMAL

    def test_normal_zero_inventory(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_symbol("SYM", 0) == RiskAction.QUOTE_NORMAL

    def test_widen_at_threshold(self, risk_mgr: RiskManager) -> None:
        # widen_threshold = 0.8 * 200 = 160, so 161 should trigger widen
        assert risk_mgr.check_symbol("SYM", 161) == RiskAction.WIDEN_SPREAD

    def test_widen_negative_inventory(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_symbol("SYM", -161) == RiskAction.WIDEN_SPREAD

    def test_stop_at_max(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_symbol("SYM", 200) == RiskAction.STOP_QUOTING

    def test_stop_above_max(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_symbol("SYM", 250) == RiskAction.STOP_QUOTING

    def test_stop_negative_at_max(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_symbol("SYM", -200) == RiskAction.STOP_QUOTING

    def test_exactly_at_widen_boundary(self, risk_mgr: RiskManager) -> None:
        # 160 is exactly the threshold; > threshold triggers widen, so 160 is normal
        assert risk_mgr.check_symbol("SYM", 160) == RiskAction.QUOTE_NORMAL


# -----------------------------------------------------------------------
# check_total_exposure
# -----------------------------------------------------------------------


class TestCheckTotalExposure:
    def test_within_limit(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_total_exposure({"A": 100, "B": -200}) is True

    def test_at_limit(self, risk_mgr: RiskManager) -> None:
        # total = 500 + 500 = 1000, not < 1000
        assert risk_mgr.check_total_exposure({"A": 500, "B": -500}) is False

    def test_above_limit(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_total_exposure({"A": 600, "B": -500}) is False

    def test_empty(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.check_total_exposure({}) is True


# -----------------------------------------------------------------------
# check_time_safety
# -----------------------------------------------------------------------


class TestCheckTimeSafety:
    def test_normal(self) -> None:
        assert RiskManager.check_time_safety(0.5) == RiskAction.QUOTE_NORMAL

    def test_widen_near_expiry(self) -> None:
        assert RiskManager.check_time_safety(0.03) == RiskAction.WIDEN_SPREAD

    def test_stop_very_near_expiry(self) -> None:
        assert RiskManager.check_time_safety(0.005) == RiskAction.STOP_QUOTING

    def test_boundary_at_0_01(self) -> None:
        # 0.01 is NOT < 0.01, so should be WIDEN (< 0.04)
        assert RiskManager.check_time_safety(0.01) == RiskAction.WIDEN_SPREAD

    def test_boundary_at_0_04(self) -> None:
        # 0.04 is NOT < 0.04, so should be NORMAL
        assert RiskManager.check_time_safety(0.04) == RiskAction.QUOTE_NORMAL


# -----------------------------------------------------------------------
# Kill switch
# -----------------------------------------------------------------------


class TestKillSwitch:
    def test_default_off(self, risk_mgr: RiskManager) -> None:
        assert risk_mgr.should_kill_all() is False

    def test_engage(self, risk_mgr: RiskManager) -> None:
        risk_mgr.set_kill_switch(True)
        assert risk_mgr.should_kill_all() is True

    def test_disengage(self, risk_mgr: RiskManager) -> None:
        risk_mgr.set_kill_switch(True)
        risk_mgr.set_kill_switch(False)
        assert risk_mgr.should_kill_all() is False
