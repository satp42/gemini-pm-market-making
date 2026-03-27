"""Risk manager -- hard limits and safety checks."""

from __future__ import annotations

import logging
from enum import Enum

from src.config import RiskSettings

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Possible risk decisions for a symbol or the whole book."""

    QUOTE_NORMAL = "quote_normal"
    WIDEN_SPREAD = "widen_spread"
    STOP_QUOTING = "stop_quoting"


class RiskManager:
    """Evaluates risk limits and returns actions for the bot loop."""

    def __init__(self, risk_settings: RiskSettings) -> None:
        self._settings = risk_settings
        self._kill_switch: bool = False

    # ------------------------------------------------------------------
    # Per-symbol checks
    # ------------------------------------------------------------------

    def check_symbol(self, symbol: str, inventory: float) -> RiskAction:
        """Determine risk action for *symbol* given current inventory.

        - STOP_QUOTING if |inventory| >= max_inventory_per_symbol
        - WIDEN_SPREAD if |inventory| > risk_widen_threshold * max
        - QUOTE_NORMAL otherwise
        """
        abs_inv = abs(inventory)
        max_inv = self._settings.max_inventory_per_symbol

        if abs_inv >= max_inv:
            logger.warning(
                "RISK STOP: %s inventory %.1f >= max %d", symbol, inventory, max_inv
            )
            return RiskAction.STOP_QUOTING

        widen_threshold = self._settings.risk_widen_threshold * max_inv
        if abs_inv > widen_threshold:
            logger.info(
                "RISK WIDEN: %s inventory %.1f > threshold %.1f",
                symbol,
                inventory,
                widen_threshold,
            )
            return RiskAction.WIDEN_SPREAD

        return RiskAction.QUOTE_NORMAL

    # ------------------------------------------------------------------
    # Portfolio-level checks
    # ------------------------------------------------------------------

    def check_total_exposure(self, inventories: dict[str, float]) -> bool:
        """Return True if total absolute inventory is within max_total_exposure."""
        total = sum(abs(v) for v in inventories.values())
        within = total < self._settings.max_total_exposure
        if not within:
            logger.warning(
                "RISK: total exposure %.1f >= max %d",
                total,
                self._settings.max_total_exposure,
            )
        return within

    # ------------------------------------------------------------------
    # Time-based checks
    # ------------------------------------------------------------------

    @staticmethod
    def check_time_safety(t_minus_t: float) -> RiskAction:
        """Check time remaining until expiry.

        - STOP_QUOTING if t_minus_t < 0.01 (~15 minutes)
        - WIDEN_SPREAD if t_minus_t < 0.04 (~1 hour)
        - QUOTE_NORMAL otherwise
        """
        if t_minus_t < 0.01:
            logger.warning(
                "RISK STOP: t_minus_t=%.4f (< 0.01, ~15min to expiry)", t_minus_t
            )
            return RiskAction.STOP_QUOTING
        if t_minus_t < 0.04:
            logger.info(
                "RISK WIDEN: t_minus_t=%.4f (< 0.04, ~1h to expiry)", t_minus_t
            )
            return RiskAction.WIDEN_SPREAD
        return RiskAction.QUOTE_NORMAL

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def should_kill_all(self) -> bool:
        """Return True if the emergency kill switch is engaged."""
        return self._kill_switch

    def set_kill_switch(self, active: bool) -> None:
        """Engage or disengage the emergency kill switch."""
        self._kill_switch = active
        if active:
            logger.critical("KILL SWITCH ENGAGED -- all quoting will stop")
        else:
            logger.info("Kill switch disengaged -- quoting may resume")
