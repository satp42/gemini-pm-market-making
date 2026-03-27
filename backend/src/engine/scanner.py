"""Market scanner -- selects which prediction markets to quote on."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import BotSettings
from src.gemini.client import GeminiClient
from src.gemini.models import Contract, Event

logger = logging.getLogger(__name__)


class MarketScanner:
    """Scans Gemini events and filters contracts eligible for market-making."""

    def __init__(self, client: GeminiClient, bot_settings: BotSettings) -> None:
        self._client = client
        self._settings = bot_settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan(self) -> tuple[list[str], dict[str, str]]:
        """Return instrument symbols eligible for quoting from active events.

        Returns a tuple of (symbols, symbol_titles) where symbol_titles maps
        each symbol to its parent event title.

        Filters applied per contract:
        1. Contract must have both a best bid and best ask.
        2. Spread (bestAsk - bestBid) >= min_spread.
        3. Symbol not in the excluded list.
        4. Time to expiry > min_time_to_expiry_hours.
        """
        events_response = await self._client.get_events(status="active")
        events = events_response.events

        selected: list[str] = []
        symbol_titles: dict[str, str] = {}
        now = datetime.now(timezone.utc)

        logger.info(
            "Scanner evaluating %d events (min_spread=%.4f, min_expiry_h=%d)",
            len(events),
            self._settings.min_spread,
            self._settings.min_time_to_expiry_hours,
        )

        for event in events:
            hours_to_expiry = self._hours_to_expiry(event, now)

            if (
                hours_to_expiry is not None
                and hours_to_expiry < self._settings.min_time_to_expiry_hours
            ):
                logger.debug(
                    "Skipping event %s -- expires in %.1f hours (min %d)",
                    event.event_ticker,
                    hours_to_expiry,
                    self._settings.min_time_to_expiry_hours,
                )
                continue

            for contract in event.contracts:
                symbol = contract.instrument_symbol
                if self._is_eligible(contract, symbol, hours_to_expiry):
                    selected.append(symbol)
                    symbol_titles[symbol] = event.title
                    spread_val = (
                        float(contract.best_ask - contract.best_bid)
                        if contract.best_ask is not None and contract.best_bid is not None
                        else 0.0
                    )
                    logger.info(
                        "Selected %s -- spread=%.4f, expiry_h=%.1f",
                        symbol,
                        spread_val,
                        hours_to_expiry if hours_to_expiry is not None else -1,
                    )

        logger.info("Scanner found %d eligible markets", len(selected))
        return selected, symbol_titles

    async def scan_newly_listed(self) -> list[str]:
        """Return symbols from newly listed events that pass filters."""
        events = await self._client.get_newly_listed()
        selected: list[str] = []
        now = datetime.now(timezone.utc)

        for event in events:
            hours_to_expiry = self._hours_to_expiry(event, now)

            if (
                hours_to_expiry is not None
                and hours_to_expiry < self._settings.min_time_to_expiry_hours
            ):
                continue

            for contract in event.contracts:
                symbol = contract.instrument_symbol
                if self._is_eligible(contract, symbol, hours_to_expiry):
                    selected.append(symbol)

        logger.info("Newly-listed scan found %d eligible markets", len(selected))
        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_eligible(
        self,
        contract: Contract,
        symbol: str,
        hours_to_expiry: float | None,
    ) -> bool:
        """Check whether a single contract passes all eligibility filters."""
        if symbol in self._settings.excluded_symbols:
            logger.debug("Rejected %s -- excluded symbol", symbol)
            return False

        if contract.best_bid is None or contract.best_ask is None:
            logger.debug("Rejected %s -- missing bid or ask", symbol)
            return False

        spread = float(contract.best_ask - contract.best_bid)
        if spread < self._settings.min_spread:
            logger.debug(
                "Rejected %s -- spread %.4f < min %.4f",
                symbol,
                spread,
                self._settings.min_spread,
            )
            return False

        if (
            hours_to_expiry is not None
            and hours_to_expiry < self._settings.min_time_to_expiry_hours
        ):
            logger.debug("Rejected %s -- expiry too soon (%.1fh)", symbol, hours_to_expiry)
            return False

        return True

    @staticmethod
    def _hours_to_expiry(event: Event, now: datetime) -> float | None:
        """Parse the event expiration date and return hours remaining, or None."""
        if not event.expiration_date:
            return None
        try:
            expiry = datetime.fromisoformat(event.expiration_date.replace("Z", "+00:00"))
            delta = expiry - now
            return delta.total_seconds() / 3600.0
        except (ValueError, TypeError):
            logger.warning("Could not parse expiration date: %s", event.expiration_date)
            return None
