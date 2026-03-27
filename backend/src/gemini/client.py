"""Async HTTP client for the Gemini Prediction Markets API."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

import httpx

from src.config import Settings
from src.gemini.auth import make_auth_headers
from src.gemini.models import (
    Event,
    EventsResponse,
    Order,
    OrderBook,
    OrderBookEntry,
    Position,
    Ticker,
    Trade,
)

logger = logging.getLogger(__name__)


class GeminiAPIError(Exception):
    """Raised when the Gemini API returns an error response."""

    def __init__(self, status_code: int, reason: str, message: str = "") -> None:
        self.status_code = status_code
        self.reason = reason
        self.message = message
        super().__init__(f"Gemini API error {status_code}: {reason} - {message}")


class GeminiClient:
    """Async client for Gemini Prediction Markets REST API.

    Provides public (unauthenticated) and private (HMAC-authenticated)
    methods. Includes retry with exponential backoff for transient failures.
    """

    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2
    RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.gemini.api_key
        self._api_secret = settings.gemini.api_secret
        self._base_url = settings.gemini.base_url
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        authenticated: bool = False,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Execute an HTTP request with exponential-backoff retry on transient errors."""
        last_exc: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                headers: dict[str, str] = {}
                if authenticated:
                    headers = make_auth_headers(
                        self._api_key,
                        self._api_secret,
                        path,
                        payload,
                    )

                response = await self._client.request(
                    method,
                    path,
                    headers=headers,
                    params=params,
                )

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    raise GeminiAPIError(
                        response.status_code,
                        "server_error",
                        response.text,
                    )

                if response.status_code >= 400:
                    body = response.json() if response.text else {}
                    raise GeminiAPIError(
                        response.status_code,
                        body.get("reason", "unknown"),
                        body.get("message", response.text),
                    )

                return response.json()

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                logger.warning(
                    "Gemini request %s %s attempt %d failed: %s",
                    method,
                    path,
                    attempt + 1,
                    exc,
                )
            except GeminiAPIError as exc:
                if exc.status_code not in self.RETRYABLE_STATUS_CODES:
                    raise
                last_exc = exc
                logger.warning(
                    "Gemini request %s %s attempt %d got %d, retrying",
                    method,
                    path,
                    attempt + 1,
                    exc.status_code,
                )

            if attempt < self.MAX_RETRIES - 1:
                delay = self.BACKOFF_FACTOR ** attempt
                await asyncio.sleep(delay)

        raise last_exc or GeminiAPIError(0, "unknown", "All retries exhausted")

    # ------------------------------------------------------------------
    # Public endpoints (no auth)
    # ------------------------------------------------------------------

    async def get_events(
        self,
        status: str = "active",
        limit: int = 50,
        offset: int = 0,
    ) -> EventsResponse:
        """Fetch prediction market events with optional status filter."""
        params: dict[str, Any] = {
            "status[]": status,
            "limit": limit,
            "offset": offset,
        }
        data = await self._request_with_retry("GET", "/v1/prediction-markets/events", params=params)

        # The API may return the list directly or wrapped in an object
        if isinstance(data, list):
            return EventsResponse(events=[Event.model_validate(e) for e in data], pagination={})

        return EventsResponse.model_validate(data)

    async def get_newly_listed(self) -> list[Event]:
        """Fetch newly listed prediction market events."""
        data = await self._request_with_retry(
            "GET", "/v1/prediction-markets/events/newly-listed"
        )
        if isinstance(data, list):
            return [Event.model_validate(e) for e in data]
        events_data = data.get("events", data) if isinstance(data, dict) else data
        return [Event.model_validate(e) for e in events_data]

    async def get_order_book(self, symbol: str) -> OrderBook:
        """Fetch the order book for a given instrument symbol."""
        data = await self._request_with_retry("GET", f"/v1/book/{symbol}")

        bids = [
            OrderBookEntry(
                price=Decimal(e["price"]),
                amount=Decimal(e["amount"]),
                timestamp=e.get("timestamp", ""),
            )
            for e in data.get("bids", [])
        ]
        asks = [
            OrderBookEntry(
                price=Decimal(e["price"]),
                amount=Decimal(e["amount"]),
                timestamp=e.get("timestamp", ""),
            )
            for e in data.get("asks", [])
        ]
        return OrderBook(bids=bids, asks=asks)

    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch ticker data for a symbol."""
        data = await self._request_with_retry("GET", f"/v1/pubticker/{symbol}")
        return Ticker.model_validate(data)

    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Fetch recent trades for a symbol."""
        params = {"limit_trades": limit}
        data = await self._request_with_retry(
            "GET", f"/v1/trades/{symbol}", params=params
        )
        return [Trade.model_validate(t) for t in data]

    # ------------------------------------------------------------------
    # Authenticated endpoints
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        outcome: str,
        price: Decimal,
        quantity: int,
        time_in_force: str = "good-til-cancel",
    ) -> Order:
        """Place a limit order on a prediction market contract.

        Parameters
        ----------
        symbol:
            Instrument symbol (e.g. ``GEMI-BTC100K-26MAR26``).
        side:
            ``buy`` or ``sell``.
        outcome:
            ``yes`` or ``no``.
        price:
            Limit price as a Decimal (0.01 - 0.99).
        quantity:
            Number of contracts.
        time_in_force:
            Order duration policy. Default ``good-til-cancel``.
        """
        payload = {
            "symbol": symbol,
            "side": side,
            "outcome": outcome,
            "price": str(price),
            "quantity": str(quantity),
            "type": "limit",
            "timeInForce": time_in_force,
        }
        data = await self._request_with_retry(
            "POST",
            "/v1/prediction-markets/order",
            authenticated=True,
            payload=payload,
        )
        return Order.model_validate(data)

    async def cancel_order(self, order_id: int) -> Order:
        """Cancel an active order by its Gemini order ID."""
        payload = {"orderId": order_id}
        data = await self._request_with_retry(
            "POST",
            "/v1/prediction-markets/order/cancel",
            authenticated=True,
            payload=payload,
        )
        return Order.model_validate(data)

    async def get_positions(self) -> list[Position]:
        """Fetch all current prediction market positions."""
        data = await self._request_with_retry(
            "POST",
            "/v1/prediction-markets/positions",
            authenticated=True,
        )
        if isinstance(data, list):
            return [Position.model_validate(p) for p in data]
        positions_data = data.get("positions", []) if isinstance(data, dict) else []
        return [Position.model_validate(p) for p in positions_data]

    async def get_active_orders(self, symbol: str | None = None) -> list[Order]:
        """Fetch active orders, optionally filtered by symbol."""
        payload: dict[str, Any] = {}
        if symbol:
            payload["symbol"] = symbol
        data = await self._request_with_retry(
            "POST",
            "/v1/prediction-markets/orders/active",
            authenticated=True,
            payload=payload,
        )
        if isinstance(data, list):
            return [Order.model_validate(o) for o in data]
        orders_data = data.get("orders", []) if isinstance(data, dict) else []
        return [Order.model_validate(o) for o in orders_data]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
