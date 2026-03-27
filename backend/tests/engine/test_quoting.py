"""Tests for the Avellaneda-Stoikov quoting engine."""

from __future__ import annotations

import math

import pytest

from src.engine.quoting import compute_quote, estimate_variance

# -----------------------------------------------------------------------
# compute_quote
# -----------------------------------------------------------------------

# Shared parameter sets -- kept short to stay within line limits
_NARROW = dict(mid_price=0.5, gamma=0.5, sigma_sq=0.01, t_minus_t=0.5, k=50.0)
_DEFAULT = dict(mid_price=0.5, gamma=0.1, sigma_sq=0.01, t_minus_t=0.5, k=1.5)


class TestComputeQuote:
    """Tests for the core A-S formula implementation."""

    def test_symmetric_quote_at_zero_inventory(self) -> None:
        """Zero inventory => reservation price equals mid, bid/ask symmetric."""
        q = compute_quote(inventory=0, **_NARROW)

        assert q.reservation_price == pytest.approx(0.5)
        assert q.bid_price == pytest.approx(
            q.reservation_price - q.spread / 2.0
        )
        assert q.mid_price == 0.5
        assert q.inventory == 0
        assert q.spread > 0

    def test_reservation_price_formula(self) -> None:
        """r = s - q * gamma * sigma^2 * (T - t)."""
        mid, inv, gamma, sigma_sq, t = 0.5, 10.0, 0.1, 0.01, 0.5
        q = compute_quote(
            mid_price=mid, inventory=inv,
            gamma=gamma, sigma_sq=sigma_sq, t_minus_t=t, k=1.5,
        )
        expected_r = mid - inv * gamma * sigma_sq * t
        assert q.reservation_price == pytest.approx(expected_r)

    def test_spread_formula(self) -> None:
        """delta* = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)."""
        gamma, sigma_sq, t, k = 0.1, 0.01, 0.5, 1.5
        q = compute_quote(
            mid_price=0.5, inventory=0,
            gamma=gamma, sigma_sq=sigma_sq, t_minus_t=t, k=k,
        )
        expected = gamma * sigma_sq * t + (2.0 / gamma) * math.log(
            1.0 + gamma / k
        )
        assert q.spread == pytest.approx(expected)

    def test_positive_inventory_shifts_reservation_down(self) -> None:
        """Long inventory should lower the reservation price."""
        zero_inv = compute_quote(inventory=0, **_NARROW)
        long_inv = compute_quote(inventory=5, **_NARROW)

        assert long_inv.reservation_price < zero_inv.reservation_price
        assert long_inv.bid_price <= zero_inv.bid_price

    def test_negative_inventory_shifts_reservation_up(self) -> None:
        """Short inventory should raise the reservation price."""
        zero_inv = compute_quote(inventory=0, **_NARROW)
        short_inv = compute_quote(inventory=-5, **_NARROW)

        assert short_inv.reservation_price > zero_inv.reservation_price
        assert short_inv.ask_price >= zero_inv.ask_price

    def test_bid_clamped_to_lower_bound(self) -> None:
        """Bid should never go below 0.01."""
        q = compute_quote(
            mid_price=0.05, inventory=100,
            gamma=0.5, sigma_sq=0.1, t_minus_t=1.0, k=1.5,
        )
        assert q.bid_price >= 0.01

    def test_ask_clamped_to_upper_bound(self) -> None:
        """Ask should never exceed 0.99."""
        q = compute_quote(
            mid_price=0.95, inventory=-100,
            gamma=0.5, sigma_sq=0.1, t_minus_t=1.0, k=1.5,
        )
        assert q.ask_price <= 0.99

    def test_quote_is_frozen_dataclass(self) -> None:
        """Quote should be immutable."""
        q = compute_quote(inventory=0, **_DEFAULT)
        with pytest.raises(AttributeError):
            q.bid_price = 0.99  # type: ignore[misc]

    def test_bid_always_less_than_or_equal_ask(self) -> None:
        """Across various parameter sets, bid <= ask."""
        params = [
            dict(inventory=0, **_DEFAULT),
            dict(
                mid_price=0.3, inventory=50,
                gamma=0.3, sigma_sq=0.05, t_minus_t=0.8, k=2.0,
            ),
            dict(
                mid_price=0.9, inventory=-20,
                gamma=0.05, sigma_sq=0.001, t_minus_t=0.1, k=0.5,
            ),
        ]
        for p in params:
            q = compute_quote(**p)
            assert q.bid_price <= q.ask_price


# -----------------------------------------------------------------------
# estimate_variance
# -----------------------------------------------------------------------


class TestEstimateVariance:
    """Tests for the variance estimator."""

    def test_returns_default_for_fewer_than_10_trades(self) -> None:
        assert estimate_variance([0.5, 0.51, 0.52]) == 0.01
        assert estimate_variance([], default=0.05) == 0.05

    def test_returns_default_for_exactly_9_trades(self) -> None:
        assert estimate_variance([0.5] * 9) == 0.01

    def test_computes_variance_for_10_trades(self) -> None:
        prices = [0.50 + i * 0.01 for i in range(10)]
        var = estimate_variance(prices)
        assert var > 0
        assert isinstance(var, float)

    def test_constant_prices_return_floor(self) -> None:
        """Constant prices => zero variance, floored to 0.0001."""
        prices = [0.5] * 20
        var = estimate_variance(prices)
        assert var == pytest.approx(0.0001)

    def test_variance_increases_with_volatility(self) -> None:
        calm = [0.50 + i * 0.001 for i in range(20)]
        wild = [0.50 + ((-1) ** i) * 0.05 for i in range(20)]
        assert estimate_variance(wild) > estimate_variance(calm)

    def test_custom_default(self) -> None:
        assert estimate_variance([1.0], default=0.42) == 0.42
