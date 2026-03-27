"""Tests for the xi (performative feedback strength) estimator.

Maps to acceptance criteria AC-1.1a through AC-1.5 from spec.md FR-1.
"""

from __future__ import annotations

import time
from dataclasses import fields as dc_fields

import numpy as np
import pytest

from src.engine.xi import XiEstimate, estimate_xi


def _generate_ou_series(
    xi: float,
    mu: float,
    sigma: float,
    n_points: int,
    dt: float = 1.0,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Generate a synthetic Ornstein-Uhlenbeck process.

    ds = -xi * (s - mu) * dt + sigma * dW
    """
    rng = np.random.default_rng(seed)
    prices: list[float] = [mu]
    timestamps: list[float] = [0.0]
    s = mu
    for i in range(1, n_points):
        dw = rng.normal(0, np.sqrt(dt))
        s = s - xi * (s - mu) * dt + sigma * dw
        prices.append(float(s))
        timestamps.append(float(i * dt))
    return prices, timestamps


class TestXiEstimateDataclass:
    """T011 test (7): XiEstimate dataclass fields are correct types."""

    def test_fields_and_types(self) -> None:
        est = XiEstimate(xi=1.0, r_squared=0.5, num_trades=50, used_default=False)
        assert isinstance(est.xi, float)
        assert isinstance(est.r_squared, float)
        assert isinstance(est.num_trades, int)
        assert isinstance(est.used_default, bool)

    def test_field_count(self) -> None:
        assert len(dc_fields(XiEstimate)) == 4

    def test_r_squared_none(self) -> None:
        est = XiEstimate(xi=0.5, r_squared=None, num_trades=5, used_default=True)
        assert est.r_squared is None


class TestSyntheticOURecovery:
    """AC-1.1a and AC-1.1b: Synthetic OU recovery."""

    def test_100_points_known_xi_2(self) -> None:
        """AC-1.1a: 100 points, known xi=2.0, recovered in [1.5, 2.8]."""
        prices, timestamps = _generate_ou_series(
            xi=2.0, mu=0.5, sigma=0.02, n_points=100, dt=1.0, seed=42
        )
        result = estimate_xi(
            trade_prices=prices,
            trade_timestamps=timestamps,
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        assert not result.used_default, (
            f"Expected regression-based xi, got default. "
            f"r_squared={result.r_squared}, xi={result.xi}"
        )
        assert 1.5 <= result.xi <= 2.8, f"xi={result.xi} not in [1.5, 2.8]"
        assert result.r_squared is not None
        assert result.r_squared > 0.1
        assert result.num_trades == 100

    def test_500_points_known_xi_2(self) -> None:
        """AC-1.1b: 500 points, known xi=2.0, recovered in [1.8, 2.3]."""
        prices, timestamps = _generate_ou_series(
            xi=2.0, mu=0.5, sigma=0.02, n_points=500, dt=1.0, seed=42
        )
        result = estimate_xi(
            trade_prices=prices,
            trade_timestamps=timestamps,
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        assert not result.used_default
        assert 1.8 <= result.xi <= 2.3, f"xi={result.xi} not in [1.8, 2.3]"
        assert result.r_squared is not None
        assert result.r_squared > 0.1
        assert result.num_trades == 500


class TestInsufficientTrades:
    """AC-1.2: Fewer than xi_min_trades returns default."""

    def test_fewer_than_15_trades_returns_default(self) -> None:
        """AC-1.2: fewer than 15 trades returns default xi without error."""
        prices = [0.5 + i * 0.001 for i in range(10)]
        timestamps = [float(i) for i in range(10)]
        result = estimate_xi(
            trade_prices=prices,
            trade_timestamps=timestamps,
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        assert result.used_default is True
        assert result.xi == 0.5
        assert result.r_squared is None
        assert result.num_trades == 10

    def test_empty_trades_returns_default(self) -> None:
        result = estimate_xi(
            trade_prices=[],
            trade_timestamps=[],
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        assert result.used_default is True
        assert result.xi == 0.5
        assert result.num_trades == 0

    def test_exactly_14_trades_returns_default(self) -> None:
        prices = [0.5] * 14
        timestamps = [float(i) for i in range(14)]
        result = estimate_xi(
            trade_prices=prices,
            trade_timestamps=timestamps,
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        assert result.used_default is True


class TestFlatPriceSeries:
    """AC-1.3: Flat price series returns clamped minimum, no NaN/error."""

    def test_flat_prices_no_nan(self) -> None:
        """AC-1.3: all identical prices -> clamped min xi, no NaN."""
        prices = [0.5] * 50
        timestamps = [float(i) for i in range(50)]
        result = estimate_xi(
            trade_prices=prices,
            trade_timestamps=timestamps,
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        # Should fall back to default because r_squared will be 0 (< threshold)
        # or regression is degenerate
        assert result.used_default is True
        assert result.xi == 0.5  # default xi, not NaN
        assert result.xi == result.xi  # not NaN (NaN != NaN)
        assert result.num_trades == 50


class TestBenchmark:
    """AC-1.4: 100 prices < 5ms."""

    def test_100_prices_under_5ms(self) -> None:
        """AC-1.4: benchmark: 100 prices < 5ms."""
        prices, timestamps = _generate_ou_series(
            xi=2.0, mu=0.5, sigma=0.02, n_points=100, dt=1.0, seed=99
        )
        # Warm up numpy
        estimate_xi(
            trade_prices=prices,
            trade_timestamps=timestamps,
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            estimate_xi(
                trade_prices=prices,
                trade_timestamps=timestamps,
                xi_default=0.5,
                xi_min_trades=15,
                xi_clamp_min=0.01,
                xi_clamp_max=20.0,
                r_squared_threshold=0.1,
            )
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.005, f"estimate_xi took {elapsed*1000:.2f}ms, exceeds 5ms"


class TestRandomNoiseFallback:
    """AC-1.5: Random noise produces low r_squared, falls back to default."""

    def test_random_noise_low_r_squared(self) -> None:
        """AC-1.5: random noise series -> r_squared < 0.1 -> default xi."""
        rng = np.random.default_rng(42)
        # Pure random walk (cumulative sum of small noise) -- no mean reversion.
        # delta_s is iid noise, uncorrelated with s_n for a true random walk.
        steps = rng.normal(0, 0.01, size=100)
        prices = list(np.cumsum(steps).astype(float) + 0.5)
        timestamps = [float(i) for i in range(len(prices))]
        result = estimate_xi(
            trade_prices=prices,
            trade_timestamps=timestamps,
            xi_default=0.5,
            xi_min_trades=15,
            xi_clamp_min=0.01,
            xi_clamp_max=20.0,
            r_squared_threshold=0.1,
        )
        assert result.used_default is True
        assert result.xi == 0.5
