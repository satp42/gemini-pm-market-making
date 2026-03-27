"""Tests for the theta parameter optimizer.

Maps to acceptance criteria AC-4.1, AC-4.3, AC-4.4 from spec.md FR-4.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import PerformativeSettings
from src.db.models import Base, ThetaParameter
from src.engine.optimizer import OptimizationProgress, run_theta_optimization


def _generate_price_series(n: int = 200, seed: int = 42) -> list[float]:
    """Generate a synthetic price series for backtesting.

    Produces a mean-reverting series around 0.5 (prediction market range).
    """
    rng = np.random.default_rng(seed)
    prices = [0.5]
    for _ in range(1, n):
        # Mean-reverting random walk
        drift = -0.5 * (prices[-1] - 0.5)
        noise = rng.normal(0, 0.02)
        new_price = prices[-1] + drift * 0.1 + noise
        new_price = max(0.05, min(0.95, new_price))
        prices.append(float(new_price))
    return prices


@pytest.fixture
def small_settings() -> PerformativeSettings:
    """Settings with small trial/simulation counts for fast tests."""
    return PerformativeSettings(
        theta_optimization_trials=10,
        theta_optimization_simulations=10,
        xi_default=0.5,
    )


@pytest.fixture
async def db_session_factory():
    """Create an in-memory SQLite async session factory for testing."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    yield factory

    await engine.dispose()


@pytest.fixture
def sample_categories() -> dict[str, list[list[float]]]:
    """Sample categories with price series for optimization."""
    return {
        "sports": [
            _generate_price_series(200, seed=i) for i in range(5)
        ],
    }


class TestOptimizationProgress:
    """Test the OptimizationProgress dataclass defaults."""

    def test_default_values(self) -> None:
        progress = OptimizationProgress()
        assert progress.running is False
        assert progress.current_trial == 0
        assert progress.total_trials == 0
        assert progress.best_value is None
        assert progress.current_category == ""
        assert progress.categories_completed == []
        assert progress.started_at is None
        assert progress.completed_at is None
        assert progress.failed is False
        assert progress.error_message == ""

    def test_mutable_fields(self) -> None:
        progress = OptimizationProgress()
        progress.running = True
        progress.current_trial = 5
        progress.best_value = -0.95
        assert progress.running is True
        assert progress.current_trial == 5
        assert progress.best_value == -0.95


class TestShortOptimization:
    """AC-4.1: After optimization, theta values differ from (1.0, 1.0, 1.0)."""

    @pytest.mark.asyncio
    async def test_produces_non_trivial_theta(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        small_settings: PerformativeSettings,
        sample_categories: dict[str, list[list[float]]],
    ) -> None:
        """AC-4.1: 10 trials, 10 sims produces theta != (1, 1, 1)."""
        progress = OptimizationProgress()

        await run_theta_optimization(
            categories=sample_categories,
            progress=progress,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        # Verify theta was persisted
        async with db_session_factory() as session:
            result = await session.execute(
                select(ThetaParameter).where(ThetaParameter.category == "sports")
            )
            row = result.scalar_one_or_none()

        assert row is not None, "ThetaParameter row should exist after optimization"
        # At least one theta value should differ from 1.0
        thetas = (row.theta0, row.theta1, row.theta2)
        assert not all(
            abs(float(t) - 1.0) < 1e-6 for t in thetas
        ), f"Expected theta != (1,1,1), got {thetas}"

        # All theta values should be within the search range [0.5, 2.0]
        for t in thetas:
            assert 0.5 <= float(t) <= 2.0, f"Theta {t} out of range [0.5, 2.0]"


class TestProgressTracking:
    """AC-4.3: Progress tracking works correctly."""

    @pytest.mark.asyncio
    async def test_progress_running_false_after_completion(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        small_settings: PerformativeSettings,
        sample_categories: dict[str, list[list[float]]],
    ) -> None:
        """AC-4.3: progress.running is True during, False after."""
        progress = OptimizationProgress()

        assert progress.running is False

        await run_theta_optimization(
            categories=sample_categories,
            progress=progress,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        assert progress.running is False
        assert progress.completed_at is not None
        assert progress.started_at is not None
        assert progress.completed_at >= progress.started_at

    @pytest.mark.asyncio
    async def test_progress_callback_updates(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        small_settings: PerformativeSettings,
        sample_categories: dict[str, list[list[float]]],
    ) -> None:
        """Progress callback updates current_trial and best_value."""
        progress = OptimizationProgress()

        await run_theta_optimization(
            categories=sample_categories,
            progress=progress,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        # After 10 trials, current_trial should be 10
        assert progress.current_trial == small_settings.theta_optimization_trials
        # best_value should be set (CARA utility is negative)
        assert progress.best_value is not None
        assert isinstance(progress.best_value, float)

    @pytest.mark.asyncio
    async def test_categories_completed_populated(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        small_settings: PerformativeSettings,
        sample_categories: dict[str, list[list[float]]],
    ) -> None:
        """Categories completed list is populated after optimization."""
        progress = OptimizationProgress()

        await run_theta_optimization(
            categories=sample_categories,
            progress=progress,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        assert "sports" in progress.categories_completed


class TestDBPersistence:
    """AC-4.4: ThetaParameter row exists after optimization."""

    @pytest.mark.asyncio
    async def test_theta_persisted_to_db(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        small_settings: PerformativeSettings,
        sample_categories: dict[str, list[list[float]]],
    ) -> None:
        """AC-4.4: ThetaParameter row exists with correct category."""
        progress = OptimizationProgress()

        await run_theta_optimization(
            categories=sample_categories,
            progress=progress,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        async with db_session_factory() as session:
            result = await session.execute(
                select(ThetaParameter).where(ThetaParameter.category == "sports")
            )
            row = result.scalar_one_or_none()

        assert row is not None
        assert row.category == "sports"
        assert row.num_trials == 10
        assert row.xi_value is not None
        assert row.objective_value is not None
        assert row.optimized_at is not None

    @pytest.mark.asyncio
    async def test_upsert_overwrites_existing(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        small_settings: PerformativeSettings,
        sample_categories: dict[str, list[list[float]]],
    ) -> None:
        """Running optimization twice upserts (updates) the same row."""
        progress = OptimizationProgress()

        # First run
        await run_theta_optimization(
            categories=sample_categories,
            progress=progress,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        async with db_session_factory() as session:
            result = await session.execute(
                select(ThetaParameter).where(ThetaParameter.category == "sports")
            )
            first_row = result.scalar_one()
            first_id = first_row.id

        # Second run
        progress2 = OptimizationProgress()
        await run_theta_optimization(
            categories=sample_categories,
            progress=progress2,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        async with db_session_factory() as session:
            result = await session.execute(
                select(ThetaParameter).where(ThetaParameter.category == "sports")
            )
            rows = result.scalars().all()

        # Should still be just one row (upsert, not duplicate)
        assert len(rows) == 1
        assert rows[0].id == first_id  # Same row updated

    @pytest.mark.asyncio
    async def test_multiple_categories(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        small_settings: PerformativeSettings,
    ) -> None:
        """Multiple categories each get their own ThetaParameter row."""
        categories = {
            "sports": [_generate_price_series(200, seed=0)],
            "politics": [_generate_price_series(200, seed=1)],
        }
        progress = OptimizationProgress()

        await run_theta_optimization(
            categories=categories,
            progress=progress,
            session_factory=db_session_factory,
            settings=small_settings,
            gamma=0.1,
            k=1.5,
            sigma_default=0.01,
        )

        async with db_session_factory() as session:
            result = await session.execute(select(ThetaParameter))
            rows = result.scalars().all()

        assert len(rows) == 2
        cats = {r.category for r in rows}
        assert cats == {"sports", "politics"}
