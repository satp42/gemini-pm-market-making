"""Tests for Phase 6: API Endpoints -- performative market-making extensions.

Covers tasks T023-T029:
- T023: ConfigUpdateRequest performative fields + validation
- T024: MarketSummary and QuoteHistoryEntry performative fields
- T025: POST /config effective response with performative params
- T026: POST /api/optimize/theta endpoint (202/409/503)
- T027: GET /api/optimize/theta/status endpoint
- T028: optimization_progress wired in lifespan
- T029: GET /markets and GET /markets/{symbol} with performative fields
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.router import (
    ConfigUpdateRequest,
    MarketSummary,
    QuoteHistoryEntry,
)
from src.db.models import Base, Quote
from src.engine.optimizer import OptimizationProgress


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def async_engine():
    """Create an in-memory SQLite async engine for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture()
async def session_factory(async_engine):
    """Create an async session factory bound to the test engine."""
    factory = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    return factory


@pytest.fixture()
async def seed_quotes(session_factory):
    """Seed some quote records with performative fields."""
    async with session_factory() as session:
        now = datetime.now(timezone.utc)
        await session.execute(
            insert(Quote).values(
                [
                    {
                        "timestamp": now,
                        "symbol": "TEST-SYM-1",
                        "mid_price": 0.5,
                        "reservation_price": 0.48,
                        "bid_price": 0.45,
                        "ask_price": 0.55,
                        "spread": 0.10,
                        "inventory": 2.0,
                        "sigma_sq": 0.01,
                        "gamma": 0.1,
                        "t_minus_t": 0.5,
                        "xi": 1.5,
                        "theta0": 1.1,
                        "theta1": 0.9,
                        "theta2": 1.05,
                        "quoting_mode": "theta",
                    },
                    {
                        "timestamp": now,
                        "symbol": "TEST-SYM-2",
                        "mid_price": 0.6,
                        "reservation_price": 0.58,
                        "bid_price": 0.55,
                        "ask_price": 0.65,
                        "spread": 0.10,
                        "inventory": 0.0,
                        "sigma_sq": 0.008,
                        "gamma": 0.1,
                        "t_minus_t": 0.3,
                        "xi": None,
                        "theta0": None,
                        "theta1": None,
                        "theta2": None,
                        "quoting_mode": None,
                    },
                ]
            )
        )
        await session.commit()


def _make_app(session_factory, bot_loop=None, optimization_progress=None):
    """Build a minimal FastAPI app with patched DB and state for testing."""
    from fastapi import FastAPI

    from src.api.router import router
    from src.config import Settings

    app = FastAPI()
    app.include_router(router, prefix="/api")

    settings = Settings()
    app.state.settings = settings
    app.state.config_overrides = {}
    app.state.bot_loop = bot_loop
    app.state.optimization_progress = optimization_progress or OptimizationProgress()

    # Patch get_session to use test session factory
    from src.api import router as router_module

    async def _test_get_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[router_module.get_session] = _test_get_session
    return app


# ---------------------------------------------------------------------------
# T023: ConfigUpdateRequest validation
# ---------------------------------------------------------------------------


class TestConfigUpdateRequestValidation:
    """T023: Performative fields on ConfigUpdateRequest with field_validator."""

    def test_valid_quoting_mode_as(self) -> None:
        req = ConfigUpdateRequest(quoting_mode="as")
        assert req.quoting_mode == "as"

    def test_valid_quoting_mode_performative(self) -> None:
        req = ConfigUpdateRequest(quoting_mode="performative")
        assert req.quoting_mode == "performative"

    def test_valid_quoting_mode_theta(self) -> None:
        req = ConfigUpdateRequest(quoting_mode="theta")
        assert req.quoting_mode == "theta"

    def test_valid_quoting_mode_none(self) -> None:
        req = ConfigUpdateRequest(quoting_mode=None)
        assert req.quoting_mode is None

    def test_invalid_quoting_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="quoting_mode must be one of"):
            ConfigUpdateRequest(quoting_mode="invalid")

    def test_performative_fields_present(self) -> None:
        req = ConfigUpdateRequest(
            quoting_mode="theta",
            xi_default=0.8,
            xi_min_trades=20,
            xi_clamp_min=0.05,
            xi_clamp_max=10.0,
            q_ref=1.0,
        )
        assert req.xi_default == 0.8
        assert req.xi_min_trades == 20
        assert req.xi_clamp_min == 0.05
        assert req.xi_clamp_max == 10.0
        assert req.q_ref == 1.0

    def test_performative_fields_default_none(self) -> None:
        req = ConfigUpdateRequest()
        assert req.quoting_mode is None
        assert req.xi_default is None
        assert req.xi_min_trades is None
        assert req.xi_clamp_min is None
        assert req.xi_clamp_max is None
        assert req.q_ref is None


# ---------------------------------------------------------------------------
# T024: MarketSummary and QuoteHistoryEntry performative fields
# ---------------------------------------------------------------------------


class TestPydanticSchemaFields:
    """T024: Performative fields on MarketSummary and QuoteHistoryEntry."""

    def test_market_summary_performative_fields_default_none(self) -> None:
        ms = MarketSummary(
            symbol="TEST",
            midPrice=0.5,
            reservationPrice=0.48,
            bidPrice=0.45,
            askPrice=0.55,
            spread=0.1,
            inventory=0,
            sigmaSquared=0.01,
            gamma=0.1,
            timeRemaining=0.5,
        )
        assert ms.xi is None
        assert ms.theta0 is None
        assert ms.theta1 is None
        assert ms.theta2 is None
        assert ms.quotingMode is None

    def test_market_summary_performative_fields_populated(self) -> None:
        ms = MarketSummary(
            symbol="TEST",
            midPrice=0.5,
            reservationPrice=0.48,
            bidPrice=0.45,
            askPrice=0.55,
            spread=0.1,
            inventory=0,
            sigmaSquared=0.01,
            gamma=0.1,
            timeRemaining=0.5,
            xi=1.5,
            theta0=1.1,
            theta1=0.9,
            theta2=1.05,
            quotingMode="theta",
        )
        assert ms.xi == 1.5
        assert ms.theta0 == 1.1
        assert ms.theta1 == 0.9
        assert ms.theta2 == 1.05
        assert ms.quotingMode == "theta"

    def test_quote_history_entry_performative_fields_default_none(self) -> None:
        entry = QuoteHistoryEntry(
            id=1,
            timestamp="2026-03-26T10:00:00",
            midPrice=0.5,
            reservationPrice=0.48,
            bidPrice=0.45,
            askPrice=0.55,
            spread=0.1,
            inventory=0,
            sigmaSquared=0.01,
            gamma=0.1,
            timeRemaining=0.5,
        )
        assert entry.xi is None
        assert entry.theta0 is None
        assert entry.theta1 is None
        assert entry.theta2 is None
        assert entry.quotingMode is None

    def test_quote_history_entry_performative_fields_populated(self) -> None:
        entry = QuoteHistoryEntry(
            id=1,
            timestamp="2026-03-26T10:00:00",
            midPrice=0.5,
            reservationPrice=0.48,
            bidPrice=0.45,
            askPrice=0.55,
            spread=0.1,
            inventory=0,
            sigmaSquared=0.01,
            gamma=0.1,
            timeRemaining=0.5,
            xi=2.0,
            theta0=1.2,
            theta1=0.8,
            theta2=1.1,
            quotingMode="performative",
        )
        assert entry.xi == 2.0
        assert entry.quotingMode == "performative"


# ---------------------------------------------------------------------------
# T025: POST /config effective response with performative params
# ---------------------------------------------------------------------------


class TestConfigEffectiveResponse:
    """T025: POST /config returns performative params in effective dict."""

    async def test_config_returns_performative_defaults(self, session_factory) -> None:
        app = _make_app(session_factory)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/config", json={"gamma": 0.5})
            assert resp.status_code == 200
            data = resp.json()
            effective = data["effective"]
            # Performative defaults must be present
            assert "quoting_mode" in effective
            assert "xi_default" in effective
            assert "xi_min_trades" in effective
            assert "xi_clamp_min" in effective
            assert "xi_clamp_max" in effective
            assert "q_ref" in effective
            # Check default values
            assert effective["quoting_mode"] == "theta"
            assert effective["xi_default"] == 0.5
            assert effective["xi_min_trades"] == 15
            assert effective["q_ref"] == 0.0

    async def test_config_override_quoting_mode(self, session_factory) -> None:
        app = _make_app(session_factory)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/config", json={"quoting_mode": "as"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["effective"]["quoting_mode"] == "as"
            assert data["overrides"]["quoting_mode"] == "as"

    async def test_config_invalid_quoting_mode_422(self, session_factory) -> None:
        app = _make_app(session_factory)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/config", json={"quoting_mode": "invalid"})
            assert resp.status_code == 422

    async def test_config_override_xi_default(self, session_factory) -> None:
        app = _make_app(session_factory)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/config", json={"xi_default": 1.5})
            assert resp.status_code == 200
            data = resp.json()
            assert data["effective"]["xi_default"] == 1.5


# ---------------------------------------------------------------------------
# T026: POST /api/optimize/theta
# ---------------------------------------------------------------------------


class TestOptimizeThetaEndpoint:
    """T026: POST /api/optimize/theta endpoint."""

    async def test_returns_503_when_bot_loop_none(self, session_factory) -> None:
        app = _make_app(session_factory, bot_loop=None)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/optimize/theta")
            assert resp.status_code == 503
            assert "Bot loop not initialized" in resp.json()["detail"]

    async def test_returns_409_when_already_running(self, session_factory) -> None:
        progress = OptimizationProgress(running=True)
        bot_loop = MagicMock()
        bot_loop._symbol_categories = {}
        app = _make_app(session_factory, bot_loop=bot_loop, optimization_progress=progress)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/optimize/theta")
            assert resp.status_code == 409
            assert "already running" in resp.json()["detail"]

    async def test_returns_202_and_starts_optimization(self, session_factory) -> None:
        progress = OptimizationProgress(running=False)
        bot_loop = MagicMock()
        bot_loop.symbol_categories = {"SYM-A": "Crypto", "SYM-B": "Crypto"}
        bot_loop.client = MagicMock()
        # Mock get_trades to return some fake trade data
        mock_trade = MagicMock()
        mock_trade.price = 0.5
        bot_loop.client.get_trades = AsyncMock(return_value=[mock_trade] * 10)

        app = _make_app(session_factory, bot_loop=bot_loop, optimization_progress=progress)

        with patch("src.api.router.run_theta_optimization", new_callable=AsyncMock) as mock_opt:
            with patch("src.api.router.get_session_factory", return_value=session_factory):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post("/api/optimize/theta")
                    assert resp.status_code == 202
                    data = resp.json()
                    assert data["status"] == "started"
                    assert "Crypto" in data["categories"]

    async def test_category_filter(self, session_factory) -> None:
        progress = OptimizationProgress(running=False)
        bot_loop = MagicMock()
        bot_loop.symbol_categories = {
            "SYM-A": "Crypto",
            "SYM-B": "Sports",
        }
        bot_loop.client = MagicMock()
        bot_loop.client.get_trades = AsyncMock(return_value=[])

        app = _make_app(session_factory, bot_loop=bot_loop, optimization_progress=progress)

        with patch("src.api.router.run_theta_optimization", new_callable=AsyncMock):
            with patch("src.api.router.get_session_factory", return_value=session_factory):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post("/api/optimize/theta?category=Crypto")
                    assert resp.status_code == 202
                    data = resp.json()
                    assert data["categories"] == ["Crypto"]


# ---------------------------------------------------------------------------
# T027: GET /api/optimize/theta/status
# ---------------------------------------------------------------------------


class TestOptimizeThetaStatus:
    """T027: GET /api/optimize/theta/status endpoint."""

    async def test_returns_idle_state(self, session_factory) -> None:
        progress = OptimizationProgress()
        app = _make_app(session_factory, optimization_progress=progress)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/optimize/theta/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["running"] is False
            assert data["currentTrial"] == 0
            assert data["totalTrials"] == 0
            assert data["bestValue"] is None
            assert data["currentCategory"] == ""
            assert data["categoriesCompleted"] == []
            assert data["startedAt"] is None
            assert data["completedAt"] is None

    async def test_returns_running_state(self, session_factory) -> None:
        now = datetime.now(timezone.utc)
        progress = OptimizationProgress(
            running=True,
            current_trial=47,
            total_trials=100,
            best_value=-0.0023,
            current_category="Crypto",
            categories_completed=[],
            started_at=now,
        )
        app = _make_app(session_factory, optimization_progress=progress)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/optimize/theta/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["running"] is True
            assert data["currentTrial"] == 47
            assert data["totalTrials"] == 100
            assert data["bestValue"] == pytest.approx(-0.0023)
            assert data["currentCategory"] == "Crypto"
            assert data["startedAt"] is not None
            assert data["completedAt"] is None

    async def test_returns_completed_state(self, session_factory) -> None:
        now = datetime.now(timezone.utc)
        progress = OptimizationProgress(
            running=False,
            current_trial=100,
            total_trials=100,
            best_value=-0.001,
            current_category="",
            categories_completed=["Crypto", "Sports"],
            started_at=now,
            completed_at=now,
        )
        app = _make_app(session_factory, optimization_progress=progress)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/optimize/theta/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["running"] is False
            assert data["categoriesCompleted"] == ["Crypto", "Sports"]
            assert data["completedAt"] is not None


# ---------------------------------------------------------------------------
# T029: GET /markets and GET /markets/{symbol} with performative fields
# ---------------------------------------------------------------------------


class TestMarketsPerformativeFields:
    """T029: Market endpoints include performative fields from QuoteRecord."""

    async def test_get_markets_includes_performative_fields(
        self, session_factory, seed_quotes
    ) -> None:
        bot_loop = MagicMock()
        bot_loop.symbol_titles = {}
        app = _make_app(session_factory, bot_loop=bot_loop)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/markets")
            assert resp.status_code == 200
            markets = resp.json()
            assert len(markets) >= 1

            # Find the symbol with performative data
            sym1 = next((m for m in markets if m["symbol"] == "TEST-SYM-1"), None)
            assert sym1 is not None
            assert sym1["xi"] == pytest.approx(1.5)
            assert sym1["theta0"] == pytest.approx(1.1)
            assert sym1["theta1"] == pytest.approx(0.9)
            assert sym1["theta2"] == pytest.approx(1.05)
            assert sym1["quotingMode"] == "theta"

            # Find the symbol without performative data
            sym2 = next((m for m in markets if m["symbol"] == "TEST-SYM-2"), None)
            assert sym2 is not None
            assert sym2["xi"] is None
            assert sym2["theta0"] is None
            assert sym2["quotingMode"] is None

    async def test_get_market_detail_includes_performative_fields(
        self, session_factory, seed_quotes
    ) -> None:
        app = _make_app(session_factory)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/markets/TEST-SYM-1")
            assert resp.status_code == 200
            data = resp.json()

            # quoteHistory entries should have performative fields
            assert len(data["quoteHistory"]) >= 1
            entry = data["quoteHistory"][0]
            assert entry["xi"] == pytest.approx(1.5)
            assert entry["theta0"] == pytest.approx(1.1)
            assert entry["theta1"] == pytest.approx(0.9)
            assert entry["theta2"] == pytest.approx(1.05)
            assert entry["quotingMode"] == "theta"

            # currentQuote should also have performative fields
            cq = data["currentQuote"]
            assert cq is not None
            assert cq["xi"] == pytest.approx(1.5)
            assert cq["quotingMode"] == "theta"

    async def test_get_market_detail_null_performative_fields(
        self, session_factory, seed_quotes
    ) -> None:
        app = _make_app(session_factory)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/markets/TEST-SYM-2")
            assert resp.status_code == 200
            data = resp.json()

            entry = data["quoteHistory"][0]
            assert entry["xi"] is None
            assert entry["theta0"] is None
            assert entry["quotingMode"] is None

            cq = data["currentQuote"]
            assert cq["xi"] is None
            assert cq["quotingMode"] is None
