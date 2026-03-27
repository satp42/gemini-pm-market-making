"""Theta parameter optimizer using Optuna with backtest simulation.

Runs category-specific optimization of (theta0, theta1, theta2) scaling
parameters for the performative quoting model. Each trial backtests a
theta vector against historical price series and evaluates expected CARA
utility.

The optimizer runs as a background task via asyncio.to_thread() to avoid
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import optuna
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import PerformativeSettings
from src.db.models import ThetaParameter
from src.engine.performative import compute_performative_quote

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose default logging (T021)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationProgress:
    """Shared progress state for theta optimization.

    Written by the optimizer thread (via Optuna callback), read by the
    API status endpoint. GIL guarantees atomicity for simple field writes.
    """

    running: bool = False
    current_trial: int = 0
    total_trials: int = 0
    best_value: float | None = None
    current_category: str = ""
    categories_completed: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed: bool = False
    error_message: str = ""


def _run_backtest_simulation(
    price_series: np.ndarray,
    theta0: float,
    theta1: float,
    theta2: float,
    gamma: float,
    k: float,
    sigma_sq: float,
    xi: float,
    dt: float = 1.0,
) -> float:
    """Run a single backtest simulation over a price series.

    At each step:
      1. Compute performative bid/ask with the given theta values.
      2. Determine fills via Poisson: P(fill) = 1 - exp(-A * exp(-k * delta) * dt)
         where A = k (simplification for prediction markets).
      3. Track inventory, cash, and compute terminal PnL.

    Returns the terminal PnL.
    """
    n = len(price_series)
    if n < 2:
        return 0.0

    inventory = 0.0
    cash = 0.0

    for i in range(n - 1):
        mid = price_series[i]
        t_remaining = max((n - i) * dt / n, 0.01)  # normalised time remaining

        quote = compute_performative_quote(
            mid_price=mid,
            inventory=inventory,
            gamma=gamma,
            sigma_sq=sigma_sq,
            t_minus_t=t_remaining,
            k=k,
            xi=xi,
            theta0=theta0,
            theta1=theta1,
            theta2=theta2,
        )

        bid = quote.bid_price
        ask = quote.ask_price

        # Distance from mid-price to bid/ask
        delta_bid = max(mid - bid, 0.0)
        delta_ask = max(ask - mid, 0.0)

        # Fill probability: P(fill) = 1 - exp(-A * exp(-k * delta) * dt)
        # A = k simplification for prediction markets
        A = k
        p_bid_fill = 1.0 - np.exp(-A * np.exp(-k * delta_bid) * dt)
        p_ask_fill = 1.0 - np.exp(-A * np.exp(-k * delta_ask) * dt)

        # Stochastic fills using the deterministic probability
        # (for reproducibility within the objective, we use the probability directly
        # as expected fill to reduce noise)
        # Bid fill: we buy at bid price
        if p_bid_fill > 0:
            cash -= bid * p_bid_fill
            inventory += p_bid_fill

        # Ask fill: we sell at ask price
        if p_ask_fill > 0:
            cash += ask * p_ask_fill
            inventory -= p_ask_fill

    # Terminal PnL: cash + inventory * final mid-price
    terminal_pnl = cash + inventory * price_series[-1]
    return terminal_pnl


async def run_theta_optimization(
    categories: dict[str, list[list[float]]],
    progress: OptimizationProgress,
    session_factory: async_sessionmaker[AsyncSession],
    settings: PerformativeSettings,
    gamma: float,
    k: float,
    sigma_default: float,
) -> None:
    """Run Optuna theta optimization for each category.

    Parameters
    ----------
    categories : dict[str, list[list[float]]]
        Mapping of category name to list of price series (each a list of floats).
    progress : OptimizationProgress
        Shared progress object updated during optimization.
    session_factory : async_sessionmaker
        SQLAlchemy async session factory for persisting results.
    settings : PerformativeSettings
        Configuration with optimization parameters.
    gamma : float
        Risk aversion parameter for CARA utility.
    k : float
        Order arrival intensity parameter.
    sigma_default : float
        Default variance for simulation.
    """
    progress.running = True
    progress.started_at = datetime.now(timezone.utc)
    progress.completed_at = None
    progress.categories_completed = []
    progress.best_value = None

    n_trials = settings.theta_optimization_trials
    n_simulations = settings.theta_optimization_simulations
    sigma_sq = sigma_default
    xi = settings.xi_default

    try:
        for category, price_series_list in categories.items():
            if not price_series_list:
                logger.warning("No price series for category '%s', skipping", category)
                continue

            progress.current_category = category
            progress.current_trial = 0
            progress.total_trials = n_trials

            # Prepare numpy arrays for all price series in this category
            np_series = [
                np.asarray(ps, dtype=np.float64)
                for ps in price_series_list
                if len(ps) >= 2
            ]
            if not np_series:
                logger.warning(
                    "No valid price series (>=2 points) for category '%s'", category
                )
                continue

            # Bind loop variables to avoid closure issues
            _np_series = np_series
            _n_simulations = n_simulations

            def objective(trial: optuna.Trial) -> float:
                theta0 = trial.suggest_float("theta0", 0.5, 2.0)
                theta1 = trial.suggest_float("theta1", 0.5, 2.0)
                theta2 = trial.suggest_float("theta2", 0.5, 2.0)

                # Run simulations across price series
                utilities: list[float] = []
                series_to_use = _np_series[:_n_simulations]

                for series in series_to_use:
                    pnl = _run_backtest_simulation(
                        price_series=series,
                        theta0=theta0,
                        theta1=theta1,
                        theta2=theta2,
                        gamma=gamma,
                        k=k,
                        sigma_sq=sigma_sq,
                        xi=xi,
                    )
                    # CARA utility: -exp(-gamma * terminal_pnl)
                    utility = -np.exp(-gamma * pnl)
                    utilities.append(float(utility))

                # If we have fewer series than n_simulations, repeat with noise
                while len(utilities) < _n_simulations:
                    idx = len(utilities) % len(_np_series)
                    series = _np_series[idx]
                    # Add small noise to create variation
                    rng = np.random.default_rng(seed=trial.number * 1000 + len(utilities))
                    noisy_series = series + rng.normal(0, 0.001, size=len(series))
                    noisy_series = np.clip(noisy_series, 0.01, 0.99)

                    pnl = _run_backtest_simulation(
                        price_series=noisy_series,
                        theta0=theta0,
                        theta1=theta1,
                        theta2=theta2,
                        gamma=gamma,
                        k=k,
                        sigma_sq=sigma_sq,
                        xi=xi,
                    )
                    utility = -np.exp(-gamma * pnl)
                    utilities.append(float(utility))

                return float(np.mean(utilities))

            def progress_callback(
                study: optuna.Study, trial: optuna.trial.FrozenTrial
            ) -> None:
                progress.current_trial = trial.number + 1
                progress.best_value = study.best_value

            study = optuna.create_study(direction="maximize")

            # Offload synchronous study.optimize to a thread
            await asyncio.to_thread(
                study.optimize,
                objective,
                n_trials=n_trials,
                callbacks=[progress_callback],
            )

            # Persist best theta for this category
            best_params = study.best_params
            best_theta0 = best_params["theta0"]
            best_theta1 = best_params["theta1"]
            best_theta2 = best_params["theta2"]

            await _upsert_theta(
                session_factory=session_factory,
                category=category,
                theta0=best_theta0,
                theta1=best_theta1,
                theta2=best_theta2,
                xi_value=xi,
                objective_value=study.best_value,
                num_trials=n_trials,
            )

            progress.categories_completed.append(category)
            logger.info(
                "Theta optimization complete for '%s': theta=(%.4f, %.4f, %.4f), "
                "objective=%.6f",
                category,
                best_theta0,
                best_theta1,
                best_theta2,
                study.best_value,
            )

    except Exception as e:
        logger.exception("Theta optimization failed")
        progress.failed = True
        progress.error_message = str(e)
    finally:
        progress.running = False
        progress.completed_at = datetime.now(timezone.utc)


async def _upsert_theta(
    session_factory: async_sessionmaker[AsyncSession],
    category: str,
    theta0: float,
    theta1: float,
    theta2: float,
    xi_value: float,
    objective_value: float,
    num_trials: int,
) -> None:
    """Upsert a ThetaParameter row for the given category.

    Uses SELECT + INSERT/UPDATE pattern for cross-dialect compatibility
    (works with both PostgreSQL and SQLite for testing).
    """
    now = datetime.now(timezone.utc)

    async with session_factory() as session:
        try:
            # Check if row exists for this category
            result = await session.execute(
                select(ThetaParameter).where(ThetaParameter.category == category)
            )
            existing = result.scalar_one_or_none()

            if existing is not None:
                existing.theta0 = theta0
                existing.theta1 = theta1
                existing.theta2 = theta2
                existing.xi_value = xi_value
                existing.objective_value = objective_value
                existing.num_trials = num_trials
                existing.optimized_at = now
                existing.updated_at = now
            else:
                row = ThetaParameter(
                    category=category,
                    theta0=theta0,
                    theta1=theta1,
                    theta2=theta2,
                    xi_value=xi_value,
                    objective_value=objective_value,
                    num_trials=num_trials,
                    optimized_at=now,
                )
                session.add(row)

            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Failed to upsert theta for category '%s'", category)
            raise
