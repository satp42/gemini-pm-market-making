"""Xi (performative feedback strength) estimator using OLS regression.

Fits an Ornstein-Uhlenbeck process to recent trade data to estimate xi,
the feedback parameter that quantifies how strongly the market maker's
actions pull the mid-price.

Pure computation -- no side effects, no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class XiEstimate:
    """Result of xi estimation for a single symbol."""

    xi: float
    r_squared: float | None
    num_trades: int
    used_default: bool


def estimate_xi(
    trade_prices: list[float],
    trade_timestamps: list[float],
    xi_default: float,
    xi_min_trades: int,
    xi_clamp_min: float,
    xi_clamp_max: float,
    r_squared_threshold: float,
) -> XiEstimate:
    """Estimate performative feedback strength xi from trade data via OLS.

    Fits the regression ``delta_s ~ beta * s_n + alpha`` using
    ``np.linalg.lstsq`` and extracts ``xi = -beta / dt``.

    Parameters
    ----------
    trade_prices : list[float]
        Recent trade prices (chronological order).
    trade_timestamps : list[float]
        Unix-epoch timestamps corresponding to *trade_prices*.
    xi_default : float
        Fallback xi when data is insufficient or regression is poor.
    xi_min_trades : int
        Minimum number of trades required to attempt regression.
    xi_clamp_min, xi_clamp_max : float
        Range to clamp the estimated xi into.
    r_squared_threshold : float
        Minimum r-squared for the regression to be considered reliable.

    Returns
    -------
    XiEstimate
        Estimated xi value with quality metadata.
    """
    n = len(trade_prices)

    # --- Edge case: insufficient trades ---
    if n < xi_min_trades:
        return XiEstimate(
            xi=xi_default,
            r_squared=None,
            num_trades=n,
            used_default=True,
        )

    # --- Build regression arrays ---
    prices = np.asarray(trade_prices, dtype=np.float64)
    timestamps = np.asarray(trade_timestamps, dtype=np.float64)

    # delta_s = s_{n+1} - s_n
    delta_s = np.diff(prices)  # length n-1
    s_n = prices[:-1]  # length n-1

    # Average time step
    dt_arr = np.diff(timestamps)
    dt = float(np.mean(dt_arr)) if len(dt_arr) > 0 else 1.0
    if dt <= 0:
        dt = 1.0

    # --- OLS: delta_s ~ beta * s_n + alpha ---
    # Design matrix: [s_n, ones]
    m = len(s_n)
    A = np.column_stack([s_n, np.ones(m)])

    # np.linalg.lstsq returns (solution, residuals, rank, singular_values)
    result = np.linalg.lstsq(A, delta_s, rcond=None)
    coeffs = result[0]  # [beta, alpha]
    residuals = result[1]

    beta = float(coeffs[0])

    # --- Compute r-squared ---
    if len(residuals) == 0:
        # Rank-deficient: no meaningful regression (e.g., flat series)
        r_squared = 0.0
    else:
        ss_res = float(residuals[0])
        ss_tot = float(np.sum((delta_s - np.mean(delta_s)) ** 2))
        if ss_tot < 1e-30:
            r_squared = 0.0
        else:
            r_squared = 1.0 - ss_res / ss_tot
            # Clamp r_squared to [0, 1] to handle numerical noise
            r_squared = max(0.0, min(1.0, r_squared))

    # --- Quality gate: r-squared threshold ---
    if r_squared < r_squared_threshold:
        logger.warning(
            "Xi regression r_squared=%.4f below threshold %.2f, using default xi=%.2f",
            r_squared,
            r_squared_threshold,
            xi_default,
        )
        return XiEstimate(
            xi=xi_default,
            r_squared=r_squared,
            num_trades=n,
            used_default=True,
        )

    # --- Extract xi = -beta / dt ---
    xi_raw = -beta / dt

    # --- Clamp to valid range ---
    xi_clamped = max(xi_clamp_min, min(xi_clamp_max, xi_raw))

    return XiEstimate(
        xi=xi_clamped,
        r_squared=r_squared,
        num_trades=n,
        used_default=False,
    )
