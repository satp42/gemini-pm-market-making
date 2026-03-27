"""Performative market-making quoting engine.

Extends the Avellaneda-Stoikov model to account for the market maker's own
price impact through the xi (feedback strength) parameter and optional
theta scaling parameters.

Formulas (from arXiv:2508.04344):
    delta_epsilon = (1 - exp(-xi*T) - xi*T*exp(-xi*T)) / xi^2
    r_perf = theta0 * s * exp(-xi*T)
             - gamma * sigma^2 * (theta1 * q_ref * delta_epsilon
                                  - theta2 * q * (exp(-2*xi*T) - 1) / (2*xi))
    spread_perf = 2/gamma * ln(1 + gamma/k)
                  - gamma * sigma^2 * (exp(-2*xi*T) - 1) / (2*xi)

When xi -> 0 the formulas degenerate to standard A&S via Taylor-series limits.
"""

from __future__ import annotations

import logging
import math

from src.engine.quoting import Quote, compute_quote

logger = logging.getLogger(__name__)

# Threshold below which Taylor-series fallbacks are used for numerical stability.
# At xi=1e-6 with T=24, direct computation still has ~11 digits of precision,
# but the guard protects against catastrophic cancellation at smaller values.
XI_EPSILON = 1e-6


def delta_epsilon(xi: float, T: float, _exp_neg_xi_T: float | None = None) -> float:
    """Compute (1 - exp(-xi*T) - xi*T*exp(-xi*T)) / xi^2.

    Taylor fallback: T^2 / 2 when abs(xi) < XI_EPSILON.

    Parameters
    ----------
    _exp_neg_xi_T : float | None
        Pre-computed exp(-xi*T) to avoid redundant exp calls. When *None*
        the value is computed internally (standalone / test usage).
    """
    if abs(xi) < XI_EPSILON:
        return T * T / 2.0
    u = xi * T
    exp_neg_u = _exp_neg_xi_T if _exp_neg_xi_T is not None else math.exp(-u)
    return (1.0 - exp_neg_u - u * exp_neg_u) / (xi * xi)


def inv_correction(xi: float, T: float, _exp_neg_2xi_T: float | None = None) -> float:
    """Compute (exp(-2*xi*T) - 1) / (2*xi).

    Taylor fallback: -T when abs(xi) < XI_EPSILON.

    Parameters
    ----------
    _exp_neg_2xi_T : float | None
        Pre-computed exp(-2*xi*T) to avoid redundant exp calls. When *None*
        the value is computed internally (standalone / test usage).
    """
    if abs(xi) < XI_EPSILON:
        return -T
    exp_neg_2u = _exp_neg_2xi_T if _exp_neg_2xi_T is not None else math.exp(-2.0 * xi * T)
    return (exp_neg_2u - 1.0) / (2.0 * xi)


def compute_performative_quote(
    mid_price: float,
    inventory: float,
    gamma: float,
    sigma_sq: float,
    t_minus_t: float,
    k: float,
    xi: float,
    theta0: float = 1.0,
    theta1: float = 1.0,
    theta2: float = 1.0,
    q_ref: float = 0.0,
    max_spread: float = 0.0,
    best_bid: float = 0.0,
    best_ask: float = 0.0,
    quoting_mode: str = "performative",
) -> Quote:
    """Compute performative reservation price and optimal spread.

    Falls back to standard A&S (``compute_quote``) if any intermediate result
    is NaN or Inf.

    Parameters
    ----------
    mid_price : float
        Current mid-price of the market.
    inventory : float
        Net inventory (positive = long).
    gamma : float
        Risk-aversion parameter.
    sigma_sq : float
        Estimated price variance.
    t_minus_t : float
        Time remaining (T - t), normalised.
    k : float
        Order-arrival intensity parameter.
    xi : float
        Performative feedback strength.
    theta0, theta1, theta2 : float
        Category-specific scaling parameters (default 1.0 each).
    q_ref : float
        Reference inventory (default 0.0).
    max_spread : float
        Maximum allowed spread (0 = no cap).
    best_bid, best_ask : float
        Current best bid/ask from the order book (0 = unused).
    """
    # --- Performative reservation price ---
    T = t_minus_t
    exp_neg_xi_T = math.exp(-xi * T)
    exp_neg_2xi_T = exp_neg_xi_T * exp_neg_xi_T
    d_eps = delta_epsilon(xi, T, _exp_neg_xi_T=exp_neg_xi_T)
    inv_corr = inv_correction(xi, T, _exp_neg_2xi_T=exp_neg_2xi_T)

    reservation_price = (
        theta0 * mid_price * exp_neg_xi_T
        - gamma * sigma_sq * (theta1 * q_ref * d_eps - theta2 * inventory * inv_corr)
    )

    # --- Performative spread ---
    spread = (
        (2.0 / gamma) * math.log(1.0 + gamma / k)
        - gamma * sigma_sq * inv_corr
    )

    # --- NaN / Inf guard: fall back to A&S ---
    if not (math.isfinite(reservation_price) and math.isfinite(spread)):
        logger.warning(
            "Non-finite performative result (r=%.6g, spread=%.6g), "
            "falling back to A&S",
            reservation_price,
            spread,
        )
        return compute_quote(
            mid_price=mid_price,
            inventory=inventory,
            gamma=gamma,
            sigma_sq=sigma_sq,
            t_minus_t=t_minus_t,
            k=k,
        )

    # --- Max spread cap ---
    if max_spread > 0 and spread > max_spread:
        spread = max_spread

    # --- Book spread cap (same logic as existing A&S path) ---
    if best_bid > 0 and best_ask > 0:
        book_spread = best_ask - best_bid
        if book_spread > 0 and spread > book_spread:
            spread = book_spread

    # --- Derive bid / ask ---
    bid = reservation_price - spread / 2.0
    ask = reservation_price + spread / 2.0

    # --- Clamp to prediction-market bounds ---
    bid = max(0.01, min(0.99, bid))
    ask = max(0.01, min(0.99, ask))

    return Quote(
        bid_price=bid,
        ask_price=ask,
        reservation_price=reservation_price,
        spread=spread,
        mid_price=mid_price,
        inventory=inventory,
        sigma_sq=sigma_sq,
        gamma=gamma,
        t_minus_t=t_minus_t,
        k=k,
        xi=xi,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        quoting_mode=quoting_mode,
    )
