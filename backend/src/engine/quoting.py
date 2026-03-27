"""Avellaneda-Stoikov quoting engine -- pure computation, no side effects."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Quote:
    """Result of the Avellaneda-Stoikov model for a single symbol."""

    bid_price: float
    ask_price: float
    reservation_price: float
    spread: float
    mid_price: float
    inventory: float
    sigma_sq: float
    gamma: float
    t_minus_t: float
    k: float


def compute_quote(
    mid_price: float,
    inventory: float,
    gamma: float,
    sigma_sq: float,
    t_minus_t: float,
    k: float,
) -> Quote:
    """Compute Avellaneda-Stoikov reservation price and optimal spread.

    Formulas
    --------
    r = s - q * gamma * sigma^2 * (T - t)
    delta* = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
    bid = r - delta*/2
    ask = r + delta*/2

    All prices are clamped to the prediction-market range [0.01, 0.99].
    """
    reservation_price = mid_price - inventory * gamma * sigma_sq * t_minus_t
    spread = gamma * sigma_sq * t_minus_t + (2.0 / gamma) * math.log(1.0 + gamma / k)

    bid = reservation_price - spread / 2.0
    ask = reservation_price + spread / 2.0

    # Clamp to prediction market bounds
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
    )


def estimate_variance(trade_prices: list[float], default: float = 0.01) -> float:
    """Estimate price variance from recent trades.

    Falls back to *default* if fewer than 10 trades are available.
    Returns at least 0.0001 to prevent zero-variance edge cases.
    """
    if len(trade_prices) < 10:
        return default

    returns = [trade_prices[i] - trade_prices[i - 1] for i in range(1, len(trade_prices))]
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    return max(variance, 0.0001)
