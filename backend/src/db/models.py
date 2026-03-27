"""SQLAlchemy ORM models for the market-making bot persistence layer."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class Quote(Base):
    """Record of every quote computed per symbol per bot cycle."""

    __tablename__ = "quotes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    mid_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    reservation_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    bid_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    ask_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    spread: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    inventory: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    sigma_sq: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    gamma: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    t_minus_t: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    # Performative market-making fields (nullable for backward compat)
    xi: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    theta0: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    theta1: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    theta2: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    quoting_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)

    __table_args__ = (
        Index("ix_quotes_timestamp", "timestamp"),
        Index("ix_quotes_symbol", "symbol"),
        Index("ix_quotes_symbol_timestamp", "symbol", "timestamp"),
    )


class OrderRecord(Base):
    """Record of every order placed or cancelled through the bot."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    gemini_order_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str] = mapped_column(String(8), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    fill_price: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)

    __table_args__ = (
        Index("ix_orders_timestamp", "timestamp"),
        Index("ix_orders_symbol", "symbol"),
        Index("ix_orders_gemini_order_id", "gemini_order_id"),
    )


class PositionSnapshot(Base):
    """Inventory snapshot per symbol per bot cycle."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    yes_quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    no_quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    net_inventory: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)

    __table_args__ = (
        Index("ix_positions_timestamp", "timestamp"),
        Index("ix_positions_symbol", "symbol"),
        Index("ix_positions_symbol_timestamp", "symbol", "timestamp"),
    )


class PnlSnapshot(Base):
    """Aggregated P&L snapshot per bot cycle."""

    __tablename__ = "pnl_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    total_realized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    total_unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    total_exposure: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    num_active_markets: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_pnl_snapshots_timestamp", "timestamp"),
    )


class ThetaParameter(Base):
    """Optimized theta parameters per market category."""

    __tablename__ = "theta_parameters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    theta0: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    theta1: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    theta2: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    xi_value: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    objective_value: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    num_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    optimized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class XiEstimateRecord(Base):
    """Record of xi estimation per symbol per bot cycle (observability)."""

    __tablename__ = "xi_estimates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    xi: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    r_squared: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    num_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    used_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_xi_estimates_timestamp", "timestamp"),
        Index("ix_xi_estimates_symbol", "symbol"),
    )
