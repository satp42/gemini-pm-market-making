"""SQLAlchemy ORM models for the market-making bot persistence layer."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, Numeric, String, func
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
    gemini_order_id: Mapped[int] = mapped_column(Integer, nullable=False)
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
