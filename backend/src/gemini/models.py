"""Pydantic models for Gemini Prediction Markets API responses."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class OrderBookEntry(BaseModel):
    """A single price level in an order book."""

    model_config = ConfigDict(populate_by_name=True)

    price: Decimal
    amount: Decimal
    timestamp: str = ""


class OrderBook(BaseModel):
    """Order book with bid and ask sides."""

    model_config = ConfigDict(populate_by_name=True)

    bids: list[OrderBookEntry] = Field(default_factory=list)
    asks: list[OrderBookEntry] = Field(default_factory=list)


class Trade(BaseModel):
    """A single executed trade."""

    model_config = ConfigDict(populate_by_name=True)

    timestamp: int
    tid: int = 0
    price: Decimal
    amount: Decimal
    type: str = ""


class Ticker(BaseModel):
    """Ticker data for a symbol."""

    model_config = ConfigDict(populate_by_name=True)

    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None
    volume: Decimal | None = None


class ContractPrices(BaseModel):
    """Price data nested inside a contract."""

    model_config = ConfigDict(populate_by_name=True)

    best_bid: Decimal | None = Field(default=None, alias="bestBid")
    best_ask: Decimal | None = Field(default=None, alias="bestAsk")
    last_trade_price: Decimal | None = Field(default=None, alias="lastTradePrice")


class Contract(BaseModel):
    """A contract within a prediction market event."""

    model_config = ConfigDict(populate_by_name=True)

    instrument_symbol: str = Field(default="", alias="instrumentSymbol")
    outcome: str = ""
    prices: ContractPrices | None = None
    open_interest: Decimal | None = Field(default=None, alias="openInterest")

    @property
    def best_bid(self) -> Decimal | None:
        return self.prices.best_bid if self.prices else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.prices.best_ask if self.prices else None

    @property
    def last_trade_price(self) -> Decimal | None:
        return self.prices.last_trade_price if self.prices else None


class Event(BaseModel):
    """A prediction market event containing one or more contracts."""

    model_config = ConfigDict(populate_by_name=True)

    event_ticker: str = Field(default="", alias="ticker")
    title: str = ""
    status: str = ""
    expiration_date: str | None = Field(default=None, alias="expiryDate")
    contracts: list[Contract] = Field(default_factory=list)
    category: str = ""
    mutually_exclusive: bool = Field(default=False, alias="mutuallyExclusive")


class EventsResponse(BaseModel):
    """Paginated response from the events endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    events: list[Event] = Field(default_factory=list, alias="data")
    pagination: dict = Field(default_factory=dict)


class Order(BaseModel):
    """An order on Gemini Prediction Markets."""

    model_config = ConfigDict(populate_by_name=True)

    order_id: int = Field(default=0, alias="orderId")
    status: str = ""
    symbol: str = ""
    side: str = ""
    outcome: str = ""
    quantity: str = "0"
    filled_quantity: str = Field(default="0", alias="filledQuantity")
    remaining_quantity: str = Field(default="0", alias="remainingQuantity")
    price: Decimal = Decimal("0")
    type: str = ""
    time_in_force: str = Field(default="", alias="timeInForce")
    timestamp: int = 0


class Position(BaseModel):
    """A held position on a prediction market contract."""

    model_config = ConfigDict(populate_by_name=True)

    symbol: str = ""
    outcome: str = ""
    quantity: Decimal = Decimal("0")
    avg_entry_price: Decimal = Field(default=Decimal("0"), alias="avgEntryPrice")
    unrealized_pnl: Decimal = Field(default=Decimal("0"), alias="unrealizedPnl")
