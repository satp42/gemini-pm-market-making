# API Contract: Performative Market Making Engine

## Overview

This document specifies the API changes for the performative market making feature. All changes are additive -- existing endpoints retain their current fields and behavior. New fields are nullable/optional for backward compatibility.

---

## Extended Endpoints

### POST /api/config (existing, extended)

Accepts new performative configuration fields alongside existing A&S, bot, and risk parameters.

**Request Body** (all fields optional, non-null values are merged into overrides):

```json
{
  "gamma": 0.5,
  "k": 10.0,
  "sigma_default": 0.01,
  "variance_window": 100,
  "bot_cycle_seconds": 10,
  "scanner_cycle_seconds": 300,
  "min_spread": 0.03,
  "max_inventory_per_symbol": 200,
  "max_total_exposure": 1000,
  "risk_widen_threshold": 0.8,
  "quoting_mode": "theta",
  "xi_default": 0.5,
  "xi_min_trades": 15,
  "xi_clamp_min": 0.01,
  "xi_clamp_max": 20.0,
  "q_ref": 0.0
}
```

**New Request Fields**:

| Field | Type | Valid Values | Description |
|-------|------|-------------|-------------|
| `quoting_mode` | string or null | "as", "performative", "theta" | Active quoting model. Takes effect next bot cycle. |
| `xi_default` | float or null | [0.01, 20.0] | Fallback xi when estimation fails |
| `xi_min_trades` | int or null | [5, 100] | Minimum trades for xi estimation |
| `xi_clamp_min` | float or null | > 0 | Lower bound for estimated xi |
| `xi_clamp_max` | float or null | > xi_clamp_min | Upper bound for estimated xi |
| `q_ref` | float or null | any float | Reference inventory for performative formula |

**Response** (200 OK):

```json
{
  "overrides": {
    "quoting_mode": "theta",
    "gamma": 0.5
  },
  "effective": {
    "gamma": 0.5,
    "k": 10.0,
    "sigma_default": 0.01,
    "variance_window": 100,
    "bot_cycle_seconds": 10,
    "scanner_cycle_seconds": 300,
    "min_spread": 0.03,
    "max_inventory_per_symbol": 200,
    "max_total_exposure": 1000,
    "risk_widen_threshold": 0.8,
    "quoting_mode": "theta",
    "xi_default": 0.5,
    "xi_min_trades": 15,
    "xi_clamp_min": 0.01,
    "xi_clamp_max": 20.0,
    "q_ref": 0.0
  }
}
```

**Validation**:
- `quoting_mode` must be one of "as", "performative", "theta". Returns 422 otherwise.
- `xi_clamp_min` must be less than `xi_clamp_max` when both are provided. Returns 422 otherwise.
- All numerical fields must be finite and non-NaN.

---

### GET /api/markets (existing, extended)

Each market summary gains performative fields.

**Response** (200 OK):

```json
{
  "markets": [
    {
      "symbol": "GEMI-BTC100K-26MAR26",
      "eventTitle": "Will BTC exceed $100K by March 26?",
      "midPrice": 0.62,
      "reservationPrice": 0.58,
      "bidPrice": 0.55,
      "askPrice": 0.65,
      "spread": 0.10,
      "inventory": 3.0,
      "sigmaSquared": 0.008,
      "gamma": 0.1,
      "timeRemaining": 0.5,
      "xi": 1.8,
      "theta0": 1.15,
      "theta1": 0.92,
      "theta2": 1.08,
      "quotingMode": "theta"
    }
  ]
}
```

**New Response Fields** (per market):

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `xi` | number | yes | Estimated feedback strength. Null for A&S mode or no recent quote. |
| `theta0` | number | yes | Theta-0 parameter. Null for A&S mode. |
| `theta1` | number | yes | Theta-1 parameter. Null for A&S mode. |
| `theta2` | number | yes | Theta-2 parameter. Null for A&S mode. |
| `quotingMode` | string | yes | "as", "performative", or "theta". Null for legacy quotes. |

---

### GET /api/markets/{symbol} (existing, extended)

**Extended fields in `quoteHistory` entries**:

```json
{
  "symbol": "GEMI-BTC100K-26MAR26",
  "quoteHistory": [
    {
      "id": 42,
      "timestamp": "2026-03-26T10:00:00Z",
      "midPrice": 0.62,
      "reservationPrice": 0.58,
      "bidPrice": 0.55,
      "askPrice": 0.65,
      "spread": 0.10,
      "inventory": 3.0,
      "sigmaSquared": 0.008,
      "gamma": 0.1,
      "timeRemaining": 0.5,
      "xi": 1.8,
      "theta0": 1.15,
      "theta1": 0.92,
      "theta2": 1.08,
      "quotingMode": "theta"
    }
  ],
  "recentTrades": [],
  "currentQuote": null
}
```

Same new fields as GET /api/markets, applied to `QuoteHistoryEntry`.

---

### WebSocket Tick Message (existing, extended)

**Extended market data objects**:

```json
{
  "type": "tick",
  "data": {
    "status": {
      "running": true,
      "uptime": 3600.0,
      "activeMarkets": 5,
      "environment": "sandbox"
    },
    "markets": [
      {
        "symbol": "GEMI-BTC100K-26MAR26",
        "eventTitle": "Will BTC exceed $100K?",
        "midPrice": 0.62,
        "reservationPrice": 0.58,
        "bidPrice": 0.55,
        "askPrice": 0.65,
        "spread": 0.10,
        "inventory": 3.0,
        "sigmaSquared": 0.008,
        "gamma": 0.1,
        "timeRemaining": 0,
        "xi": 1.8,
        "theta0": 1.15,
        "theta1": 0.92,
        "theta2": 1.08,
        "quotingMode": "theta"
      }
    ],
    "positions": [],
    "pnl": {}
  }
}
```

Same new fields as REST endpoints. Fields are null when bot is in A&S mode.

---

## New Endpoints

### POST /api/optimize/theta

Trigger theta optimization for all (or a specific) prediction market category.

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `category` | string | no | (all) | Specific category to optimize. If omitted, all categories are optimized. |

**Request Body**: None.

**Response** (202 Accepted):

```json
{
  "status": "started",
  "categories": ["Crypto", "Sports", "Politics", "Economics"]
}
```

**Error Responses**:

| Status | Body | Condition |
|--------|------|-----------|
| 409 Conflict | `{"detail": "Optimization is already running"}` | `optimization_progress.running == True` |
| 503 Service Unavailable | `{"detail": "Bot loop not initialized"}` | `app.state.bot_loop is None` |

**Behavior**:
1. Checks if optimization is already running (409 if yes).
2. Derives categories from `bot_loop._symbol_categories` (distinct non-empty values).
3. If `category` query param is provided, filters to that single category.
4. For each symbol in target categories, fetches trade data via `GeminiClient.get_trades()`.
5. Initializes `app.state.optimization_progress` with `running=True`.
6. Launches `asyncio.create_task(run_theta_optimization(...))`.
7. Returns immediately with 202.

---

### GET /api/optimize/theta/status

Get current and last theta optimization status.

**Request**: No parameters.

**Response** (200 OK) -- Idle state:

```json
{
  "running": false,
  "currentTrial": 0,
  "totalTrials": 100,
  "bestValue": null,
  "currentCategory": "",
  "categoriesCompleted": ["Crypto", "Sports"],
  "startedAt": "2026-03-25T14:00:00Z",
  "completedAt": "2026-03-25T14:30:00Z"
}
```

**Response** (200 OK) -- Running state:

```json
{
  "running": true,
  "currentTrial": 47,
  "totalTrials": 100,
  "bestValue": -0.0023,
  "currentCategory": "Crypto",
  "categoriesCompleted": [],
  "startedAt": "2026-03-26T10:00:00Z",
  "completedAt": null
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `running` | boolean | Whether optimization is currently in progress |
| `currentTrial` | integer | Current trial number (1-indexed, 0 when idle) |
| `totalTrials` | integer | Total configured trials per category |
| `bestValue` | number or null | Best objective value found so far in current category |
| `currentCategory` | string | Category currently being optimized ("" when idle) |
| `categoriesCompleted` | string[] | Categories that have finished optimization |
| `startedAt` | string (ISO 8601) or null | When the current/last optimization started |
| `completedAt` | string (ISO 8601) or null | When the last optimization completed (null if running) |

---

## Pydantic Schema Changes

### ConfigUpdateRequest (extended)

**File**: `backend/src/api/router.py`, line 113

```python
class ConfigUpdateRequest(BaseModel):
    # Existing fields
    gamma: float | None = None
    k: float | None = None
    sigma_default: float | None = None
    variance_window: int | None = None
    bot_cycle_seconds: int | None = None
    scanner_cycle_seconds: int | None = None
    min_spread: float | None = None
    max_inventory_per_symbol: int | None = None
    max_total_exposure: int | None = None
    risk_widen_threshold: float | None = None
    # New performative fields
    quoting_mode: str | None = None
    xi_default: float | None = None
    xi_min_trades: int | None = None
    xi_clamp_min: float | None = None
    xi_clamp_max: float | None = None
    q_ref: float | None = None
```

### MarketSummary (extended)

**File**: `backend/src/api/router.py`, line 37

```python
class MarketSummary(BaseModel):
    # Existing fields
    symbol: str
    eventTitle: str = ""
    midPrice: float
    reservationPrice: float
    bidPrice: float
    askPrice: float
    spread: float
    inventory: float
    sigmaSquared: float
    gamma: float
    timeRemaining: float
    # New performative fields
    xi: float | None = None
    theta0: float | None = None
    theta1: float | None = None
    theta2: float | None = None
    quotingMode: str | None = None
```

### QuoteHistoryEntry (extended)

**File**: `backend/src/api/router.py`, line 51

Same 5 new fields as MarketSummary, all `float | None = None` / `str | None = None`.

---

## Frontend TypeScript Types

### MarketData (extended)

**File**: `frontend/src/lib/types.ts`, line 3

```typescript
export interface MarketData {
  symbol: string;
  eventTitle?: string;
  midPrice: number;
  reservationPrice: number;
  bidPrice: number;
  askPrice: number;
  spread: number;
  inventory: number;
  sigmaSquared: number;
  gamma: number;
  timeRemaining: number;
  // Performative fields
  xi?: number;
  theta0?: number;
  theta1?: number;
  theta2?: number;
  quotingMode?: string;
}
```

### ActivityLogEntry (extended)

**File**: `frontend/src/lib/types.ts`, line 47

Add `"mode_switch"` to the `type` union:

```typescript
export interface ActivityLogEntry {
  timestamp: string;
  type:
    | "order_placed"
    | "order_filled"
    | "order_cancelled"
    | "risk_alert"
    | "mode_switch"
    | "info";
  message: string;
  symbol?: string;
}
```

---

## Backward Compatibility

| Change | Impact | Mitigation |
|--------|--------|------------|
| New fields on MarketSummary | None -- fields are optional with defaults | Existing frontend ignores unknown fields |
| New fields on ConfigUpdateRequest | None -- all new fields are `| None = None` | Existing POST /config payloads work unchanged |
| New fields in WebSocket tick | None -- frontend uses optional chaining | Existing dashboard renders without performative data |
| New endpoints | None -- additive | Existing API consumers unaffected |
| New DB columns on quotes | None -- nullable | Existing queries return NULL for new columns |
