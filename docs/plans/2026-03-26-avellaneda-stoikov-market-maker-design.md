# Avellaneda-Stoikov Market Maker for Gemini Prediction Markets

## Overview

A market-making bot that uses the Avellaneda-Stoikov model to provide liquidity on Gemini Prediction Markets. The bot adjusts its mid-price quote based on inventory risk, automatically rebalancing exposure while capturing the bid-ask spread.

The system consists of a Python (FastAPI) backend running the bot as an async background task, a PostgreSQL database for persistence, and a Next.js dashboard for real-time monitoring.

## Core Model

### Reservation Price

The mid-price quote is adjusted by inventory risk:

```
r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
```

- `s` = current mid-price (from order book)
- `q` = net inventory (positive = long YES, negative = long NO)
- `gamma` = risk aversion parameter
- `sigma^2` = variance of recent trade prices
- `T - t` = time remaining until event expiry (normalized to days)

### Optimal Spread

```
delta* = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
```

- `k` = order arrival intensity parameter

### Bid/Ask Placement

```
p_bid = r - delta* / 2
p_ask = r + delta* / 2
```

Both clamped to [0.01, 0.99] for prediction market bounds.

## Tech Stack

- **Backend:** Python 3.12+, FastAPI, uvicorn, httpx, SQLAlchemy (async), asyncpg
- **Frontend:** Next.js 15, TypeScript, Tailwind CSS, Recharts
- **Database:** PostgreSQL
- **Package Management:** uv (backend), npm (frontend)
- **Environment:** Sandbox by default (`GEMINI_ENV=sandbox`), flag to switch to live

## Project Structure

```
gemini-pm-market-making/
├── backend/
│   ├── pyproject.toml
│   ├── .env.example
│   ├── src/
│   │   ├── main.py             # FastAPI app + bot lifecycle
│   │   ├── config.py           # Env var loading via pydantic-settings
│   │   ├── gemini/
│   │   │   ├── client.py       # Authenticated REST client (httpx)
│   │   │   ├── auth.py         # HMAC-SHA384 signature generation
│   │   │   └── models.py       # Pydantic models for API responses
│   │   ├── engine/
│   │   │   ├── scanner.py      # Market Scanner
│   │   │   ├── book.py         # Order Book Monitor
│   │   │   ├── quoting.py      # Avellaneda-Stoikov core
│   │   │   ├── orders.py       # Order Manager
│   │   │   ├── positions.py    # Position Tracker
│   │   │   ├── risk.py         # Risk Manager
│   │   │   └── loop.py         # Main bot cycle orchestrator
│   │   ├── api/
│   │   │   ├── router.py       # REST endpoints for dashboard
│   │   │   └── ws.py           # WebSocket streaming to frontend
│   │   └── db/
│   │       ├── database.py     # SQLAlchemy async engine
│   │       ├── models.py       # ORM models
│   │       └── migrations/     # Alembic migrations
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── app/                # Next.js App Router
│   │   ├── components/         # Dashboard widgets
│   │   ├── hooks/              # useWebSocket, usePositions, etc.
│   │   └── lib/                # API client, types
│   └── tailwind.config.ts
└── docker-compose.yml          # PostgreSQL
```

## Gemini API Client

### Authentication

Every authenticated request requires three headers:

1. Build JSON payload with `nonce` (Unix ms timestamp) and `request` path
2. Base64-encode the payload
3. HMAC-SHA384 sign with API secret
4. Send headers: `X-GEMINI-APIKEY`, `X-GEMINI-PAYLOAD`, `X-GEMINI-SIGNATURE`

### Base URLs

- Sandbox: `https://api.sandbox.gemini.com`
- Production: `https://api.gemini.com`

Toggled by `GEMINI_ENV` environment variable.

### Endpoints Used

**Public (no auth):**
- `GET /v1/prediction-markets/events?status[]=active` — list active events
- `GET /v1/prediction-markets/events/newly-listed` — newly listed events
- `GET /v1/book/{instrumentSymbol}` — order book depth
- `GET /v1/pubticker/{instrumentSymbol}` — ticker data
- `GET /v1/trades/{instrumentSymbol}` — recent trades

**Authenticated:**
- `POST /v1/prediction-markets/order` — place order (limit only, side: buy/sell, outcome: yes/no, timeInForce: good-til-cancel)
- `POST /v1/prediction-markets/order/cancel` — cancel order by ID
- `POST /v1/prediction-markets/positions` — current positions
- `POST /v1/prediction-markets/orders/active` — outstanding orders

### Client Design

Single `GeminiClient` class using `httpx.AsyncClient`. Built-in retry with exponential backoff for transient failures. All responses parsed into Pydantic models with `Decimal` price fields for precision.

## Quoting Engine

### Inputs Per Tick

- `s` — mid-price from `(best_bid + best_ask) / 2`
- `q` — net inventory from position tracker
- `gamma` — from `AS_GAMMA` env var (default 0.1)
- `sigma_sq` — rolling variance over last N trades (default N=100)
- `T_minus_t` — time to expiry normalized to days
- `k` — from `AS_K` env var (default 1.5)

### Variance Estimation

Rolling window over the last 100 trades. Compute price deltas, then variance. If fewer than 10 trades exist, fallback to `AS_SIGMA_DEFAULT` (0.01).

### Edge Cases

- `T_minus_t < 0.01` (~15 min to expiry): risk manager takes over, widens or stops
- `bid >= ask` after clamping: skip quoting for that tick
- Empty order book side: use last trade price as `s`

## Bot Loop

Runs as an async background task, every `BOT_CYCLE_SECONDS` (default 10):

1. **Scanner** (every `SCANNER_CYCLE_SECONDS`, default 300): Fetch active events, filter by spread >= `MIN_SPREAD`, time to expiry > `MIN_TIME_TO_EXPIRY_HOURS`, not in `EXCLUDED_SYMBOLS`
2. **Book Monitor**: For each active symbol, fetch order book and recent trades, compute mid-price and variance
3. **Position Tracker**: Fetch positions and active orders, compute `{symbol: net_inventory}`
4. **Risk Check**: Validate `|q| < MAX_INVENTORY_PER_SYMBOL` and total exposure < `MAX_TOTAL_EXPOSURE`. Breach = cancel all orders for that symbol, skip quoting, log alert
5. **Quote & Execute**: Cancel stale orders, compute new quotes via A-S model, place new bid and ask
6. **Persist & Broadcast**: Write snapshot to PostgreSQL, push over WebSocket to dashboard

Each step isolated per symbol — one symbol's failure doesn't block others.

## Database Schema

### `quotes`
Every quote computed, one row per symbol per cycle:
- `id`, `timestamp`, `symbol`, `mid_price`, `reservation_price`, `bid_price`, `ask_price`, `spread`, `inventory`, `sigma_sq`, `gamma`, `t_minus_t`

### `orders`
Every order placed or cancelled:
- `id`, `timestamp`, `symbol`, `gemini_order_id`, `side`, `outcome`, `price`, `quantity`, `status` (placed/filled/cancelled), `fill_price`

### `positions`
Inventory snapshot per symbol per cycle:
- `id`, `timestamp`, `symbol`, `yes_quantity`, `no_quantity`, `net_inventory`, `unrealized_pnl`

### `pnl_snapshots`
Aggregated P&L per cycle:
- `id`, `timestamp`, `total_realized_pnl`, `total_unrealized_pnl`, `total_exposure`, `num_active_markets`

### Data Retention
Daily cleanup prunes rows older than `DATA_RETENTION_DAYS` (default 7), keeping hourly samples for long-term history.

## API Layer

### REST Endpoints

- `GET /api/status` — Bot state, uptime, active market count, environment
- `GET /api/markets` — Actively quoted markets with current quotes and inventory
- `GET /api/markets/{symbol}` — Detail: quote history, order book, recent trades
- `GET /api/pnl?range=1h|6h|24h|7d` — P&L time series
- `GET /api/positions` — Current inventory with unrealized P&L
- `POST /api/bot/start` / `POST /api/bot/stop` — Control bot loop
- `POST /api/config` — Update runtime parameters (gamma, k, max inventory) without restart

### WebSocket

- `ws://localhost:8000/ws/dashboard` — Single multiplexed connection
- Pushes every cycle: `{ type: "tick", data: { markets, positions, pnl, quotes } }`
- Event-driven: `{ type: "order_fill", data: {...} }`, `{ type: "risk_alert", data: {...} }`

CORS configured for `FRONTEND_URL` env var.

## Dashboard

Dark theme, single-page layout with four panels updating in real-time:

### Top Bar
Bot status indicator (green/red), environment badge (SANDBOX/LIVE), start/stop button.

### Panel 1 — P&L Chart (full width)
Recharts line chart: realized + unrealized P&L over time. Toggle 1h/6h/24h/7d. Green/red fill.

### Panel 2 — Inventory Heatmap (left)
Table of active markets. Columns: symbol, net inventory (color-coded green/yellow/red by proximity to limit), bid/ask, spread captured. Sorted by absolute inventory.

### Panel 3 — Reservation Price vs Mid (right)
Per-market chart: mid-price (blue), reservation price (orange), bid/ask band (shaded). Tooltip explains the inventory skew in plain English.

### Panel 4 — Activity Log (bottom, full width)
Scrolling feed of bot actions in plain English, explaining *why* each action was taken:
- "Placed BID on GEMI-BTC100K-YES at $0.47 (mid: $0.50, inventory: +30)"
- "Risk alert: inventory on GEMI-FEDJAN26 hit 180/200 limit, widening spread"
- "Order filled: SOLD 5 GEMI-BTC100K-YES at $0.52, realized +$0.15"

Dashboard hydrates from REST on load, stays live via WebSocket.

## Configuration

All via `.env` file:

```env
# Gemini API
GEMINI_API_KEY=your-api-key
GEMINI_API_SECRET=your-api-secret
GEMINI_ENV=sandbox

# Avellaneda-Stoikov Parameters
AS_GAMMA=0.1
AS_K=1.5
AS_SIGMA_DEFAULT=0.01
AS_VARIANCE_WINDOW=100

# Bot Behavior
BOT_CYCLE_SECONDS=10
SCANNER_CYCLE_SECONDS=300
MIN_SPREAD=0.03
MIN_TIME_TO_EXPIRY_HOURS=1
EXCLUDED_SYMBOLS=

# Risk Limits
MAX_INVENTORY_PER_SYMBOL=200
MAX_TOTAL_EXPOSURE=1000
RISK_WIDEN_THRESHOLD=0.8

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/gemini_mm

# Frontend
FRONTEND_URL=http://localhost:3000

# Data
DATA_RETENTION_DAYS=7
```

Runtime updates via `POST /api/config` apply in-memory for the next cycle. Restart resets to `.env` values.
