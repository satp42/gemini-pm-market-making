# Feature Specification: Performative Market Making Engine

## 1. Overview

### Problem Statement

The current market-making bot uses the standard Avellaneda-Stoikov (A&S) quoting model, which treats the mid-price as an exogenous process -- it assumes the market maker's own orders have no influence on price dynamics. In prediction markets on Gemini, where liquidity is thin and a small number of automated participants dominate order flow, this assumption breaks down. A market maker's posted quotes shift the order book and pull the mid-price toward them, creating a feedback loop (the "performative" effect). The existing A&S engine in `backend/src/engine/quoting.py` ignores this feedback, leading to:

1. **Systematic mis-pricing of reservation price**: The A&S formula `r = s - q * gamma * sigma^2 * (T-t)` overpays for inventory when the mid-price is drifting toward the market maker's quotes, and under-compensates when the drift works against it.
2. **Suboptimal spread setting**: The A&S spread `delta = gamma * sigma^2 * (T-t) + 2/gamma * ln(1 + gamma/k)` does not account for the mean-reverting component that the market maker's own orders introduce.
3. **Missed arbitrage opportunities**: When competing A&S bots push price in a predictable direction, the current engine cannot exploit the drift because it models price as a pure diffusion.

The research paper "Performative Market Making" (arXiv:2508.04344) provides a corrected model that accounts for this feedback loop through a parameter xi (performative feedback strength), which quantifies how strongly the market maker's actions pull the mid-price.

### Goals & Success Criteria

**Goals**:
1. Replace the A&S reservation price and spread formulas with performative-aware equivalents that account for the market maker's own price impact.
2. Exploit non-performative competitors by using the performative model's awareness of predictable drift to systematically profit against standard A&S bots.
3. Learn separate model parameters (theta vectors) for different prediction market categories (Crypto, Sports, Politics, Economics) to capture category-specific microstructure differences.
4. Demonstrate a measurable edge over vanilla A&S market makers through back-tested simulation results.

**Success Criteria**:
- Quoted bid-ask prices reflect the performative discount on mid-price, observable as the reservation price diverging from the simple A&S formula when xi is nonzero.
- The bot estimates xi from live trade data for each active market within 2 bot cycles (under 20 seconds) after market selection.
- Theta parameters per category converge within 100 Optuna trials against historical trade data, producing measurable negative-utility improvement over vanilla A&S in simulation.
- The system operates within the same risk limits, cycle time, and price constraints as the current bot (10-second cycles, [0.01, 0.99] price clamping, all existing risk checks preserved).
- Dashboard displays new performative-specific metrics (xi estimate, active theta values, quoting model type) alongside existing quote visualization.

---

## 2. User Stories & Use Cases

### Primary Use Cases

**Scenario 1: Normal Performative Quoting Cycle**

1. Bot cycle starts. Scanner has selected 5 active symbols across Crypto and Politics categories.
2. For each symbol, the system fetches market state (order book + 100 recent trades) -- this is the existing flow in `OrderBookMonitor.get_market_state`.
3. Xi estimator runs OU regression on the 100 trade prices. For a Crypto symbol with active trading, it estimates xi=1.8.
4. Theta lookup finds optimized values for the Crypto category: (theta0=1.15, theta1=0.92, theta2=1.08).
5. The performative reservation price is computed: the mid-price is discounted by `exp(-1.8 * 0.5)` and the inventory correction uses the theta-scaled terms.
6. The performative spread is computed: wider than A&S due to the mean-reversion correction.
7. Risk checks pass (all existing checks in `RiskManager` are preserved). Bid and ask are placed, clamped to [0.01, 0.99].
8. Quote record is persisted with xi=1.8, theta values, and mode="theta".
9. Dashboard receives the tick via WebSocket and displays xi, theta, and mode.

**Scenario 2: Fallback Chain Activation**

1. Bot cycle starts for a newly listed Politics symbol with only 8 trades.
2. Xi estimator finds fewer than 15 trades (the `xi_min_trades` threshold). Logs WARNING: "Insufficient trades for xi estimation, using default xi=0.5".
3. Theta lookup for Politics category finds no optimized parameters (optimization has not run yet for this category).
4. System falls back from "theta" mode to "performative" mode. Logs WARNING: "No theta parameters for category 'Politics', falling back to vanilla performative".
5. Vanilla performative quote is computed with xi=0.5 and theta=(1,1,1).
6. Quote record is persisted with mode="performative", xi=0.5.

**Scenario 3: Theta Optimization Run**

1. Operator triggers optimization via API endpoint, or the auto-schedule fires (default: every 24 hours).
2. System derives categories from distinct `event.category` values across all active events (fetched via `get_events()`): ["Crypto", "Sports", "Politics", "Economics"].
3. For "Crypto", system collects trade history across all active Crypto symbols.
4. Optuna study begins: 100 trials, each running 100 simulations of the theta-enhanced performative model against the historical mid-price series, minimizing expected negative CARA utility.
5. After approximately 5 minutes, optimization completes. Best theta for Crypto: (1.15, 0.92, 1.08).
6. Results are persisted to the `theta_parameters` database table.
7. On the next bot cycle, the live quoting engine picks up the new theta values.
8. Activity log shows: "Theta optimization completed for Crypto: theta=(1.15, 0.92, 1.08)".

**Scenario 4: Mode Switch via API**

1. Operator sends `POST /config` with `{"quoting_mode": "as"}` to revert to standard A&S quoting.
2. Config is updated in runtime overrides.
3. On the next bot cycle, `_process_symbol` uses the standard `compute_quote` function unchanged.
4. Dashboard status bar updates to show "Avellaneda-Stoikov" mode.
5. Operator sends `POST /config` with `{"quoting_mode": "theta"}` to resume performative quoting.
6. Next cycle resumes performative quoting with the latest xi estimates and theta parameters.

### Edge Cases

- **Flat price series**: All 100 trades have identical prices. OU regression produces near-zero slope. Xi defaults to the minimum clamp (0.01) rather than producing NaN or division-by-zero.
- **Extremely high xi**: Xi estimated at 20.0 (upper clamp). `exp(-20 * 0.5) = 4.5e-5`, effectively zeroing the mid-price component. Reservation price is dominated by the inventory correction term. The system clamps to valid range and quotes normally.
- **Bid-ask crossing after performative spread**: Very wide performative spread combined with reservation price near the boundary could produce bid > ask after [0.01, 0.99] clamping. System detects this, skips quoting for that cycle, and logs a warning.
- **Empty categories list**: All events have empty-string categories (Gemini does not populate the field). System logs a warning and uses default theta (1,1,1) for all symbols.
- **Concurrent optimization and quoting**: Theta optimization is running in the background while live quoting reads theta values. Reads use the last-committed theta from the database; no locking needed because writes are atomic row replacements.

---

## 3. Functional Requirements

### Core Requirements

#### FR-1: Xi (Performative Feedback Strength) Estimation

The system must estimate the performative feedback parameter xi for each active market by fitting an Ornstein-Uhlenbeck (OU) process to the mid-price time series derived from recent trade data.

**Behavior**:
1. On each bot cycle, for each active symbol, use the trade prices already fetched by `OrderBookMonitor.get_market_state` (up to 100 trades).
2. Compute price changes (`delta_s = s_{n+1} - s_n`) and estimate xi via simple OLS regression:
   ```
   delta_s ~ beta * s_n + alpha
   ```
   Xi is extracted as `xi = -beta / dt`, where `dt` is the average time step between consecutive trades. The inventory-dependent drift term (`-gamma * sigma^2 * q * (T-t)`) from the full OU dynamics is intentionally omitted from the regression because: (a) the market maker's own inventory drift is small relative to market-wide mean reversion, and (b) the estimation uses market-observable mid-price changes, not the market maker's own model-internal state. This simplification keeps the estimator stateless -- it requires only trade prices and timestamps, not inventory history.
3. Clamp xi to a valid range [xi_clamp_min, xi_clamp_max] (defaults: [0.01, 20.0]).
4. **R-squared quality gate**: If the regression r_squared is below 0.1, the xi estimate is considered unreliable. The system falls back to `xi_default` (from config) and logs a WARNING: "Xi regression r_squared={value} below threshold 0.1 for {symbol}, using default xi={xi_default}".
5. If fewer than `xi_min_trades` trades (default: 15) are available, fall back to `xi_default` (default: 0.5).

**Acceptance Criteria**:
- AC-1.1a: Given a synthetic trade series of 100 data points generated from an OU process with known xi=2.0, the estimator recovers xi within the range [1.5, 2.8].
- AC-1.1b: Given a synthetic trade series of 500 data points generated from an OU process with known xi=2.0, the estimator recovers xi within the range [1.8, 2.3].
- AC-1.2: Given fewer than 15 trades, the system uses the configured default xi without error.
- AC-1.3: Given a flat price series (all identical prices), xi defaults to the minimum clamp (0.01) rather than producing a division-by-zero or NaN.
- AC-1.4: Xi estimation adds no more than 5 milliseconds of computation per symbol per cycle.
- AC-1.5: Given a trade series with random noise (no mean-reversion signal), regression r_squared is below 0.1, and the system falls back to default xi with a logged warning.

#### FR-2: Performative Reservation Price Computation

Replace the current A&S reservation price with the performative-aware formula that discounts the mid-price by the feedback factor and applies a dual inventory correction.

**Behavior**:
1. Compute the vanilla performative reservation price:
   ```
   delta_epsilon = (1 - exp(-xi*(T-t)) - xi*(T-t)*exp(-xi*(T-t))) / xi^2
   r_perf = s * exp(-xi*(T-t)) - gamma * sigma^2 * (q_ref * delta_epsilon - q * (exp(-2*xi*(T-t)) - 1) / (2*xi))
   ```
   Where: `s` = current mid-price, `xi` = estimated feedback strength, `T-t` = time remaining, `gamma` = risk aversion, `sigma^2` = estimated price variance, `q` = current net inventory, `q_ref` = reference inventory (default 0).

2. When theta parameters are available, use the theta-enhanced formula:
   ```
   r_theta = theta0 * s * exp(-xi*(T-t)) - gamma * sigma^2 * (theta1 * q_ref * delta_epsilon - theta2 * q * (exp(-2*xi*(T-t)) - 1) / (2*xi))
   ```

3. Clamp the final reservation price to [0.01, 0.99].

**Acceptance Criteria**:
- AC-2.1: When xi approaches 0 (using L'Hopital limits), the performative reservation price converges to the standard A&S reservation price within floating-point tolerance of 1e-6.
- AC-2.2: When xi > 0 and inventory q = 0, the reservation price is strictly less than mid-price s (the discount effect is visible).
- AC-2.3: When xi > 0 and inventory q > 0 (long), the reservation price shifts downward more aggressively than A&S.
- AC-2.4: All outputs remain within [0.01, 0.99] regardless of input values.

#### FR-3: Performative Spread Computation

Replace the current A&S spread formula with the performative-aware spread that accounts for the mean-reverting component.

**Behavior**:
1. Compute spread:
   ```
   spread_perf = 2/gamma * ln(1 + gamma/k) - (gamma * sigma^2 / (2*xi)) * (exp(-2*xi*(T-t)) - 1)
   ```
   The term `(exp(-2*xi*(T-t)) - 1)` is negative, so the subtraction adds a positive quantity -- the performative spread is wider than the A&S minimum spread component.

2. Minimum spread floor is `2/gamma * ln(1 + gamma/k)` (the A&S order-arrival component). Spread can only increase from the performative correction, never decrease below this floor.

3. **Max spread cap**: The existing `max_spread` cap (default 0.10) applies to performative spreads as well. If the computed performative spread exceeds `max_spread`, it is clamped to `max_spread`. This preserves the risk management invariant that the bot never posts excessively wide quotes, regardless of quoting mode.

4. Derive bid and ask: `bid = r_perf - spread_perf / 2`, `ask = r_perf + spread_perf / 2`. Both clamped to [0.01, 0.99].

**Acceptance Criteria**:
- AC-3.1: When xi approaches 0, the spread equals the standard A&S spread within floating-point tolerance of 1e-6.
- AC-3.2: When xi > 0, the spread is at least as wide as the A&S spread.
- AC-3.3: Bid is always strictly less than ask after clamping. If they would cross, the system skips quoting for that cycle and logs a warning.
- AC-3.4: With default parameters (gamma=0.1, k=1.5, sigma^2=0.01, xi=1.0, T-t=0.5), spread remains under 0.50.

#### FR-4: Category-Specific Theta Parameter Learning

For each Gemini prediction market category, learn separate theta vectors (theta0, theta1, theta2) by running simulations optimized via Optuna to minimize expected negative CARA utility.

**Behavior**:
1. Derive the list of active categories from the `Event.category` field across all events already fetched by `MarketScanner.scan()` (via `get_events()`). There is no separate categories API endpoint; categories are the distinct non-empty values of `event.category` from the events response.
2. For each category, collect historical trade data across all symbols in that category.
3. Run Optuna optimization: theta0 in [0.5, 2.0], theta1 in [0.5, 2.0], theta2 in [0.5, 2.0]. Up to 100 trials, each running 100 simulated trading sessions.
4. Store the best theta vector per category in the `theta_parameters` database table.
5. **Prerequisite -- Scanner category mapping**: `MarketScanner.scan()` currently returns `(symbols, symbol_titles, symbol_expiry_hours)` but does not return category information. Extend `scan()` to also return `symbol_categories: dict[str, str]` mapping each symbol to its event category (derived from `Event.category`). Symbols whose events have an empty-string category are mapped to `""`.
6. On each bot cycle, look up the theta vector for the current symbol's category (using the symbol-to-category mapping from the scanner). If no optimized theta exists for that category, fall back to (1.0, 1.0, 1.0).
7. Re-optimization is triggered manually via API or runs automatically on a configurable schedule (default: every 24 hours).

**Acceptance Criteria**:
- AC-4.1: After optimization, theta values for at least one category differ from (1.0, 1.0, 1.0).
- AC-4.2: Optimization for a single category completes within 10 minutes (100 trials x 100 simulations).
- AC-4.3: Live quoting continues uninterrupted while optimization runs in the background.
- AC-4.4: Theta values are persisted across bot restarts.
- AC-4.5: If no events have a populated category field, the system logs a warning and uses default thetas for all symbols.

#### FR-5: Quoting Mode Selection and Fallback

The system must support switching between quoting models and falling back gracefully.

**Behavior**:
1. Three quoting modes: "as" (standard A&S, unchanged), "performative" (vanilla performative with theta=(1,1,1)), "theta" (theta-enhanced performative with category-specific parameters).
2. Mode is configurable at runtime via the `/config` endpoint (new field: `quoting_mode`).
3. Default mode is "theta". Fallback chain: theta -> performative (if no theta available) -> as (if xi estimation fails).
4. Active mode is included in every persisted quote record.

**Acceptance Criteria**:
- AC-5.1: Setting `quoting_mode` to "as" produces identical quotes to the current system given the same inputs.
- AC-5.2: Setting `quoting_mode` to "performative" produces quotes that differ from A&S when xi is nonzero.
- AC-5.3: The fallback chain activates correctly when dependencies are unavailable, with each fallback logged at WARNING level.
- AC-5.4: Mode changes take effect on the next bot cycle without restart.

#### FR-6: Dashboard Integration

The frontend dashboard must display performative-model-specific data alongside existing visualizations.

**Behavior**:
1. Extend market summary data to include: `xi`, `theta0`, `theta1`, `theta2`, `quotingMode`.
2. The reservation price chart visually distinguishes between A&S and performative reservation prices.
3. The status bar shows the active quoting mode.
4. The activity log includes entries for xi estimation events, theta optimization completions, and fallback activations.

**Acceptance Criteria**:
- AC-6.1: Dashboard loads without error when performative fields are present in the API response.
- AC-6.2: Dashboard loads without error when performative fields are absent (backward compatibility).
- AC-6.3: Xi and theta values are displayed with 3 decimal places.
- AC-6.4: Quoting mode is displayed as a human-readable label ("Avellaneda-Stoikov", "Performative", "Theta-Enhanced").

### API Requirements

#### Extended Endpoints

**`POST /config`** (existing, extended):
- New accepted fields: `quoting_mode`, `xi_default`, `xi_min_trades`, `xi_clamp_min`, `xi_clamp_max`, `q_ref`, `theta_optimization_trials`, `theta_optimization_simulations`, `theta_auto_optimize_hours`.
- Response includes new fields in the `effective` config object.

**`POST /optimize/theta`** (new):
- Triggers theta optimization for all categories (or a specific category if `category` query param is provided).
- Returns immediately with a job status (optimization runs asynchronously).
- Response: `{"status": "started", "categories": ["Crypto", "Sports", ...]}`.

**`GET /optimize/theta/status`** (new):
- Returns current optimization status: running/idle, last completed time per category, current trial progress if running.

**`GET /markets`** (existing, extended):
- Response includes new fields per market: `xi`, `theta0`, `theta1`, `theta2`, `quotingMode`.
- New fields are nullable (null when not applicable or when viewing historical A&S-only data).

**WebSocket tick message** (existing, extended):
- Market data objects include new fields: `xi`, `theta0`, `theta1`, `theta2`, `quotingMode`.

### Data Requirements

#### Modified Entities

**Quote Record** (extends existing `quotes` table):
| Column | Type | Nullable | Description |
|---|---|---|---|
| `xi` | float | yes | Estimated feedback strength used for this quote |
| `theta0` | float | yes | Theta-0 parameter (null for A&S mode) |
| `theta1` | float | yes | Theta-1 parameter (null for A&S mode) |
| `theta2` | float | yes | Theta-2 parameter (null for A&S mode) |
| `quoting_mode` | string(16) | yes | "as", "performative", or "theta" |

#### New Entities

**Theta Parameters** (new table `theta_parameters`):
| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | int | no | Primary key |
| `category` | string(64) | no | Prediction market category (unique) |
| `theta0` | float | no | Optimized theta-0 value |
| `theta1` | float | no | Optimized theta-1 value |
| `theta2` | float | no | Optimized theta-2 value |
| `xi_value` | float | no | Xi value used during optimization |
| `objective_value` | float | no | Best negative utility achieved |
| `num_trials` | int | no | Optuna trials completed |
| `optimized_at` | datetime | no | When optimization completed |
| `created_at` | datetime | no | Row creation time |
| `updated_at` | datetime | yes | Last update time (set on re-optimization) |

**Xi Estimates** (new table `xi_estimates`, for observability/analysis):
| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | int | no | Primary key |
| `timestamp` | datetime | no | Estimation time |
| `symbol` | string(128) | no | Market symbol |
| `xi` | float | no | Estimated value |
| `num_trades` | int | no | Trades used in estimation |
| `r_squared` | float | yes | Regression fit quality |

---

## 4. Non-Functional Requirements

### Performance

- Xi estimation must complete in under 5 milliseconds per symbol (in-memory regression on up to 100 data points).
- Performative quote computation must complete in under 1 millisecond per symbol (mathematical formula evaluation).
- Total bot cycle time must remain under 10 seconds when quoting across up to 20 simultaneous markets.
- Theta optimization must run in a separate background task and must not block or delay live quoting cycles.

### Security

- No new external API credentials are introduced. All Gemini API calls use the existing HMAC authentication in `gemini/auth.py`.
- Theta optimization results are stored locally in the database. No sensitive data leaves the system.
- The new `/optimize/theta` endpoint must respect the same access controls as the existing `/bot/start` and `/bot/stop` endpoints.

### Reliability

- **Numerical stability**: All exponential terms must be guarded against overflow (e.g., `exp(-xi*(T-t))` when xi * (T-t) exceeds 700). Division by xi must use L'Hopital limits or A&S fallback when xi approaches zero.
- **Backward compatibility**: The existing database schema is extended with nullable columns, not replaced. Old quote records remain queryable. The REST API returns all current fields unchanged; new fields are additive. The WebSocket message format is preserved with additive new fields.
- **Graceful degradation**: If xi estimation fails for any reason, the system falls back to A&S quoting for that symbol. If theta optimization fails, the system continues with default theta (1,1,1). No single failure path can prevent the bot from quoting.
- All intermediate computations use 64-bit floating point. The system must produce finite, non-NaN results for all valid input combinations.

---

## 5. Technical Design

> Note: This section provides architectural guidance. Implementation details are deliberately high-level to avoid prescribing specific code structure.

### Architecture Overview

The performative engine is a drop-in replacement for the quoting computation step within the existing `BotLoop._process_symbol` flow. It sits between the market state fetch and the risk check / order placement steps. The rest of the pipeline (scanner, book monitor, position tracker, order manager, risk manager) remains unchanged.

**Prerequisite: Config Override Propagation**

The existing `_process_symbol` in `loop.py` reads quoting parameters (gamma, k, etc.) directly from `self._settings.avellaneda_stoikov`, but the `/config` endpoint writes runtime overrides to `app.state.config_overrides` -- a dict that the bot loop never consults. Before implementing performative mode switching, the BotLoop must be updated to read `quoting_mode` and all performative parameters (`xi_default`, `xi_min_trades`, `xi_clamp_min`, `xi_clamp_max`, `q_ref`) from `config_overrides` first, falling back to `self._settings` when no override exists. Without this fix, `POST /config` changes to `quoting_mode` will have no effect on live quoting.

**Prerequisite: WebSocket Quote Broadcast**

The current WebSocket broadcast does not properly transmit quote data per symbol (the `quote_summary` key is unpopulated). This must be fixed before performative fields (xi, theta, quotingMode) can be displayed on the dashboard in real time. FR-6 depends on this prerequisite being resolved.

### Key Components

1. **Xi Estimator**: A stateless function that takes a list of trade prices and returns an estimated xi value. Called per-symbol per-cycle.
2. **Performative Quoter**: A stateless function (analogous to the existing `compute_quote`) that takes mid-price, inventory, xi, theta, gamma, sigma^2, T-t, k, and q_ref, and returns a Quote object with performative bid/ask/reservation/spread.
3. **Theta Store**: A lightweight read layer over the `theta_parameters` database table. Caches the latest theta per category in memory, refreshed periodically.
4. **Theta Optimizer**: A background async task that runs Optuna studies per category. Writes results to the database. Decoupled from the live quoting path.
5. **Mode Router**: Logic within `_process_symbol` that selects which quoter to call based on the configured mode and available data (fallback chain).

### Data Flow

```
Trade Prices (from OrderBookMonitor)
        |
        v
  Xi Estimator --> xi value (or default)
        |
        v
  Theta Store --> theta vector (or default)
        |
        v
  Mode Router --> selects quoter
        |
        +-- "as" --> existing compute_quote()
        +-- "performative" --> performative_quote(xi, theta=(1,1,1))
        +-- "theta" --> performative_quote(xi, theta=category_theta)
        |
        v
  Risk Checks (unchanged)
        |
        v
  Order Placement (unchanged)
```

### Dependencies

| Dependency | Type | Description |
|---|---|---|
| `GET /v1/trades/{symbol}` | External API | Trade history for xi estimation (already called by `OrderBookMonitor`) |
| `GET /v1/book/{symbol}` | External API | Order book for mid-price (already called by `OrderBookMonitor`) |
| `GET /v1/prediction-markets/events` | External API | Event metadata including category field (already called by `MarketScanner`). Categories are derived from distinct `event.category` values -- no separate categories endpoint exists. |
| Optuna | Python package | Hyperparameter optimization for theta learning (new dependency) |
| NumPy | Python package | Numerical computation for OU regression and simulation (new dependency) |
| Existing `compute_quote` | Internal | Preserved for A&S fallback mode |
| Existing `Quote` dataclass | Internal | See "Quote Dataclass Strategy" below |
| Existing `QuoteRecord` model | Internal | Extended with new nullable columns |
| Existing `/config` endpoint | Internal | Extended to accept new parameters |

#### Quote Dataclass Strategy

The existing `Quote` in `quoting.py` is `@dataclass(frozen=True)` with exactly 10 fields (`bid_price`, `ask_price`, `reservation_price`, `spread`, `mid_price`, `inventory`, `sigma_sq`, `gamma`, `t_minus_t`, `k`). Adding xi and theta fields would change the constructor signature and break all existing construction sites (including the `WIDEN_SPREAD` path in `_process_symbol` and all unit tests that construct `Quote` directly).

**Recommended approach**: Change `Quote` to non-frozen and add the new fields as optional with defaults:
- `xi: float | None = None`
- `theta0: float | None = None`
- `theta1: float | None = None`
- `theta2: float | None = None`
- `quoting_mode: str | None = None`

This preserves backward compatibility -- all existing construction sites continue to work without modification because the new fields have defaults. The `frozen=True` removal is acceptable because `Quote` instances are never used as dict keys or set members; they are created, read, and discarded each cycle. The test `test_quote_is_frozen_dataclass` in `test_quoting.py` must be updated to reflect this change.

**Alternative approach** (if immutability must be preserved): Create a `PerformativeQuote` dataclass that wraps `Quote` with additional fields. The bot loop uses `PerformativeQuote` when in performative/theta mode and `Quote` when in A&S mode. This adds branching complexity but preserves the frozen invariant.

---

## 6. API Specification

### Endpoints

#### `POST /optimize/theta` (New)

Trigger theta optimization for prediction market categories.

**Query Parameters**:
| Parameter | Type | Required | Description |
|---|---|---|---|
| `category` | string | no | Specific category to optimize. If omitted, all categories are optimized. |

**Response** (202 Accepted):
```json
{
  "status": "started",
  "categories": ["Crypto", "Sports", "Politics", "Economics"]
}
```

**Error Responses**:
- 409 Conflict: "Optimization is already running"
- 503 Service Unavailable: "Bot loop not initialized"

#### `GET /optimize/theta/status` (New)

Get current theta optimization status.

**Response** (200 OK):
```json
{
  "running": false,
  "lastRun": "2026-03-25T14:30:00Z",
  "categories": {
    "Crypto": {
      "theta0": 1.15,
      "theta1": 0.92,
      "theta2": 1.08,
      "objectiveValue": -0.0023,
      "optimizedAt": "2026-03-25T14:30:00Z",
      "numTrials": 100
    },
    "Sports": null
  }
}
```

### Request/Response Formats

#### Extended `GET /markets` Response

Each market summary object gains these additional nullable fields:

```json
{
  "symbol": "GEMI-BTC100K-26MAR26",
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
```

#### Extended `POST /config` Request

New accepted fields (all optional):

```json
{
  "quoting_mode": "theta",
  "xi_default": 0.5,
  "xi_min_trades": 15,
  "xi_clamp_min": 0.01,
  "xi_clamp_max": 20.0,
  "q_ref": 0.0,
  "theta_optimization_trials": 100,
  "theta_optimization_simulations": 100,
  "theta_auto_optimize_hours": 24
}
```

---

## 7. Configuration & Parameters

| Parameter | Default | Valid Range | Description |
|---|---|---|---|
| `quoting_mode` | "theta" | "as", "performative", "theta" | Active quoting model |
| `xi_default` | 0.5 | [0.01, 20.0] | Fallback xi when estimation fails or insufficient data |
| `xi_min_trades` | 15 | [5, 100] | Minimum trades required for xi estimation |
| `xi_clamp_min` | 0.01 | > 0 | Lower bound for estimated xi |
| `xi_clamp_max` | 20.0 | > xi_clamp_min | Upper bound for estimated xi |
| `q_ref` | 0.0 | any float | Reference (target) inventory for performative formula |
| `theta_optimization_trials` | 100 | [10, 1000] | Optuna trials per category per optimization run |
| `theta_optimization_simulations` | 100 | [10, 1000] | Simulations per Optuna trial |
| `theta_auto_optimize_hours` | 24 | [1, 168] | Hours between automatic theta re-optimization |

All existing A&S parameters (`gamma`, `k`, `sigma_default`, `variance_window`) and risk parameters remain unchanged and continue to apply to the performative model.

**Validation Rules**:
- `xi_clamp_min` must be less than `xi_clamp_max`. Rejected with descriptive error otherwise.
- `quoting_mode` must be one of the three valid values. Rejected with descriptive error otherwise.
- All numerical parameters must be finite and non-NaN.

---

## 8. Testing Strategy

### Unit Tests

- **Xi estimator**: Verify recovery of known xi from synthetic OU data. Verify fallback on insufficient data. Verify clamping on edge cases (flat series, extreme values).
- **Performative quoter**: Verify mathematical equivalence to A&S at xi=0. Verify directional properties (discount effect, spread widening). Verify [0.01, 0.99] clamping under all input combinations.
- **Theta store**: Verify cache loading from database. Verify default fallback when no row exists. Verify cache refresh after optimization writes.
- **Mode router**: Verify correct function is called for each mode. Verify fallback chain logic when dependencies are missing.

### Integration Tests

- **End-to-end bot cycle with performative mode**: Start bot in "theta" mode with mock Gemini responses. Verify that quotes placed use performative formulas and that quote records contain xi and theta values.
- **Fallback integration**: Start bot with insufficient trade data and no theta parameters. Verify the fallback chain fires correctly and quotes are still placed using A&S.
- **Theta optimization**: Run a short optimization (10 trials, 10 simulations) against mock data. Verify theta values are persisted and picked up by the next quoting cycle.
- **Config update**: Change quoting_mode at runtime via API. Verify next cycle uses the new mode.

### Performance Tests

- **Xi estimation timing**: Benchmark xi estimation on 100 trade prices. Verify it completes under 5ms on target hardware.
- **Quote computation timing**: Benchmark performative quote computation. Verify it completes under 1ms.
- **Optimization duration**: Run full optimization (100 trials, 100 simulations) on a single category. Measure wall-clock time and verify it completes under 10 minutes.

---

## 9. Deployment & Operations

### Deployment Steps

1. Run database migration to add new columns to `quotes` table and create `theta_parameters` and `xi_estimates` tables. Use Alembic for database migrations. If Alembic is not yet configured in the project, add `alembic init` and create the initial migration as a prerequisite before generating the performative-engine migration.
2. Deploy updated backend with new performative engine code.
3. Deploy updated frontend with extended dashboard components.
4. The bot starts in "theta" mode by default. With no theta parameters in the database, it falls back to "performative" mode using default xi.
5. Trigger initial theta optimization via `POST /optimize/theta` once the bot has accumulated trade data.

### Monitoring

- **Xi estimation quality**: Track distribution of estimated xi values over time. Alert if xi is consistently at the clamp boundaries (may indicate estimation is not working).
- **Theta convergence**: Track objective values across optimization runs. Alert if values are not improving or are worsening.
- **Fallback frequency**: Track how often the system falls back from theta to performative to A&S. High fallback rates may indicate insufficient market data.
- **Quote computation latency**: Track per-cycle computation time. Alert if xi estimation or quote computation exceeds expected bounds.
- All existing monitoring (risk alerts, kill switch status, API errors) remains in place.

### Rollback Plan

1. Set `quoting_mode` to "as" via `POST /config`. This immediately reverts to standard A&S quoting on the next cycle. No deployment required.
2. If database schema changes cause issues, the new columns are all nullable and do not affect existing queries. The migration can be reversed by dropping the new columns and tables.
3. Full code rollback: revert to previous backend deployment. The old code ignores the new database columns (they are nullable and not queried by old code).

---

## 10. Open Questions & Risks

### Known Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Xi estimation is unreliable with < 50 trades | Medium | Incorrect quote pricing | Configurable minimum trades threshold; fallback to A&S when data is sparse |
| Theta optimization overfits to historical data | Medium | Poor live performance vs. simulation | Cross-validate on held-out data; monitor live vs. simulated performance |
| Performative spread is too wide for thin markets | Low | No fills, bot provides no liquidity | Spread can be capped via existing min_spread parameter; mode can be switched to A&S |
| Competing bots adapt to performative quoting | Low (hackathon timeframe) | Edge diminishes | The model is fundamentally more accurate; adaptation takes time |
| Gemini API rate limits exceeded during theta optimization | Low | Optimization fails or is throttled | Optimization reuses already-fetched historical data; no additional live API calls during optimization beyond initial data fetch |

### Open Questions

1. **Theta optimization data threshold**: What is the minimum amount of historical trade data required per category before theta optimization should be attempted? If a category has only 1-2 symbols each with fewer than 50 trades, running 10,000 simulations may produce unreliable theta values. Should there be a minimum data threshold (e.g., 500 total trades across all symbols in the category) below which optimization is skipped?

2. **Per-symbol mode override**: Should all symbols use the same quoting mode globally, or should it be possible to run different modes on different symbols (e.g., "theta" on Crypto, "as" on Sports where xi estimation is unreliable)? Per-symbol mode selection would add complexity but could improve performance in categories with sparse data.

---

## Constraints

1. **Gemini API rate limits**: The system must not increase the number of API calls per cycle. Xi estimation reuses trade data already fetched by `OrderBookMonitor.get_market_state`.
2. **Price bounds**: All quoted prices must remain in [0.01, 0.99] as required by Gemini prediction markets.
3. **Order types**: Only limit orders are supported.
4. **Existing risk framework**: All existing risk checks (inventory limits per symbol, total exposure cap, time-to-expiry safety, kill switch) must continue to function identically. The performative engine affects only the reservation price and spread computation step, not the risk decision flow.
5. **Python backend**: Must integrate with the existing FastAPI backend. Optuna and NumPy are added as new Python dependencies.
6. **Single bot instance**: Only one instance of the bot runs against a Gemini account at a time.

## Assumptions

1. Active prediction markets on Gemini have at least 15 trades within the 100-trade window frequently enough to estimate xi. When they do not, the default xi provides a reasonable approximation.
2. Xi is approximately stationary over short horizons (minutes to hours). Re-estimating per cycle with the latest 100 trades is sufficient.
3. Gemini prediction market categories are relatively stable. New categories default to theta (1.0, 1.0, 1.0) until the next optimization run.
4. The paper's simulation parameters (sigma=2, T=1, N=200, k=1.5, gamma=0.1) are reference points for validation, not production defaults. Live parameters differ and are configurable.
5. `T-t` is normalized as a fraction of a day (24h = 1.0), consistent with the existing `_process_symbol` logic which defaults to 0.5.
6. Symbols with empty-string categories (where Gemini does not populate the `Event.category` field) use the default theta vector (1.0, 1.0, 1.0), which reduces to the vanilla performative formula. These symbols are not grouped into a separate "default" category for theta optimization.

## Out of Scope

1. **Multi-agent simulation**: Simulating multiple competing performative agents. Theta optimization uses single-agent simulation against historical price data.
2. **Real-time WebSocket-based xi streaming**: Xi is estimated per cycle from REST data, not from the WebSocket book ticker stream.
3. **Adaptive xi window**: Dynamically adjusting the number of trades used for xi estimation based on market activity.
4. **Cross-category theta transfer learning**: Theta parameters are learned independently per category.
5. **Automated A/B testing**: Side-by-side statistical comparison of A&S vs. performative quoting on live markets.
6. **Event outcome modeling**: The performative model operates on price dynamics only, not on the probability of event outcomes.
7. **Mobile-responsive dashboard**: The dashboard targets desktop use only (existing Next.js application).

---

## Self-Critique Summary

### Verification Trace

**Requirements Completeness**: COMPLETE. All four components from the feature description are covered: xi estimation (FR-1), performative formulas (FR-2, FR-3), theta learning (FR-4), and the competitive edge (implicit in the model design, described in Business Goals). Edge cases documented: insufficient trades, zero xi, NaN guards, bid-ask crossing, empty categories. Error scenarios: fallback chain in FR-5 with three levels of degradation.

**Stakeholder Coverage**: COMPLETE. Bot operator needs addressed via dashboard integration (FR-6) and mode switching (FR-5). Risk management preservation is explicit in Constraints. Background optimizer process described in FR-4 with non-interference guarantee (AC-4.3). Frontend backward compatibility addressed in AC-6.2.

**Scope Clarity**: COMPLETE. Seven explicit Out of Scope items. Constraints section defines five hard boundaries. The line between "what this feature does" and "what it does not" is unambiguous.

**Acceptance Criteria Testability**: COMPLETE. Every FR has numbered, concrete acceptance criteria. FR-1: synthetic data recovery with numeric bounds, insufficient data fallback, flat series edge case. FR-2: mathematical equivalence at xi=0, directional property tests. FR-3: monotonicity and floor guarantees, crossing guard. FR-4: non-triviality, timing, persistence. FR-5: identity test for A&S mode, fallback logging verification.

**Business Value Traceability**: COMPLETE. FR-1 through FR-3 trace to Goal 1 (quoting accuracy). FR-4 traces to Goal 3 (category optimization). FR-5 traces to operational safety. FR-6 traces to operator visibility. Goal 2 (exploit non-performative competitors) is an emergent property of the model's mathematics rather than a separate requirement. Goal 4 (hackathon differentiation) is validated through the simulation-based theta optimization in FR-4.

### Revisions Made During Self-Critique

- Added numerical stability requirements (NFR Reliability section) after identifying that large xi values could cause `exp(-xi*(T-t))` overflow -- a critical production safety concern.
- Added bid-ask crossing guard to FR-3 (AC-3.3) after tracing the scenario where wide performative spread plus boundary clamping produces crossed quotes.
- Added the `xi_estimates` table to Data Requirements for observability, enabling debugging of xi estimation quality over time.
- Strengthened FR-5 fallback criteria to require WARNING-level logging at each fallback step, ensuring operators can diagnose why the preferred mode is not active.
- Added rollback plan with three tiers of severity (config change, column drop, full code revert).

### Revisions Made During Validation Review (v2)

- **[CRITICAL] Config override propagation**: Added prerequisite noting that `BotLoop._process_symbol` reads from `self._settings` but `/config` writes to `app.state.config_overrides`. The bot loop must be updated to read performative parameters from config_overrides with fallback to settings.
- **[CRITICAL] Quote dataclass strategy**: Documented that `Quote` is `@dataclass(frozen=True)` with 10 fields and adding fields breaks all construction sites. Specified recommended approach (remove frozen, add optional fields with defaults) and alternative approach (wrapper dataclass).
- **[CRITICAL] Categories derivation**: Corrected FR-4, Scenario 3, dependency table, and edge cases to reflect that categories are derived from `Event.category` values in `get_events()` responses -- there is no separate `get_categories()` API method or endpoint.
- **[CRITICAL] Scanner category mapping**: Added prerequisite to extend `MarketScanner.scan()` to return `symbol_categories: dict[str, str]` mapping symbols to event categories, since the current return tuple does not include category information.
- **[MODERATE] OU regression clarification**: Simplified FR-1 regression formula to `delta_s ~ beta * s_n + alpha` with `xi = -beta / dt`. Documented why the inventory-dependent drift term is intentionally omitted.
- **[MODERATE] WebSocket broadcast prerequisite**: Added note that `quote_summary` key in WebSocket broadcast is currently unpopulated and must be fixed before FR-6 can function.
- **[MODERATE] theta_parameters updated_at**: Added `updated_at` (datetime, nullable) column to theta_parameters table definition.
- **[MODERATE] Max spread cap on performative spread**: Added explicit step in FR-3 noting that `max_spread` (default 0.10) caps performative spreads, preserving the risk management invariant.
- **[MODERATE] Tightened AC-1.1**: Split into AC-1.1a (100 points, recovery in [1.5, 2.8]) and AC-1.1b (500 points, recovery in [1.8, 2.3]).
- **[MODERATE] Migration tooling**: Added Alembic requirement to deployment steps.
- **[MODERATE] Resolved Open Question 1**: Symbols with empty-string categories use default theta (1.0, 1.0, 1.0). Moved to Assumptions section.
- **[MODERATE] R-squared quality gate**: Added r_squared < 0.1 threshold to FR-1 with fallback to default xi and WARNING log. Added AC-1.5.
