# Data Model: Performative Market Making Engine

## Entity Overview

```
Quote (modified)          ThetaParameter (new)        XiEstimateRecord (new)
+-----------------+      +-------------------+       +-------------------+
| id              |      | id                |       | id                |
| timestamp       |      | category (unique) |       | timestamp         |
| symbol          |      | theta0            |       | symbol            |
| mid_price       |      | theta1            |       | xi                |
| reservation_price|     | theta2            |       | r_squared         |
| bid_price       |      | xi_value          |       | num_trades        |
| ask_price       |      | objective_value   |       | used_default      |
| spread          |      | num_trials        |       +-------------------+
| inventory       |      | optimized_at      |
| sigma_sq        |      | created_at        |
| gamma           |      | updated_at        |
| t_minus_t       |      +-------------------+
| xi (new)        |
| theta0 (new)    |
| theta1 (new)    |
| theta2 (new)    |
| quoting_mode(new)|
+-----------------+
```

---

## Modified Entity: Quote

**Table**: `quotes`
**ORM Model**: `backend/src/db/models.py` class `Quote`

### Existing Fields (unchanged)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | Integer | no | Primary key, autoincrement |
| `timestamp` | DateTime(tz) | no | server_default=func.now() |
| `symbol` | String(128) | no | Market symbol |
| `mid_price` | Numeric(18,8) | no | Order book mid-price |
| `reservation_price` | Numeric(18,8) | no | Computed reservation price |
| `bid_price` | Numeric(18,8) | no | Posted bid price |
| `ask_price` | Numeric(18,8) | no | Posted ask price |
| `spread` | Numeric(18,8) | no | Computed spread |
| `inventory` | Numeric(18,8) | no | Net position at time of quote |
| `sigma_sq` | Numeric(18,8) | no | Estimated price variance |
| `gamma` | Numeric(18,8) | no | Risk aversion parameter used |
| `t_minus_t` | Numeric(18,8) | no | Time remaining (fraction of day) |

### New Fields (all nullable for backward compat)

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `xi` | Numeric(18,8) | yes | NULL | Estimated feedback strength used for this quote |
| `theta0` | Numeric(18,8) | yes | NULL | Theta-0 parameter (NULL for A&S mode) |
| `theta1` | Numeric(18,8) | yes | NULL | Theta-1 parameter (NULL for A&S mode) |
| `theta2` | Numeric(18,8) | yes | NULL | Theta-2 parameter (NULL for A&S mode) |
| `quoting_mode` | String(16) | yes | NULL | "as", "performative", or "theta" |

### Existing Indexes (unchanged)

- `ix_quotes_timestamp` on `timestamp`
- `ix_quotes_symbol` on `symbol`
- `ix_quotes_symbol_timestamp` on `(symbol, timestamp)`

### Validation Rules

- `xi`: When not NULL, must be in [0.01, 20.0] (enforced by xi estimator clamping)
- `theta0`, `theta1`, `theta2`: When not NULL, must be in [0.5, 2.0] (enforced by Optuna search bounds)
- `quoting_mode`: When not NULL, must be one of: "as", "performative", "theta"
- All existing field constraints remain unchanged

### State Transitions

A quote record is immutable once written. No updates. The `quoting_mode` field reflects the mode that was active when the quote was computed:
- `NULL` / `"as"`: Standard Avellaneda-Stoikov (xi, theta fields are NULL)
- `"performative"`: Vanilla performative with theta=(1,1,1) (theta fields are 1.0)
- `"theta"`: Theta-enhanced performative (theta fields reflect optimized values)

---

## New Entity: ThetaParameter

**Table**: `theta_parameters`
**ORM Model**: `backend/src/db/models.py` class `ThetaParameter`

### Fields

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | Integer | no | autoincrement | Primary key |
| `category` | String(64) | no | - | Prediction market category (unique constraint) |
| `theta0` | Numeric(18,8) | no | - | Optimized theta-0 value |
| `theta1` | Numeric(18,8) | no | - | Optimized theta-1 value |
| `theta2` | Numeric(18,8) | no | - | Optimized theta-2 value |
| `xi_value` | Numeric(18,8) | no | - | Average xi used during optimization |
| `objective_value` | Numeric(18,8) | no | - | Best negative CARA utility achieved |
| `num_trials` | Integer | no | - | Optuna trials completed |
| `optimized_at` | DateTime(tz) | no | - | When optimization completed |
| `created_at` | DateTime(tz) | no | server_default=func.now() | Row creation time |
| `updated_at` | DateTime(tz) | yes | NULL | Last re-optimization time |

### Indexes

- Unique constraint on `category` (only one theta set per category)

### Relationships

- No foreign keys. Category is a string matching `Event.category` from Gemini API.
- Queried by BotLoop `_load_theta_cache()` to populate `self._theta_cache`.
- Written by `run_theta_optimization()` in `optimizer.py` via upsert (INSERT ON CONFLICT UPDATE).

### Validation Rules

- `category`: Non-empty string, max 64 characters.
- `theta0`, `theta1`, `theta2`: Must be in [0.5, 2.0] (enforced by Optuna search bounds).
- `xi_value`: Must be positive (average of xi values used in simulation).
- `objective_value`: Negative float (negative CARA utility; lower is better).
- `num_trials`: Positive integer.

### State Transitions

```
[Not Exists] ---(first optimization)---> [Created]
                                              |
                        (re-optimization)     |
                              +---------------+
                              |
                              v
                          [Updated]
                      (updated_at set,
                       theta values replaced)
```

- A category row is created on first optimization.
- On re-optimization, the row is updated in place (upsert): `theta0`, `theta1`, `theta2`, `xi_value`, `objective_value`, `num_trials`, `optimized_at` are replaced; `updated_at` is set to current time.
- Rows are never deleted (manual cleanup only).

---

## New Entity: XiEstimateRecord

**Table**: `xi_estimates`
**ORM Model**: `backend/src/db/models.py` class `XiEstimateRecord`

### Fields

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | Integer | no | autoincrement | Primary key |
| `timestamp` | DateTime(tz) | no | server_default=func.now() | Estimation time |
| `symbol` | String(128) | no | - | Market symbol |
| `xi` | Numeric(18,8) | no | - | Estimated xi value (or default) |
| `r_squared` | Numeric(18,8) | yes | NULL | Regression fit quality (NULL if insufficient trades) |
| `num_trades` | Integer | no | - | Number of trades used in estimation |
| `used_default` | Boolean | no | False | Whether the default xi was used instead of regression |

### Indexes

- `ix_xi_estimates_timestamp` on `timestamp`
- `ix_xi_estimates_symbol` on `symbol`

### Relationships

- No foreign keys. Symbol matches the quote table's symbol column.
- Written by `BotLoop._persist_xi_estimate()` (fire-and-forget).
- This table is for observability and debugging only. It is not read by the bot loop.

### Validation Rules

- `xi`: Must be in [0.01, 20.0] (clamped by estimator).
- `r_squared`: When not NULL, must be in [0.0, 1.0].
- `num_trades`: Non-negative integer.

### State Transitions

- Append-only. Records are never updated or deleted by the application.
- Subject to data retention cleanup (existing `data_retention_days` config, default 7 days).

---

## Engine-Layer Value Objects (not persisted)

### XiEstimate (dataclass in `xi.py`)

| Field | Type | Description |
|-------|------|-------------|
| `xi` | float | Estimated or default xi value |
| `r_squared` | float or None | Regression quality (None if insufficient trades) |
| `num_trades` | int | Trades used |
| `used_default` | bool | Whether default was used |

Not a DB model. Used as the return type of `estimate_xi()` and passed to `_persist_xi_estimate()`.

### OptimizationProgress (dataclass in `optimizer.py`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `running` | bool | False | Whether optimization is in progress |
| `current_trial` | int | 0 | Current trial number (1-indexed) |
| `total_trials` | int | 0 | Total trials configured |
| `best_value` | float or None | None | Best objective value so far |
| `current_category` | str | "" | Category being optimized |
| `categories_completed` | list[str] | [] | Categories finished |
| `started_at` | datetime or None | None | When optimization started |
| `completed_at` | datetime or None | None | When optimization finished |

Not a DB model. Lives on `app.state.optimization_progress`. Written by optimizer thread, read by API status endpoint.

### Quote (modified dataclass in `quoting.py`)

| Field | Type | Default | New? |
|-------|------|---------|------|
| `bid_price` | float | (required) | No |
| `ask_price` | float | (required) | No |
| `reservation_price` | float | (required) | No |
| `spread` | float | (required) | No |
| `mid_price` | float | (required) | No |
| `inventory` | float | (required) | No |
| `sigma_sq` | float | (required) | No |
| `gamma` | float | (required) | No |
| `t_minus_t` | float | (required) | No |
| `k` | float | (required) | No |
| `xi` | float or None | None | Yes |
| `theta0` | float or None | None | Yes |
| `theta1` | float or None | None | Yes |
| `theta2` | float or None | None | Yes |
| `quoting_mode` | str or None | None | Yes |

Changed from `@dataclass(frozen=True)` to `@dataclass`. New fields have `None` defaults for backward compatibility with all existing construction sites.

### MarketState (modified dataclass in `book.py`)

| Field | Type | Default | New? |
|-------|------|---------|------|
| `mid_price` | float | (required) | No |
| `best_bid` | float | (required) | No |
| `best_ask` | float | (required) | No |
| `spread` | float | (required) | No |
| `trade_prices` | list[float] | [] | No |
| `trade_timestamps` | list[float] | [] | Yes |

Remains `@dataclass(frozen=True)`. New field has default for backward compatibility.
