# Minimal-Changes Architecture Design: Performative Market Making

## Problem Decomposition

To add performative quoting with the absolute minimum disruption to the existing codebase, I will solve these subproblems in order:

| # | Subproblem | Depends On | Why This Order |
|---|------------|------------|----------------|
| 1 | Requirements Clarification | - | Foundation for all decisions |
| 2 | Pattern Discovery | 1 | Need requirements to identify relevant patterns |
| 3 | Design Approaches | 1, 2 | Need requirements + patterns to generate valid options |
| 4 | Architecture Decision | 1, 2, 3 | Select from approaches using patterns as criteria |
| 5 | Component Design | 1, 2, 4 | Implement decision following discovered patterns |
| 6 | Integration Mapping | 2, 5 | Connect new components to existing code |
| 7 | Data Flow | 5, 6 | Trace data through integrated components |
| 8 | Build Sequence | 5, 6, 7 | Order implementation based on dependencies |

---

## Step 2.1: Requirements Clarification

*Foundation for all decisions.*

**Functional requirements (from spec.md FR-1 through FR-6)**:

1. Estimate xi per symbol per cycle from trade prices via OU regression (OLS on up to 100 trades).
2. Replace A&S reservation price and spread with performative formulas, using xi and theta parameters.
3. Learn category-specific theta vectors via Optuna, triggered via API or 24h auto-schedule.
4. Support three quoting modes ("as", "performative", "theta") switchable via `/config` with fallback chain.
5. Persist xi, theta, quoting_mode on quote records; new tables for theta_parameters and xi_estimates.
6. Extend dashboard WebSocket and REST responses with xi, theta, quotingMode fields.

**Non-functional requirements**:

- Xi estimation < 5ms per symbol.
- Quote computation < 1ms per symbol.
- Theta optimization non-blocking (background thread).
- No new external dependencies beyond `numpy` and `optuna`.
- Backward compatible: nullable new columns, additive API fields.

**Key constraints (resolved decisions)**:

- Quote dataclass: remove `frozen=True`, add optional fields with defaults.
- Global quoting mode only (no per-symbol overrides).
- Config overrides: pass dict reference to BotLoop.
- Optuna: `asyncio.to_thread()`.
- OLS: `np.linalg.lstsq`.
- Numerical: `xi < 1e-6` threshold with Taylor fallback.

**Feeds into**: Steps 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8.

---

## Step 2.2: Codebase Pattern Analysis

*Using requirements from Step 2.1...*

### Discovered Patterns

| Pattern | Evidence | File:Line |
|---------|----------|-----------|
| **Stateless engine functions** | `compute_quote()` is a pure function taking all inputs as args, returning a frozen dataclass. `estimate_variance()` follows the same pattern. | `quoting.py:25-93`, `quoting.py:96-108` |
| **Single `quoting.py` module** | All quoting logic (dataclass + functions) lives in one file. No subpackage structure for engine. | `backend/src/engine/quoting.py` |
| **BotLoop orchestrates** | `_process_symbol()` calls engine functions in sequence: fetch state -> compute quote -> risk check -> place. | `loop.py:255-372` |
| **Settings via pydantic sub-models** | Each concern has its own `BaseSettings` subclass composed under `Settings`. | `config.py:32-117` |
| **Config overrides as mutable dict** | `app.state.config_overrides` is a plain dict written by API, but BotLoop does NOT read it (bug). | `main.py:45`, `router.py:540-596` |
| **Scanner returns tuple** | `scan()` returns `(symbols, titles, expiry_hours)` tuple. Category data available but not returned. | `scanner.py:26-101` |
| **MarketState is frozen dataclass** | Trade prices available as `list[float]`, but timestamps are discarded in `book.py:54`. | `book.py:13-21`, `book.py:53-54` |
| **Quote DB model mirrors dataclass** | `QuoteRecord` in `models.py:17-41` has the same fields as the `Quote` dataclass. | `models.py:17-41` |
| **DB schema via create_all** | No Alembic migrations yet; `Base.metadata.create_all` in `database.py:52-53`. | `database.py:49-54` |
| **WebSocket broadcast via on_tick callback** | BotLoop builds market_list dicts and calls `self.on_tick()`. | `loop.py:200-237` |
| **WIDEN_SPREAD reconstructs Quote** | The risk path at `loop.py:333-344` constructs a new `Quote` by copying all fields -- adding new required fields would break this. | `loop.py:333-344` |
| **Trade model has timestamps** | `Trade.timestamp` (Unix epoch int) is available from Gemini API but discarded by `OrderBookMonitor`. | `models.py:34`, `book.py:54` |

### Key Abstraction Boundaries

```
[Scanner] --> symbols, titles, expiry_hours  (add: categories)
[OrderBookMonitor] --> MarketState           (add: trade_timestamps)
[quoting.py] --> Quote                       (add: performative functions)
[BotLoop._process_symbol] --> orchestration  (add: mode routing)
[config.py] --> Settings                     (add: PerformativeSettings)
[models.py] --> DB models                    (add: columns + tables)
[router.py] --> API                          (add: endpoints + fields)
```

**Feeds into**: Steps 2.3, 2.4, 2.5, 2.6.

---

## Step 2.3: Design Approaches

*Using requirements from Step 2.1 and patterns from Step 2.2...*

### Approach 1: Inline Everything in Existing Files (Probability: 0.85)

Add `compute_performative_quote()` and `estimate_xi()` to `quoting.py`. Add `PerformativeSettings` to `config.py`. Add theta/xi DB models to `models.py`. Modify `_process_symbol()` in `loop.py` for mode routing. Add optimization endpoint logic directly in `router.py`.

- **Pattern alignment**: Follows existing single-module patterns. `quoting.py` already holds all quoting logic.
- **Pros**: Fewest new files (potentially zero). Maximum code locality. Follows existing structure exactly.
- **Cons**: `quoting.py` grows significantly. Optimizer logic in `router.py` is heavy for a router file. `loop.py` gets more complex.

### Approach 2: New `performative.py` Engine Module + Minimal Changes Elsewhere (Probability: 0.82)

Create one new file `backend/src/engine/performative.py` containing `estimate_xi()`, `compute_performative_quote()`, and `ThetaOptimizer` class. Modify existing files minimally: `quoting.py` (unfroze Quote), `loop.py` (mode routing in `_process_symbol`), `config.py` (add settings), `models.py` (add columns/tables), `router.py` (add endpoints), `scanner.py` (add categories), `book.py` (add timestamps).

- **Pattern alignment**: Partially follows pattern (engine components are in separate files: `quoting.py`, `risk.py`, `scanner.py`, `sweep.py`).
- **Pros**: Clean separation of new logic. Existing files change minimally. One new file vs zero.
- **Cons**: Optimizer class in engine module is unusual (existing engine components are stateless).

### Approach 3: Modify `quoting.py` + New `optimizer.py` (Probability: 0.80)

Keep xi estimation and performative formulas in `quoting.py` (they are pure computation, matching the module's purpose). Put the stateful Optuna optimizer in a separate `backend/src/engine/optimizer.py`. This separates pure math from async/stateful optimization.

- **Pattern alignment**: Strong. Pure computation stays in `quoting.py` (existing pattern). Stateful async logic gets its own module (like `scanner.py` is stateful).
- **Pros**: Natural separation of concerns. `quoting.py` remains the "all quoting math" module. Optimizer is naturally different (async, stateful, background).
- **Cons**: Two new function clusters in `quoting.py` makes it longer. One new file for optimizer.

### Approach 4: Plugin Architecture with Strategy Pattern (Probability: 0.06)

Define a `QuotingStrategy` protocol/ABC with `compute()` method. Implement `ASStrategy` and `PerformativeStrategy`. BotLoop selects strategy based on mode config. Each strategy lives in its own file.

- **Pattern alignment**: Poor. Existing codebase uses plain functions, not strategy objects. Introduces a new abstraction layer.
- **Pros**: Clean extensibility for future quoting models. Strong OOP separation.
- **Cons**: Over-engineered for three modes where two share 90% of logic. Breaks existing patterns. Multiple new files. Large diff.

### Approach 5: Microservice Separation (Probability: 0.03)

Run theta optimization as a separate service communicating via the database. Keep quoting inline.

- **Pattern alignment**: None. Single-process architecture throughout.
- **Pros**: Full isolation of optimization workload.
- **Cons**: Massive operational overhead. Docker Compose changes. New service boilerplate. Completely disproportionate.

### Approach 6: Event-Sourced Quoting Pipeline (Probability: 0.02)

Replace the imperative `_process_symbol` with an event pipeline where each step publishes events consumed by the next. Xi estimation, quoting, risk checks become independent event handlers.

- **Pattern alignment**: None. Existing code is imperative sequential.
- **Pros**: Theoretically more testable and composable.
- **Cons**: Complete rewrite of the core loop. Enormous diff. Introduces event bus dependency. Antithetical to "minimal changes."

**Feeds into**: Step 2.4.

---

## Step 2.4: Architecture Decision

*Using approaches from Step 2.3, patterns from Step 2.2, and requirements from Step 2.1...*

**Chosen approach: Approach 3 -- Modify `quoting.py` + New `optimizer.py`**

**Rationale**:

1. **Pattern alignment** (from Step 2.2): The codebase treats `quoting.py` as the "all quoting math" module. Xi estimation and performative formulas are pure computation -- they belong there. The optimizer is async and stateful, which is a different concern, matching how `scanner.py` (async, stateful) is separate from `quoting.py` (pure math).

2. **Minimal diff**: Only one new file (`optimizer.py`). All other changes are modifications to existing files. Approach 1 avoids the new file but stuffs optimizer logic into `router.py`, which is worse. Approach 2 creates a catch-all `performative.py` that mixes pure math with stateful optimization.

3. **Backward compatibility** (from Step 2.1 constraints): Removing `frozen=True` from `Quote` and adding optional fields with defaults means zero changes to existing construction sites (`loop.py:333-344`, any tests).

4. **Against Approach 4**: The strategy pattern introduces abstractions not present in the codebase. The mode routing is a simple if/elif in `_process_symbol` -- no need for polymorphism with only three modes where two call the same function with different theta defaults.

**Trade-offs accepted**:

- `quoting.py` grows from 109 lines to approximately 200 lines. Acceptable -- it remains a single cohesive module of pure quoting math.
- One new file (`optimizer.py`) adds to the engine directory. Acceptable -- follows the existing pattern of one file per engine concern.

**Feeds into**: Steps 2.5, 2.6, 2.7, 2.8.

---

## Step 2.5: Component Design

*Using chosen approach from Step 2.4 and patterns from Step 2.2...*

### Modified Files (7 files)

#### 1. `backend/src/engine/quoting.py`

**Current**: `Quote` dataclass (frozen), `compute_quote()`, `estimate_variance()`.

**Changes**:
- Remove `frozen=True` from `@dataclass` decorator on `Quote`.
- Add 5 optional fields to `Quote`: `xi: float | None = None`, `theta0: float | None = None`, `theta1: float | None = None`, `theta2: float | None = None`, `quoting_mode: str | None = None`.
- Add constant `XI_EPSILON = 1e-6`.
- Add function `estimate_xi(trade_prices: list[float], trade_timestamps: list[int], xi_default: float, xi_min_trades: int, xi_clamp_min: float, xi_clamp_max: float, r_squared_threshold: float) -> tuple[float, float | None]` -- returns `(xi, r_squared)`. Uses `np.linalg.lstsq`. Falls back to `xi_default` when insufficient trades or poor fit.
- Add function `compute_performative_quote(mid_price, inventory, gamma, sigma_sq, t_minus_t, k, xi, theta0, theta1, theta2, q_ref, max_spread, best_bid, best_ask) -> Quote` -- applies performative reservation price and spread formulas. Returns a `Quote` with `xi`, `theta0/1/2`, `quoting_mode` fields populated.
- Add two private helper functions: `_delta_epsilon(xi, T) -> float` and `_inv_correction(xi, T) -> float` with Taylor fallback at `XI_EPSILON`.

**Responsibilities**: All pure quoting math -- A&S, performative, xi estimation.

**Dependencies**: `numpy` (new import, for `estimate_xi` only), `math` (existing).

#### 2. `backend/src/engine/loop.py`

**Current**: `BotLoop` class orchestrating the main cycle.

**Changes**:
- Add `config_overrides: dict` parameter to `__init__()` signature. Store as `self._config_overrides`. This is a mutable dict reference shared with `app.state.config_overrides`.
- Add `self._symbol_categories: dict[str, str] = {}` state field.
- Modify `_maybe_scan()`: unpack 4th return value `symbol_categories` from scanner, store in `self._symbol_categories`.
- Modify `_process_symbol()`:
  - Read `quoting_mode` from `self._config_overrides.get("quoting_mode", "theta")`.
  - Read performative params from `self._config_overrides` with fallback to `self._settings.performative.*`.
  - If mode is `"as"`: call existing `compute_quote()` as-is. Set `quote.quoting_mode = "as"`.
  - If mode is `"performative"` or `"theta"`: call `estimate_xi()` with `state.trade_prices` and `state.trade_timestamps`. Then determine theta: if mode is `"theta"`, look up category theta from `self._theta_cache`. If not found, log warning, fall back to `(1,1,1)`. Call `compute_performative_quote()`.
  - If performative computation returns non-finite results, fall back to `compute_quote()` and log warning.
- Add `self._theta_cache: dict[str, tuple[float, float, float]] = {}` and method `_refresh_theta_cache()` that reads from DB periodically.
- Modify `_persist_quote()`: pass `xi`, `theta0`, `theta1`, `theta2`, `quoting_mode` to `QuoteRecord`.
- Modify `_persist_xi_estimate()`: new helper to write `XiEstimate` records.
- Modify WebSocket broadcast (market_list builder around line 205): add `xi`, `theta0`, `theta1`, `theta2`, `quotingMode` fields.
- Import `estimate_xi`, `compute_performative_quote` from `quoting.py`.

**Dependencies**: `quoting.py` (extended), `models.py` (extended), `config.py` (extended).

#### 3. `backend/src/config.py`

**Current**: `Settings` with sub-models `GeminiSettings`, `ASSettings`, `BotSettings`, `RiskSettings`, `DatabaseSettings`, `AppSettings`.

**Changes**:
- Add `PerformativeSettings` class (new sub-model):
  - `xi_default: float = 0.5`
  - `xi_min_trades: int = 15`
  - `xi_clamp_min: float = 0.01`
  - `xi_clamp_max: float = 20.0`
  - `r_squared_threshold: float = 0.1`
  - `q_ref: float = 0.0`
  - `quoting_mode: str = "theta"` (default mode)
  - `theta_optimization_trials: int = 100`
  - `theta_optimization_simulations: int = 100`
  - `theta_auto_optimize_hours: int = 24`
  - `env_prefix = "PERF_"`
- Add `performative: PerformativeSettings = Field(default_factory=PerformativeSettings)` to `Settings`.

**Dependencies**: None new.

#### 4. `backend/src/db/models.py`

**Current**: `Quote`, `OrderRecord`, `PositionSnapshot`, `PnlSnapshot` models.

**Changes**:
- Add 5 nullable columns to `Quote` (the DB model):
  - `xi: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)`
  - `theta0: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)`
  - `theta1: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)`
  - `theta2: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)`
  - `quoting_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)`
- Add `ThetaParameter` model (new table `theta_parameters`):
  - `id`, `category` (unique), `theta0`, `theta1`, `theta2`, `xi_value`, `objective_value`, `num_trials`, `optimized_at`, `created_at`, `updated_at`.
  - Index on `category`.
- Add `XiEstimate` model (new table `xi_estimates`):
  - `id`, `timestamp`, `symbol`, `xi`, `num_trades`, `r_squared`.
  - Index on `(symbol, timestamp)`.

**Dependencies**: None new (uses existing SQLAlchemy imports).

#### 5. `backend/src/engine/scanner.py`

**Current**: `scan()` returns `tuple[list[str], dict[str, str], dict[str, float]]`.

**Changes**:
- Extend `scan()` return type to `tuple[list[str], dict[str, str], dict[str, float], dict[str, str]]` -- 4th element is `symbol_categories: dict[str, str]`.
- In the `for event in events` / `for contract in event.contracts` loop, when a symbol is selected, also record `symbol_categories[symbol] = event.category`.
- Return the 4th element.

**Dependencies**: None new.

#### 6. `backend/src/engine/book.py`

**Current**: `MarketState` with `trade_prices: list[float]`. Timestamps discarded.

**Changes**:
- Add `trade_timestamps: list[int] = field(default_factory=list)` to `MarketState` dataclass.
- In `get_market_state()`, change `trade_prices = [float(t.price) for t in trades]` to also capture timestamps: `trade_timestamps = [t.timestamp for t in trades]`.
- Pass `trade_timestamps=trade_timestamps` to `MarketState` constructor.

**Dependencies**: None new.

#### 7. `backend/src/api/router.py`

**Current**: REST endpoints for status, markets, pnl, positions, config, sweep, bot start/stop.

**Changes**:
- Extend `ConfigUpdateRequest` with new optional fields: `quoting_mode: str | None = None`, `xi_default: float | None = None`, `xi_min_trades: int | None = None`, `xi_clamp_min: float | None = None`, `xi_clamp_max: float | None = None`, `q_ref: float | None = None`.
- Extend `MarketSummary` with nullable fields: `xi: float | None = None`, `theta0: float | None = None`, `theta1: float | None = None`, `theta2: float | None = None`, `quotingMode: str | None = None`.
- Extend `QuoteHistoryEntry` similarly.
- Update `update_config()` effective config dict to include performative params.
- Add `POST /optimize/theta` endpoint: triggers `ThetaOptimizer.run()` from `optimizer.py`.
- Add `GET /optimize/theta/status` endpoint: reads from `app.state.optimization_progress`.
- Update `get_markets()` and `get_market_detail()` to include new fields from DB (nullable, backward-compatible).

**Dependencies**: `optimizer.py` (new).

### New Files (1 file)

#### 8. `backend/src/engine/optimizer.py`

**Responsibilities**: Stateful theta optimization via Optuna. Background async task management. Progress reporting.

**Interface**:
- `OptimizationProgress` dataclass: `running`, `category`, `current_trial`, `total_trials`, `best_value`, `started_at`, `completed_at`.
- `ThetaOptimizer` class:
  - `__init__(self, session_factory, settings, progress)` -- takes DB session factory, performative settings, progress state.
  - `async run(self, categories: list[str], symbol_categories: dict[str, str])` -- runs optimization for given categories.
  - Internal: `_objective(trial, trade_data, xi, settings) -> float` -- single Optuna trial (synchronous, runs 100 simulations).
  - Internal: `_simulate_session(theta, trade_data, xi, settings) -> float` -- single simulation, returns CARA utility.
  - Uses `asyncio.to_thread(study.optimize, ...)` to offload CPU work.
  - After completion, upserts results to `theta_parameters` table.

**Dependencies**: `optuna`, `numpy`, `asyncio`, `sqlalchemy`, `quoting.py` (for `compute_performative_quote`).

### Frontend Changes (2 files)

#### 9. `frontend/src/lib/types.ts`

**Changes**: Add optional fields to `MarketData` interface:
- `xi?: number`
- `theta0?: number`
- `theta1?: number`
- `theta2?: number`
- `quotingMode?: string`

#### 10. Frontend display components (minimal)

Display `quotingMode` in status bar, `xi`/theta values where MarketData is rendered. These are additive -- existing renders are unaffected because new fields are optional.

**Feeds into**: Steps 2.6, 2.7, 2.8.

---

## Step 2.6: Integration Mapping

*Using component design from Step 2.5 and patterns from Step 2.2...*

### Exact Integration Points

#### `main.py` line 62: Pass config_overrides to BotLoop

**Current**:
```python
bot_loop = BotLoop(
    settings=settings,
    client=gemini_client,
    session_factory=get_session_factory(),
)
```

**After**: Add `config_overrides=app.state.config_overrides` parameter.

#### `loop.py` line 34: Accept config_overrides

**Current**: `__init__` takes `settings`, `client`, `session_factory`.

**After**: Add `config_overrides: dict[str, Any] = None` parameter. Default to `{}` if None. Store as `self._config_overrides`.

#### `loop.py` line 20: Add imports

**Current**: `from src.engine.quoting import Quote, compute_quote, estimate_variance`.

**After**: Also import `estimate_xi, compute_performative_quote`.

#### `loop.py` line 298-316: Mode routing in _process_symbol

**Current**: Always calls `compute_quote()` at line 306.

**After**: Read mode from `self._config_overrides.get("quoting_mode", self._settings.performative.quoting_mode)`. Branch:
- `"as"` -> existing `compute_quote()` call unchanged.
- `"performative"` / `"theta"` -> call `estimate_xi()` with `state.trade_prices, state.trade_timestamps`, then call `compute_performative_quote()`.

#### `loop.py` line 382: Extend _maybe_scan

**Current**: `symbols, titles, expiry_hours = await self._scanner.scan()` at line 383.

**After**: `symbols, titles, expiry_hours, categories = await self._scanner.scan()`. Store `self._symbol_categories.update(categories)`.

#### `loop.py` line 434: Extend _persist_quote

**Current**: `QuoteRecord(...)` at line 434 passes 10 fields.

**After**: Add `xi=quote.xi`, `theta0=quote.theta0`, `theta1=quote.theta1`, `theta2=quote.theta2`, `quoting_mode=quote.quoting_mode`.

#### `loop.py` ~line 205: Extend WebSocket broadcast

**Current**: market_list dicts have keys up to `timeRemaining`.

**After**: Add `"xi"`, `"theta0"`, `"theta1"`, `"theta2"`, `"quotingMode"` keys (nullable).

#### `scanner.py` line 26 and 84: Extend return

**Current**: Returns 3-tuple. Line 84: `selected.append(symbol)` plus `symbol_titles[symbol] = event.title`.

**After**: Also `symbol_categories[symbol] = event.category`. Return 4-tuple.

#### `book.py` line 54: Capture timestamps

**Current**: `trade_prices = [float(t.price) for t in trades]`.

**After**: Also `trade_timestamps = [t.timestamp for t in trades]`. Pass to `MarketState`.

#### `router.py` line 540: Extend /config endpoint

**Current**: `ConfigUpdateRequest` has 10 fields. `effective` dict has 10 keys.

**After**: Add `quoting_mode` and performative params to both.

#### `router.py`: Add two new endpoints

**After `run_parameter_sweep`**:
- `@router.post("/optimize/theta")` -- reads `app.state.bot_loop._symbol_categories`, creates/runs `ThetaOptimizer`, stores progress on `app.state.optimization_progress`.
- `@router.get("/optimize/theta/status")` -- reads `app.state.optimization_progress`.

**Feeds into**: Steps 2.7, 2.8.

---

## Step 2.7: Data Flow

*Using components from Step 2.5 and integration points from Step 2.6...*

### Flow 1: Normal Performative Quoting Cycle (per symbol)

```
1. BotLoop._process_symbol(symbol, inventories)
   |
2. state = OrderBookMonitor.get_market_state(symbol)
   --> MarketState(mid_price, best_bid, best_ask, spread,
                   trade_prices=[...],        # existing
                   trade_timestamps=[...])     # NEW
   |
3. Read mode from self._config_overrides.get("quoting_mode", settings.performative.quoting_mode)
   |
   +-- mode == "as" ----> compute_quote(...) as today, set quote.quoting_mode = "as" --> step 7
   |
4. (xi, r_squared) = estimate_xi(
       state.trade_prices, state.trade_timestamps,
       xi_default, xi_min_trades, xi_clamp_min, xi_clamp_max, r_squared_threshold
   )
   --> If <15 trades: xi = xi_default, r_squared = None
   --> If r_squared < 0.1: xi = xi_default, log warning
   --> Else: xi from OLS, clamped to [0.01, 20.0]
   |
5. Theta lookup:
   +-- mode == "theta" --> category = self._symbol_categories.get(symbol, "")
   |                       theta = self._theta_cache.get(category, (1.0, 1.0, 1.0))
   |                       if theta == (1.0,1.0,1.0) and category not in cache: log warning, effective_mode = "performative"
   +-- mode == "performative" --> theta = (1.0, 1.0, 1.0), effective_mode = "performative"
   |
6. quote = compute_performative_quote(
       mid_price=state.mid_price, inventory=inventory,
       gamma=gamma, sigma_sq=sigma_sq, t_minus_t=t_minus_t, k=k,
       xi=xi, theta0=theta[0], theta1=theta[1], theta2=theta[2],
       q_ref=q_ref, max_spread=max_spread,
       best_bid=state.best_bid, best_ask=state.best_ask
   )
   --> Quote with all fields populated including xi, theta*, quoting_mode
   |
   +-- If not math.isfinite(quote.reservation_price):
       log warning, fallback to compute_quote(), set quoting_mode = "as"
   |
7. Risk checks (unchanged) --> possibly WIDEN_SPREAD
   |
   +-- WIDEN_SPREAD: reconstruct Quote with widened spread
       (works because Quote is no longer frozen;
        or: rebuild with all fields including new optional ones)
   |
8. Persist: QuoteRecord with xi, theta0-2, quoting_mode (nullable columns)
   Persist: XiEstimate record (fire-and-forget)
   |
9. WebSocket broadcast: market_list dict includes xi, theta*, quotingMode
```

### Flow 2: Theta Optimization (triggered via API)

```
1. POST /optimize/theta
   |
2. Read categories from bot_loop._symbol_categories (distinct values)
   |
3. Create ThetaOptimizer with session_factory, settings
   Set app.state.optimization_progress.running = True
   |
4. asyncio.create_task(optimizer.run(categories, symbol_categories))
   Return 202 immediately
   |
5. For each category:
   a. Gather trade data from DB (or live API) for all symbols in category
   b. study = optuna.create_study(direction="minimize")
   c. await asyncio.to_thread(study.optimize, objective_fn, n_trials=100,
                               callbacks=[progress_callback])
      |
      objective_fn(trial):
        theta0 = trial.suggest_float("theta0", 0.5, 2.0)
        theta1 = trial.suggest_float("theta1", 0.5, 2.0)
        theta2 = trial.suggest_float("theta2", 0.5, 2.0)
        total_utility = 0
        for sim in range(100):
            utility += simulate_session(theta, trade_data, xi)
        return -total_utility / 100    # minimize negative CARA utility
      |
   d. Upsert ThetaParameter row for this category
   e. Update progress
   |
6. optimization_progress.running = False
   bot_loop._theta_cache refreshed on next cycle
```

### Flow 3: Mode Switch via API

```
1. POST /config {"quoting_mode": "as"}
   |
2. router.update_config() -> overrides["quoting_mode"] = "as"
   (same mutable dict referenced by bot_loop._config_overrides)
   |
3. Next bot cycle: _process_symbol reads overrides["quoting_mode"] == "as"
   --> calls compute_quote() (existing A&S path)
   |
4. POST /config {"quoting_mode": "theta"}
   |
5. Next cycle: reads "theta", resumes performative quoting
```

**Feeds into**: Step 2.8.

---

## Step 2.8: Build Sequence

*Using all previous steps...*

### Phase 1: Foundation (no behavioral change)

These changes are backward-compatible and can be merged independently.

- [ ] **1a.** `config.py`: Add `PerformativeSettings` sub-model with all fields and defaults. Add `performative` field to `Settings`. No existing behavior changes.
- [ ] **1b.** `quoting.py`: Remove `frozen=True` from `Quote` dataclass. Add 5 optional fields with `None` defaults. Existing construction sites unchanged.
- [ ] **1c.** `book.py`: Add `trade_timestamps: list[int]` to `MarketState`. Capture timestamps in `get_market_state()`. Existing consumers unaffected (they don't use timestamps).
- [ ] **1d.** `scanner.py`: Extend `scan()` to return 4-tuple with `symbol_categories`. Update the single call site in `loop.py:383` to unpack 4 values and store `self._symbol_categories`.
- [ ] **1e.** `models.py`: Add 5 nullable columns to `Quote` DB model. Add `ThetaParameter` and `XiEstimate` models.
- [ ] **1f.** `db/database.py`: `create_all` will auto-create new columns/tables on next startup (existing behavior for dev). For production, generate Alembic migration (see research.md Section 4).
- [ ] **1g.** `main.py`: Pass `config_overrides=app.state.config_overrides` to `BotLoop()`.
- [ ] **1h.** `loop.py`: Accept `config_overrides` parameter in `__init__`. Store as `self._config_overrides`. Add `self._symbol_categories`, `self._theta_cache` state fields.
- [ ] **1i.** `frontend/src/lib/types.ts`: Add optional `xi`, `theta0`, `theta1`, `theta2`, `quotingMode` to `MarketData`.
- [ ] **1j.** Update test for `Quote` frozen check (if exists) to reflect `frozen=True` removal.

**Verification**: Bot starts, runs existing A&S quoting unchanged. New DB columns exist. New config fields have defaults. Dashboard renders without error.

### Phase 2: Xi Estimation + Performative Quoting

Depends on: Phase 1 complete.

- [ ] **2a.** `quoting.py`: Add `XI_EPSILON` constant, `_delta_epsilon()`, `_inv_correction()` helper functions with Taylor fallback.
- [ ] **2b.** `quoting.py`: Add `estimate_xi()` function using `np.linalg.lstsq`.
- [ ] **2c.** `quoting.py`: Add `compute_performative_quote()` function implementing performative reservation price and spread formulas.
- [ ] **2d.** `loop.py`: Import `estimate_xi`, `compute_performative_quote`. Modify `_process_symbol()` to implement mode routing: read `quoting_mode` from overrides, branch to A&S or performative path.
- [ ] **2e.** `loop.py`: Extend `_persist_quote()` to pass xi, theta, quoting_mode to `QuoteRecord`.
- [ ] **2f.** `loop.py`: Add `_persist_xi_estimate()` helper; call it after xi estimation.
- [ ] **2g.** `loop.py`: Extend WebSocket broadcast market_list entries with xi, theta, quotingMode fields.
- [ ] **2h.** `router.py`: Extend `ConfigUpdateRequest` with `quoting_mode` and performative fields. Extend `update_config()` effective config. Extend `MarketSummary` and `QuoteHistoryEntry` with new nullable fields. Update `get_markets()` and `get_market_detail()`.

**Verification**: Set `quoting_mode = "performative"` via `/config`. Observe quotes differ from A&S. Set `quoting_mode = "as"` and observe identical quotes to before. Check xi_estimates table has rows. Check quote records have xi and quoting_mode populated. Dashboard shows new fields.

### Phase 3: Theta Optimization

Depends on: Phase 2 complete.

- [ ] **3a.** Create `backend/src/engine/optimizer.py`: `OptimizationProgress` dataclass, `ThetaOptimizer` class with `run()`, `_objective()`, `_simulate_session()` methods.
- [ ] **3b.** `router.py`: Add `POST /optimize/theta` endpoint. Initialize `ThetaOptimizer`, launch via `asyncio.create_task`. Store progress on `app.state.optimization_progress`.
- [ ] **3c.** `router.py`: Add `GET /optimize/theta/status` endpoint. Read from `app.state.optimization_progress` and `theta_parameters` table.
- [ ] **3d.** `loop.py`: Add `_refresh_theta_cache()` method that queries `theta_parameters` table. Call it periodically (e.g., every scanner cycle) in `_maybe_scan()`.
- [ ] **3e.** `loop.py`: In the `"theta"` mode branch of `_process_symbol()`, look up `self._theta_cache` for the symbol's category. Implement fallback to `(1.0, 1.0, 1.0)` with warning log.

**Verification**: Trigger `POST /optimize/theta`. Check `GET /optimize/theta/status` shows progress. After completion, check `theta_parameters` table has rows. Set mode to `"theta"`, observe quotes use category-specific theta values. Quoting continues uninterrupted during optimization.

### Phase 4: Dashboard Polish (optional, low priority)

Depends on: Phase 2 complete.

- [ ] **4a.** Display quoting mode in frontend status bar.
- [ ] **4b.** Display xi and theta values in market detail views.
- [ ] **4c.** Add optimization trigger button and status display.

---

## Key Architectural Decisions

| Challenge | Solution | Trade-offs | Pattern Reference |
|-----------|----------|------------|-------------------|
| Where to put performative math | In `quoting.py` alongside existing A&S math | File grows ~100 lines; but maintains single-module-for-all-quoting-math pattern | `quoting.py` is the quoting math module (Step 2.2) |
| Where to put optimizer | New `optimizer.py` in engine/ | One new file; but separates stateful async from pure math | `scanner.py` precedent for stateful engine components (Step 2.2) |
| Quote dataclass mutation | Remove `frozen=True`, add optional fields | Loses immutability guarantee; but Quote is never hashed/stored in sets, and all construction sites are preserved | WIDEN_SPREAD reconstruction at `loop.py:333-344` (Step 2.2) |
| Config override propagation | Pass mutable dict reference to BotLoop | Implicit coupling via shared mutable state; but minimal change, matches existing `app.state.config_overrides` pattern | `main.py:45`, `router.py:556-558` (Step 2.2) |
| Trade timestamps for xi | Add field to `MarketState`, capture in `book.py` | Slight memory increase; but data already fetched, just not retained | `book.py:53-54` discards timestamps (Step 2.2) |
| Mode routing location | if/elif in `_process_symbol()` | Adds branching to the main loop; but avoids strategy pattern over-engineering for 3 modes | `_process_symbol` is already the orchestration point (Step 2.2) |
| Theta cache | In-memory dict on BotLoop, refreshed periodically | Stale for up to one scanner cycle; but optimization runs at most daily, and staleness is bounded | Scanner pattern: refresh on cadence (Step 2.2) |
| Database schema changes | Nullable columns + `create_all` for dev, Alembic for prod | `create_all` only adds tables/columns, cannot alter; Alembic needed for production | `database.py:49-54` current pattern (Step 2.2) |

---

## Critical Details

### Error Handling

- **Xi estimation failure** (numpy error, empty arrays): catch `Exception` in `estimate_xi`, return `(xi_default, None)`, log warning. Never let xi failure crash the cycle.
- **Performative math produces NaN/Inf**: guard with `math.isfinite()` check on reservation_price and spread after computation. Fall back to `compute_quote()` with `quoting_mode = "as"`.
- **Theta optimization failure**: catch `Exception` in `ThetaOptimizer.run()`, set `progress.running = False`, log error. Live quoting unaffected (uses cached theta or defaults).
- **Scanner 4-tuple unpacking**: if scanner fails, existing exception handler at `loop.py:388` catches it and keeps previous symbol list + categories.

### State Management

- **Theta cache**: `dict[str, tuple[float, float, float]]` on BotLoop. Refreshed in `_maybe_scan()` alongside symbol refresh. Key is category string, value is `(theta0, theta1, theta2)`.
- **Optimization progress**: `OptimizationProgress` dataclass on `app.state`. Single writer (optimizer callback thread), single reader (API handler). GIL provides atomicity for attribute reads/writes -- no lock needed.
- **Config overrides**: same mutable dict instance shared between `app.state` and `BotLoop`. API writes, BotLoop reads. Thread-safe for dict key operations under GIL.

### Testing Strategy

- **Unit tests for `estimate_xi()`**: synthetic OU data with known xi, verify recovery within tolerance. Edge cases: flat prices, <15 trades, noisy data with low r-squared.
- **Unit tests for `compute_performative_quote()`**: verify degeneration to A&S when xi->0 (within 1e-6 tolerance). Verify spread >= A&S spread. Verify clamping. Verify NaN guard.
- **Unit tests for `_delta_epsilon()` and `_inv_correction()`**: Taylor vs direct at boundary (xi = 1e-6, 1e-7, 1e-5). Verify continuity.
- **Integration test for mode switching**: start in "as" mode, switch to "performative" via `/config`, verify quotes change.
- **Integration test for fallback chain**: "theta" mode with no theta_parameters rows -> falls back to performative. With xi estimation failure -> falls back to A&S.

### Performance

- `estimate_xi()`: ~50-100us for 100 trades (numpy lstsq on 100x2 matrix). Well within 5ms budget.
- `compute_performative_quote()`: ~1-5us (a few `math.exp` and arithmetic operations). Well within 1ms budget.
- Theta cache refresh: single SELECT query per scanner cycle (every 300s). Negligible.
- Theta optimization: 5-10 minutes per category, fully offloaded via `asyncio.to_thread()`. Zero impact on event loop.

### Security

- No new credentials or external services.
- Optuna runs in-process, in-memory storage. No new network exposure.
- New `/optimize/theta` endpoint has no auth (matching existing `/bot/start` pattern). Add auth if deploying beyond local use.
- Config overrides are server-side only; no client-injectable parameters reach math functions without clamping.

---

## File Change Summary

| File | Change Type | Estimated Diff Size |
|------|-------------|---------------------|
| `backend/src/engine/quoting.py` | Modify | +100 lines (xi estimation, performative formulas, Quote fields) |
| `backend/src/engine/loop.py` | Modify | +80 lines (mode routing, config overrides, theta cache, persist changes) |
| `backend/src/config.py` | Modify | +20 lines (PerformativeSettings sub-model) |
| `backend/src/db/models.py` | Modify | +45 lines (Quote columns, ThetaParameter, XiEstimate) |
| `backend/src/api/router.py` | Modify | +80 lines (extended schemas, 2 new endpoints, config extension) |
| `backend/src/engine/scanner.py` | Modify | +5 lines (4th return value) |
| `backend/src/engine/book.py` | Modify | +4 lines (trade_timestamps field + capture) |
| `backend/src/main.py` | Modify | +1 line (pass config_overrides) |
| `backend/src/engine/optimizer.py` | **New** | ~120 lines (ThetaOptimizer, simulation, progress) |
| `frontend/src/lib/types.ts` | Modify | +5 lines (optional fields) |
| **Total** | 1 new file, 9 modified files | ~460 lines added |

---

## Self-Critique Verification

### Verification Questions

| # | Question | Finding |
|---|----------|---------|
| 1 | **Decomposition validity**: Is the subproblem table present with dependencies? | Verified -- Stage 1 table present at top with all 8 subproblems and "Depends On" column populated. |
| 2 | **Sequential solving chain**: Does each step reference prior steps? | Verified -- Step 2.2 references "requirements from Step 2.1"; Step 2.3 references "requirements from Step 2.1 and patterns from Step 2.2"; Step 2.4 references all three prior steps; and so on through Step 2.8. |
| 3 | **Pattern alignment**: Does the architecture follow discovered codebase patterns? | Verified -- Quoting math stays in `quoting.py` (existing pattern). Stateful optimizer gets its own file (matches `scanner.py` pattern). Config uses sub-model pattern (matches `ASSettings` pattern). Mode routing lives in `_process_symbol` (existing orchestration point). |
| 4 | **Decisiveness**: Are there any "could do X or Y" statements? | Verified -- Approach 3 is chosen definitively. No hedging in component design. Every file change specifies exact additions. |
| 5 | **Blueprint completeness**: Can a developer implement from this alone? | Verified -- Every modified file has exact change descriptions. New functions have full signatures. Integration points reference specific line numbers. Build sequence has 4 ordered phases with verification steps. |
| 6 | **WIDEN_SPREAD backward compatibility**: Does the Quote change break `loop.py:333-344`? | Verified -- The WIDEN_SPREAD path constructs `Quote(bid_price=bid, ask_price=ask, reservation_price=..., spread=..., mid_price=..., inventory=..., sigma_sq=..., gamma=..., t_minus_t=..., k=...)` with all 10 positional args. With `frozen=True` removed and new fields having defaults, this construction works unchanged. The new optional fields default to `None`. |

### Least-to-Most Verification Checklist

- [x] Stage 1 decomposition table is present with all subproblems listed
- [x] Dependencies between subproblems are explicitly stated
- [x] Each Stage 2 step starts with "Using X from Step N..."
- [x] No step references information from a later step (no forward dependencies)
- [x] Final blueprint sections cite their source steps
