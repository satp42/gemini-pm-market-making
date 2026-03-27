# Tasks: Performative Market Making Engine

## Complexity Analysis

**Overall Complexity**: High -- introduces three new engine modules (xi estimation, performative quoting, theta optimization), modifies core bot loop orchestration, extends API/DB/frontend, and requires numerical stability guarantees.

**High-Risk Areas**:
1. **Numerical stability in performative formulas** (T012-T013) -- Taylor-series fallbacks for xi near zero, NaN/Inf guards
2. **Theta optimizer blocking event loop** (T020-T021) -- asyncio.to_thread() offloading CPU-bound Optuna
3. **Mode routing in _process_symbol** (T015) -- inline branching that touches the critical quoting path
4. **WIDEN_SPREAD field propagation** (T018) -- subtle data loss bug if performative fields not copied
5. **Alembic async migration setup** (T008) -- first-time migration infrastructure with asyncpg driver

**Uncertainty Areas**:
- Optuna simulation objective function fidelity (T020): backtest simulation design is novel
- Xi OLS recovery accuracy with real (non-synthetic) market data (T010): validated only with synthetic OU data

## Least-to-Most Decomposition Chain

```
Level 0 (zero dependencies):
  - Install numpy, optuna dependencies
  - Quote dataclass: remove frozen, add optional fields
  - PerformativeSettings config class
  - MarketState: add trade_timestamps field
  - DB models: add nullable columns + new tables
  - Alembic initialization

Level 1 (depends on Level 0):
  - Fix config override propagation bug (BotLoop.__init__)
  - Scanner: extend to 4-tuple with symbol_categories
  - Update frozen dataclass test
  - Alembic migration generation

Level 2 (depends on Level 0-1, parallel tracks):
  Track A - Xi Estimation:
    - Create xi.py with estimate_xi() pure function
    - Create test_xi.py
  Track B - Performative Quoter:
    - Create performative.py with compute_performative_quote()
    - Create test_performative.py

Level 3 (depends on Levels 0-2):
  Track C - Mode Routing:
    - Mode-aware routing in _process_symbol
    - Theta cache loading from DB
    - Xi persistence (fire-and-forget)
    - WIDEN_SPREAD field propagation fix
    - Extend _process_symbol return dict + WebSocket broadcast
  Track D - Theta Optimizer:
    - Create optimizer.py with run_theta_optimization()
    - Create test_optimizer.py

Level 4 (depends on Level 3):
  - API: extend ConfigUpdateRequest, MarketSummary, QuoteHistoryEntry
  - API: POST /optimize/theta + GET /optimize/theta/status
  - API: wire optimization_progress in lifespan
  - API: extend GET /markets with performative fields
  - API: extend POST /config effective response

Level 5 (depends on Level 4):
  - Frontend: extend MarketData type
  - Frontend: display xi/theta in market detail
  - Frontend: quoting mode in StatusBar
  - Frontend: mode_switch ActivityLogEntry type

Level 6 (depends on Level 4):
  - Run Alembic migrations
  - Full test suite verification
  - Docker build and deploy verification
```

## Implementation Strategy

**Mixed approach**: Bottom-to-top for the engine modules (xi.py, performative.py, optimizer.py are pure building blocks), top-to-bottom for the integration layer (mode routing in _process_symbol calls the building blocks). This reflects that the complex algorithms need validation first, while the orchestration logic is well-defined by the existing _process_symbol pattern.

## Task Dependency Graph

```
T001 ──┐
T002 ──┤
T003 ──┼──> T009 (frozen test update)
T004 ──┤
T005 ──┼──> T006 (config override fix) ──> T015-T019 (mode routing)
T007 ──┤                                         |
T008 ──┼──> T009 (Alembic migration gen)          |
       |                                         v
       ├──> T010-T011 (xi estimation) ──────> T015-T019
       |                                         |
       ├──> T012-T013 (performative quoter) ──> T015-T019
       |                                         |
       ├──> T020-T022 (theta optimizer) ────────>|
       |                                         v
       |                                   T023-T029 (API)
       |                                         |
       |                                         v
       |                                   T030-T033 (frontend)
       |                                         |
       v                                         v
  T034-T037 (migration/deploy) <──────────────────┘
```

## Tasks

### Phase 1: Setup and Prerequisites

**Goal**: Prepare codebase for performative features. No new functionality. All existing tests must still pass.

- [x] T001 Add `numpy` dependency to `backend/pyproject.toml`
- [x] T002 [P] Add `optuna` dependency to `backend/pyproject.toml`
- [x] T003 [P] Remove `frozen=True` from Quote dataclass and add 5 optional fields (`xi`, `theta0`, `theta1`, `theta2`, `quoting_mode` all `float | None = None` / `str | None = None`) in `backend/src/engine/quoting.py` at line 9. The 10 existing required fields remain positional. New fields have `None` defaults for backward compatibility with all existing construction sites including WIDEN_SPREAD at `loop.py:337-348`.
- [x] T004 [P] Add `PerformativeSettings` class to `backend/src/config.py` after `AppSettings` (after line 107). Fields: `quoting_mode: str = "theta"`, `xi_default: float = 0.5`, `xi_min_trades: int = 15`, `xi_clamp_min: float = 0.01`, `xi_clamp_max: float = 20.0`, `r_squared_threshold: float = 0.1`, `q_ref: float = 0.0`, `theta_optimization_trials: int = 100`, `theta_optimization_simulations: int = 100`, `theta_auto_optimize_hours: float = 24.0`. Add to `Settings` class as `performative: PerformativeSettings = Field(default_factory=PerformativeSettings)` at line 123. Use `env_prefix="PERF_"`.
- [x] T005 [P] Add `trade_timestamps: list[float] = field(default_factory=list)` to `MarketState` in `backend/src/engine/book.py` at line 21. Populate from `Trade.timestamp` (int, Unix epoch) in `get_market_state()` at line 54: add `trade_timestamps = [float(t.timestamp) for t in trades]` alongside existing `trade_prices`. Pass to `MarketState` constructor at line 76.
- [x] T006 Add `config_overrides: dict[str, Any]` parameter to `BotLoop.__init__` in `backend/src/engine/loop.py` at line 34. Store as `self._config_overrides`. Add `self._symbol_categories: dict[str, str] = {}` and `self._theta_cache: dict[str, tuple[float, float, float]] = {}` to state section. Update `backend/src/main.py` at line 62-66 to pass `config_overrides=app.state.config_overrides` to BotLoop constructor. This fixes the config override propagation bug.
- [x] T007 [P] Extend `MarketScanner.scan()` return type to 4-tuple in `backend/src/engine/scanner.py`. Add `symbol_categories: dict[str, str]` dict at line 44. Populate from `event.category` inside contract loop at line 97 (alongside `symbol_titles[symbol] = event.title`). Update return at line 114 to include `symbol_categories`. Update `_maybe_scan()` in `backend/src/engine/loop.py` at line 398 to destructure 4-tuple: `symbols, titles, expiry_hours, categories = await self._scanner.scan()` and store `self._symbol_categories = categories`.
- [x] T008 [P] Add nullable columns to Quote ORM model in `backend/src/db/models.py` after line 35: `xi: Mapped[float | None]`, `theta0: Mapped[float | None]`, `theta1: Mapped[float | None]`, `theta2: Mapped[float | None]`, `quoting_mode: Mapped[str | None]` (all `Numeric(18,8)` / `String(16)`, `nullable=True`). Add new `ThetaParameter` model (table `theta_parameters`) with fields: `id`, `category` (String(64), unique), `theta0`, `theta1`, `theta2`, `xi_value`, `objective_value`, `num_trials`, `optimized_at`, `created_at`, `updated_at`. Add new `XiEstimateRecord` model (table `xi_estimates`) with fields: `id`, `timestamp`, `symbol`, `xi`, `r_squared` (nullable), `num_trades`, `used_default` (Boolean). Include appropriate indexes per data-model.md.
- [x] T009 Initialize Alembic with async template in `backend/` directory. Create `backend/alembic.ini` and `backend/alembic/env.py` configured to use project's async engine via `src.config.get_settings().database.url` and `src.db.models.Base.metadata` as target_metadata. Use `pool.NullPool` for migrations. Generate initial migration (`alembic revision --autogenerate -m "initial schema"`) and performative migration (`alembic revision --autogenerate -m "add performative market making tables"`). Follow research.md Section 4 setup steps exactly.

**Definition of Done**:
- All existing tests pass (run `pytest` from `backend/`)
- BotLoop starts without error with `config_overrides={}` passed
- Scanner returns 4-tuple; `_maybe_scan` destructures correctly
- MarketState accepts optional `trade_timestamps`
- Alembic `upgrade head` succeeds

---

### Phase 2: Xi Estimation Engine (US1 -- Xi Feedback Strength Estimation)

**Goal**: Implement OLS-based xi estimator as a standalone testable module. Maps to FR-1.

**Story**: As the bot operator, I can have per-symbol feedback strength (xi) estimated automatically from trade data so that the performative model uses market-specific parameters.

- [x] T010 [US1] Create `backend/src/engine/xi.py` (~80 lines). Implement `XiEstimate` dataclass with fields: `xi: float`, `r_squared: float | None`, `num_trades: int`, `used_default: bool`. Implement `estimate_xi(trade_prices: list[float], trade_timestamps: list[float], xi_default: float, xi_min_trades: int, xi_clamp_min: float, xi_clamp_max: float, r_squared_threshold: float) -> XiEstimate` pure function. Use `np.linalg.lstsq` for OLS regression of `delta_s ~ beta * s_n + alpha`. Extract `xi = -beta / dt`. Handle edge cases: (1) fewer than `xi_min_trades` trades -> return default; (2) flat series / rank-deficient -> `len(residuals)==0` -> r_squared=0 -> default; (3) low r_squared < threshold -> default with warning log; (4) clamp xi to [xi_clamp_min, xi_clamp_max]. Compute r_squared as `1 - ss_res/ss_tot`.
- [x] T011 [US1] Create `backend/tests/engine/test_xi.py`. Tests: (1) synthetic OU recovery with 100 points, known xi=2.0, recovered in [1.5, 2.8] (AC-1.1a); (2) synthetic OU with 500 points, recovered in [1.8, 2.3] (AC-1.1b); (3) fewer than 15 trades returns default xi without error (AC-1.2); (4) flat price series (all identical) returns clamped minimum xi, no NaN/error (AC-1.3); (5) benchmark: 100 prices < 5ms (AC-1.4); (6) random noise series produces r_squared < 0.1, falls back to default (AC-1.5); (7) XiEstimate dataclass fields are correct types.

**Definition of Done**:
- `pytest backend/tests/engine/test_xi.py` -- all pass
- Benchmark: 100 prices < 5ms
- Known xi=2.0 recovered within [1.5, 2.8] for 100 points

---

### Phase 3: Performative Quoter Engine (US2 -- Performative Reservation Price and Spread)

**Goal**: Implement performative reservation price and spread formulas. Maps to FR-2, FR-3.

**Story**: As the bot operator, I can use a performative quoting model that accounts for the market maker's own price impact so that quotes are more accurate in thin prediction markets.

- [x] T012 [P] [US2] Create `backend/src/engine/performative.py` (~120 lines). Implement: (1) `XI_EPSILON = 1e-6` constant; (2) `delta_epsilon(xi: float, T: float) -> float` with Taylor fallback `T*T/2` when `abs(xi) < XI_EPSILON`; (3) `inv_correction(xi: float, T: float) -> float` computing `(exp(-2*xi*T) - 1) / (2*xi)` with Taylor fallback `-T`; (4) `compute_performative_quote(mid_price, inventory, gamma, sigma_sq, t_minus_t, k, xi, theta0=1.0, theta1=1.0, theta2=1.0, q_ref=0.0, max_spread=0.0, best_bid=0.0, best_ask=0.0) -> Quote` that computes performative reservation price and spread per FR-2/FR-3 formulas. Include NaN/Inf guard with fallback to `compute_quote()`. Apply same clamping logic as existing A&S: [0.01, 0.99] bounds, max_spread cap, book spread cap. Set `Quote.xi`, `Quote.theta0/1/2`, `Quote.quoting_mode` on returned Quote. Import `compute_quote` from `quoting.py` for fallback.
- [x] T013 [US2] Create `backend/tests/engine/test_performative.py`. Tests: (1) A&S equivalence at xi=0: performative quote matches `compute_quote()` within 1e-6 for reservation_price and spread (AC-2.1, AC-3.1); (2) discount effect: xi>0, q=0 -> reservation_price < mid_price (AC-2.2); (3) aggressive inventory shift: xi>0, q>0 -> reservation_price shifts down more than A&S (AC-2.3); (4) clamping: extreme inputs stay in [0.01, 0.99] (AC-2.4); (5) spread >= A&S spread when xi>0 (AC-3.2); (6) bid < ask invariant across parameter sets (AC-3.3); (7) default params (gamma=0.1, k=1.5, sigma_sq=0.01, xi=1.0, T-t=0.5) spread < 0.50 (AC-3.4); (8) theta scaling: theta0=2.0 doubles mid-price component; (9) max_spread cap respected; (10) benchmark: quote computation < 1ms.

**Definition of Done**:
- `pytest backend/tests/engine/test_performative.py` -- all pass
- Benchmark: quote computation < 1ms
- A&S equivalence at xi=0 within 1e-6

---

### Phase 4: Mode Routing and Live Integration (US3 -- Quoting Mode Selection and Fallback)

**Goal**: Wire xi estimation and performative quoting into the live bot loop with mode switching. Maps to FR-5.

**Story**: As the bot operator, I can switch between A&S, performative, and theta-enhanced quoting modes at runtime so that I can compare model performance and fall back safely.

- [x] T014 Update frozen dataclass test in `backend/tests/engine/test_quoting.py` at line 89-93. Change `test_quote_is_frozen_dataclass` to verify Quote is a mutable dataclass with 15 fields (10 original + 5 new optional). Assert that the 5 new fields default to `None`. Assert that setting a field does not raise `AttributeError`.
- [x] T015 [US3] Replace hardcoded `compute_quote()` call in `_process_symbol` at `backend/src/engine/loop.py` lines 310-320. Read `quoting_mode` from `self._config_overrides.get("quoting_mode", self._settings.performative.quoting_mode)`. If "as": existing `compute_quote()` path unchanged. If "performative" or "theta": (a) call `estimate_xi()` with `state.trade_prices`, `state.trade_timestamps`, and performative config params; (b) lookup theta from `self._theta_cache.get(self._symbol_categories.get(symbol, ""), None)`; (c) if mode=="theta" and no theta found, log WARNING and fall back to performative (theta=(1,1,1)); (d) call `compute_performative_quote()` with xi, theta, and all existing params. Import `estimate_xi` from `xi.py` and `compute_performative_quote` from `performative.py`.
- [x] T016 [US3] Add `_load_theta_cache()` private method to `BotLoop` in `backend/src/engine/loop.py`. Query `ThetaParameter` table via `self._session_factory`, return `{category: (theta0, theta1, theta2)}`. Call from `_maybe_scan()` after scanner returns, storing result in `self._theta_cache`. Import `ThetaParameter` from `src.db.models`.
- [x] T017 [US3] Add `_persist_xi_estimate()` method to `BotLoop` in `backend/src/engine/loop.py`. Fire-and-forget pattern (same as `_persist_quote`). Write `XiEstimateRecord` with fields from `XiEstimate` dataclass. Call from `_process_symbol` when mode != "as". Import `XiEstimateRecord` from `src.db.models`.
- [x] T018 [US3] Fix WIDEN_SPREAD path at `backend/src/engine/loop.py` lines 337-348. When constructing the new `Quote(...)`, copy performative fields from original quote: add `xi=quote.xi`, `theta0=quote.theta0`, `theta1=quote.theta1`, `theta2=quote.theta2`, `quoting_mode=quote.quoting_mode` as keyword args. Without this, performative metadata is lost when risk widens the spread.
- [x] T019 [P] [US3] Extend `_persist_quote()` at `backend/src/engine/loop.py` lines 449-461 to include `xi=quote.xi`, `theta0=quote.theta0`, `theta1=quote.theta1`, `theta2=quote.theta2`, `quoting_mode=quote.quoting_mode` in the `QuoteRecord(...)` constructor. Extend `_process_symbol` return dict at lines 376-387 with `xi`, `theta0`, `theta1`, `theta2`, `quoting_mode`. Extend WebSocket broadcast `market_list` at lines 208-220 with `xi`, `theta0`, `theta1`, `theta2`, `quotingMode` (camelCase per convention).

**Definition of Done**:
- Bot starts in "theta" mode, performative formulas used (xi populated in logs)
- Mode switch via config_overrides takes effect next cycle (AC-5.4)
- Fallback chain works: missing theta -> performative; insufficient trades -> xi_default (AC-5.3)
- "as" mode produces identical quotes to current system (AC-5.1)
- QuoteRecord rows have xi, theta, quoting_mode populated
- WebSocket messages include performative fields
- Existing tests still pass after `test_quote_is_frozen_dataclass` update

---

### Phase 5: Theta Optimizer (US4 -- Category-Specific Theta Parameter Learning)

**Goal**: Background Optuna optimization per category. Maps to FR-4.

**Story**: As the bot operator, I can trigger theta optimization per market category so that the performative model uses category-tuned parameters for better performance.

- [x] T020 [US4] Create `backend/src/engine/optimizer.py` (~150 lines). Implement `OptimizationProgress` dataclass with fields: `running: bool = False`, `current_trial: int = 0`, `total_trials: int = 0`, `best_value: float | None = None`, `current_category: str = ""`, `categories_completed: list[str] = field(default_factory=list)`, `started_at: datetime | None = None`, `completed_at: datetime | None = None`. Implement `run_theta_optimization(categories: dict[str, list[list[float]]], progress: OptimizationProgress, session_factory, settings: PerformativeSettings) -> None` async function. For each category: (a) create Optuna study (in-memory, direction="minimize"); (b) define objective that suggests theta0/1/2 in [0.5, 2.0], runs N_sim simulations replaying price series, computes fills via Poisson `P(fill) = 1 - exp(-A * exp(-k * delta) * dt)`, tracks inventory/cash/PnL, returns `mean(-exp(-gamma * PnL[-1]))` CARA utility; (c) use `asyncio.to_thread(study.optimize, objective, n_trials=N, callbacks=[progress_callback])`; (d) persist best theta per category to `theta_parameters` via upsert (INSERT ON CONFLICT UPDATE).
- [x] T021 [US4] Add `optuna` to `backend/pyproject.toml` if not already added in T002 (idempotent check). Suppress Optuna's default logging to avoid noisy trial output during optimization.
- [x] T022 [US4] Create `backend/tests/engine/test_optimizer.py`. Tests: (1) short optimization (10 trials, 10 sims) produces theta values that differ from (1.0, 1.0, 1.0) (AC-4.1); (2) progress tracking: progress.running is True during optimization, False after (AC-4.3); (3) DB persistence: ThetaParameter row exists after optimization with correct category (AC-4.4); (4) progress callback updates current_trial and best_value.

**Definition of Done**:
- `pytest backend/tests/engine/test_optimizer.py` -- all pass
- Theta values differ from (1,1,1) after optimization
- ThetaParameter rows persisted to DB
- Progress.running is False after completion

---

### Phase 6: API Endpoints (US5 -- API and Mode Control)

**Goal**: REST API for mode switching, optimization triggers, and status. Maps to FR-5, FR-6.

**Story**: As the bot operator, I can control the quoting mode and trigger theta optimization via the REST API so that I can manage the bot without restart.

- [x] T023 [US5] Add performative fields to `ConfigUpdateRequest` at `backend/src/api/router.py` line 113: `quoting_mode: str | None = None`, `xi_default: float | None = None`, `xi_min_trades: int | None = None`, `xi_clamp_min: float | None = None`, `xi_clamp_max: float | None = None`, `q_ref: float | None = None`. Add validation: `quoting_mode` must be one of "as", "performative", "theta" (use `@field_validator`).
- [x] T024 [P] [US5] Add performative fields to `MarketSummary` at `backend/src/api/router.py` line 37: `xi: float | None = None`, `theta0: float | None = None`, `theta1: float | None = None`, `theta2: float | None = None`, `quotingMode: str | None = None`. Add same 5 fields to `QuoteHistoryEntry` at line 51.
- [x] T025 [US5] Extend POST /config effective response at `backend/src/api/router.py` lines 601-629. Add performative params to the `effective` dict: `quoting_mode`, `xi_default`, `xi_min_trades`, `xi_clamp_min`, `xi_clamp_max`, `q_ref`, reading from overrides with fallback to `settings.performative.*`.
- [x] T026 [US5] Add `POST /api/optimize/theta` endpoint to `backend/src/api/router.py`. Check `app.state.optimization_progress.running` (409 Conflict if True). Check `app.state.bot_loop` is not None (503 if None). Derive categories from `bot_loop._symbol_categories` (distinct non-empty values). If `category` query param provided, filter to that category. For each symbol in target categories, fetch trade data via `GeminiClient.get_trades(symbol, limit=200)`. Initialize progress with `running=True`. Launch `asyncio.create_task(run_theta_optimization(...))`. Return 202 with `{"status": "started", "categories": [...]}`.
- [x] T027 [P] [US5] Add `GET /api/optimize/theta/status` endpoint to `backend/src/api/router.py`. Return `app.state.optimization_progress` as JSON with camelCase field names: `running`, `currentTrial`, `totalTrials`, `bestValue`, `currentCategory`, `categoriesCompleted`, `startedAt`, `completedAt`.
- [x] T028 [US5] Wire `app.state.optimization_progress` in `backend/src/main.py` lifespan (after line 45). Import `OptimizationProgress` from `src.engine.optimizer`. Set `app.state.optimization_progress = OptimizationProgress()`.
- [x] T029 [US5] Extend GET /markets and GET /markets/{symbol} in `backend/src/api/router.py` to include performative fields from QuoteRecord. In `get_markets()` at line 210, add `xi=float(q.xi) if q.xi is not None else None` (and same for theta0, theta1, theta2, quoting_mode) to `MarketSummary()`. Same pattern in `get_market_detail()` at lines 247 and 265 for `QuoteHistoryEntry` and `currentQuote`.

**Definition of Done**:
- POST /config with `{"quoting_mode": "theta"}` returns updated effective config with performative fields
- POST /config with invalid quoting_mode returns 422
- POST /optimize/theta returns 202 Accepted
- POST /optimize/theta when running returns 409 Conflict
- GET /optimize/theta/status returns progress JSON
- GET /markets includes xi, theta, quotingMode fields (null for A&S quotes)
- GET /markets/{symbol} quoteHistory includes performative fields

---

### Phase 7: Frontend Dashboard (US6 -- Dashboard Integration)

**Goal**: Display performative data on dashboard. Maps to FR-6.

**Story**: As the bot operator, I can see xi estimates, theta parameters, and the active quoting mode on the dashboard so that I can monitor the performative model's behavior.

- [x] T030 [US6] Add `xi?: number`, `theta0?: number`, `theta1?: number`, `theta2?: number`, `quotingMode?: string` to `MarketData` interface in `frontend/src/lib/types.ts` at line 14 (before closing brace). All optional with `?` suffix.
- [x] T031 [US6] Add `"mode_switch"` to `ActivityLogEntry.type` union in `frontend/src/lib/types.ts` at line 49-54. Insert `| "mode_switch"` between `"risk_alert"` and `"info"`.
- [x] T032 [US6] Display quoting mode in `frontend/src/components/StatusBar.tsx` center section (after Markets div, around line 79). Add a new `<div>` element: read `quotingMode` from a new optional prop or from the first market's data. Map mode strings to labels: `"as"` -> `"Avellaneda-Stoikov"`, `"performative"` -> `"Performative"`, `"theta"` -> `"Theta-Enhanced"`, default -> `"--"`. Display as `<span className="text-gray-500 mr-1">Mode:</span><span className="font-mono text-gray-200">{label}</span>`.
- [x] T033 [US6] Display xi and theta values in market detail component. Find the component that renders individual market data (likely `ReservationPriceChart.tsx` or a market detail page). Add a section showing: `xi` formatted to 3 decimal places (or "--" when null/undefined), `theta0/theta1/theta2` each to 3 decimal places (or "--"). Use the pattern `value?.toFixed(3) ?? "--"`.

**Definition of Done**:
- Dashboard loads without error when performative fields are present (AC-6.1)
- Dashboard loads without error when performative fields are absent (AC-6.2)
- Xi/theta show 3 decimal places (AC-6.3)
- Mode label correct for all three modes (AC-6.4)

---

### Phase 8: Migration, Verification, and Deploy

**Goal**: Apply DB migrations, run full verification, deploy.

- [x] T034 Run `alembic upgrade head` against dev database from `backend/` directory. Verify migration applies cleanly.
- [x] T035 [P] Verify existing data integrity: new columns are NULL on all pre-existing rows, no data corruption. Run `alembic downgrade base && alembic upgrade head` to verify round-trip.
- [x] T036 Run full test suite: `pytest` from `backend/`. All tests pass including new test files (test_xi.py, test_performative.py, test_optimizer.py) and updated test_quoting.py.
- [x] T037 Docker build and deploy verification. Verify bot runs live in "theta" mode, dashboard displays performative data, mode switching works end-to-end.

**Definition of Done**:
- `alembic upgrade head` succeeds
- Round-trip: `alembic downgrade base && alembic upgrade head` succeeds
- Full pytest passes
- Bot runs live, dashboard displays performative data

---

## Testing Strategy

**Unit Tests** (included in task DoD, not separate tasks):
- `test_xi.py`: synthetic OU recovery, edge cases, performance benchmark
- `test_performative.py`: A&S equivalence, formula properties, clamping, benchmark
- `test_optimizer.py`: short optimization run, progress tracking, DB persistence
- `test_quoting.py`: updated frozen test for mutable Quote with 15 fields

**Integration Tests** (verified manually as part of Phase 4 and Phase 8):
- Mode routing: bot starts in each mode, produces correct quote types
- Fallback chain: missing theta -> performative; sparse trades -> xi_default
- Config override propagation: POST /config changes take effect next cycle
- WebSocket broadcast includes performative fields
- DB persistence: QuoteRecord, XiEstimateRecord, ThetaParameter rows correct

**Acceptance Criteria Traceability**:
| AC | Test Location | Task |
|----|--------------|------|
| AC-1.1a/b | test_xi.py | T011 |
| AC-1.2 | test_xi.py | T011 |
| AC-1.3 | test_xi.py | T011 |
| AC-1.4 | test_xi.py | T011 |
| AC-1.5 | test_xi.py | T011 |
| AC-2.1 | test_performative.py | T013 |
| AC-2.2 | test_performative.py | T013 |
| AC-2.3 | test_performative.py | T013 |
| AC-2.4 | test_performative.py | T013 |
| AC-3.1 | test_performative.py | T013 |
| AC-3.2 | test_performative.py | T013 |
| AC-3.3 | test_performative.py | T013 |
| AC-3.4 | test_performative.py | T013 |
| AC-4.1 | test_optimizer.py | T022 |
| AC-4.3 | test_optimizer.py | T022 |
| AC-4.4 | test_optimizer.py | T022 |
| AC-5.1 | Phase 4 integration | T015 |
| AC-5.2 | Phase 4 integration | T015 |
| AC-5.3 | Phase 4 integration | T015 |
| AC-5.4 | Phase 4 integration | T015 |
| AC-6.1-6.4 | Phase 7 manual | T030-T033 |

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Phase |
|------|-----------|--------|------------|-------|
| Quote dataclass change breaks existing tests | Medium | Low | T014 updates frozen test immediately; run full suite after T003 | 1 |
| Scanner 4-tuple breaks callers | Low | Low | Only one caller (`_maybe_scan`), updated in same task T007 | 1 |
| Xi estimation unreliable with sparse data | Medium | Medium | Configurable `xi_min_trades` + `xi_default` fallback + r-squared gate (T010) | 2 |
| Performative formulas produce NaN/Inf | Low | High | Taylor fallback at xi<1e-6 + `math.isfinite()` guard + A&S fallback (T012) | 3 |
| Theta optimization blocks event loop | Low | High | `asyncio.to_thread()` offloads; NumPy releases GIL (T020) | 5 |
| Config overrides not reaching BotLoop | Existing bug | High | Dict reference sharing fix (T006) | 1 |
| Alembic migration fails on existing data | Low | Medium | All new columns nullable; no destructive ops (T008, T034) | 1, 8 |
| WIDEN_SPREAD loses performative metadata | High | Medium | Explicit field copy in T018 | 4 |
| Optuna simulation not representative | Medium | Medium | Uses actual historical trade data; manual trigger allows validation (T020) | 5 |

## Parallel Execution Opportunities

**Phase 1**: T001+T002 (deps), T003 (Quote), T004 (config), T005 (MarketState), T007 (scanner), T008 (DB models) can all run in parallel. T006 depends on T004 (needs PerformativeSettings). T009 depends on T008 (needs DB models for autogenerate).

**Phase 2 + Phase 3**: T010-T011 (xi) and T012-T013 (performative) are fully independent and can run in parallel.

**Phase 4 + Phase 5**: T015-T019 (mode routing) and T020-T022 (optimizer) both depend on Phases 2+3 but are independent of each other. Can run in parallel.

**Phase 7 + Phase 8**: T030-T033 (frontend) and T034-T037 (migration) both depend on Phase 6 but are independent of each other. Can run in parallel.

## Implementation Strategy Notes

**MVP Scope**: Phases 1-4 (Setup + Xi + Performative + Mode Routing) deliver a fully functional performative quoting engine with fallback chain. This is independently testable and deployable even without theta optimization or API extensions.

**Incremental Milestones**:
1. After Phase 1: Codebase prepared, no functional change, all tests pass
2. After Phase 2+3: Two new engine modules independently tested
3. After Phase 4: Live performative quoting with mode switching
4. After Phase 5: Background theta optimization operational
5. After Phase 6: Full API control surface
6. After Phase 7: Dashboard visibility
7. After Phase 8: Production-ready with migrations

## High Complexity/Uncertainty Tasks Requiring Attention

**T012: Implement performative quoter (performative.py)**
- Complexity: High (numerical stability, Taylor-series fallbacks, NaN guards, formula correctness)
- Uncertainty: Low (formulas fully specified in spec.md and research.md)

**T015: Mode routing in _process_symbol**
- Complexity: High (touches critical quoting path, inline branching, fallback chain logic)
- Uncertainty: Low (routing logic is straightforward if/elif)

**T020: Implement theta optimizer (optimizer.py)**
- Complexity: High (Optuna integration, simulation objective function, asyncio.to_thread, DB upsert)
- Uncertainty: Medium (simulation fidelity -- Poisson fill model and CARA utility may need tuning)

**T009: Alembic async setup**
- Complexity: Medium (first-time migration infrastructure)
- Uncertainty: Medium (asyncpg driver compatibility with Alembic autogenerate)

Would you like me to:
1. Decompose any of these high-risk tasks into smaller pieces?
2. Add spike/research tasks to reduce uncertainty on the optimizer simulation?
3. Proceed as-is with these risks documented?
