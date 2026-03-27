# Implementation Plan: Performative Market Making Engine

## Technical Context

### Current State

The bot uses a standard Avellaneda-Stoikov (A&S) quoting model implemented as a pure function `compute_quote()` in `backend/src/engine/quoting.py:25-93`. The `Quote` dataclass is `@dataclass(frozen=True)` with 10 required fields. `BotLoop._process_symbol()` at `loop.py:258-387` orchestrates: fetch market state -> compute A&S quote -> risk check -> place orders. Config overrides via `POST /config` write to `app.state.config_overrides` but BotLoop never reads from it (propagation bug). Scanner returns a 3-tuple without category info despite `Event.category` existing at `gemini/models.py:95`. `MarketState` at `book.py:13-21` carries trade prices but discards timestamps.

### Target State

Three new engine files (`xi.py`, `performative.py`, `optimizer.py`) following the one-concern-per-file pattern. `_process_symbol` routes between A&S and performative quoting via an inline if/elif block reading the global `quoting_mode` from config overrides. Xi is estimated per-symbol per-cycle via OLS regression. Theta parameters are cached as a plain dict on BotLoop, refreshed on scanner cadence. Optuna optimization runs in a background thread via `asyncio.to_thread()`. The API exposes mode switching, optimization triggers, and progress status. The dashboard displays xi, theta, and quoting mode.

### Unknowns / NEEDS CLARIFICATION

All technical unknowns have been resolved in `research.md`:
- Optuna integration: `asyncio.to_thread()` (Section 1)
- OLS regression: `np.linalg.lstsq` (Section 2)
- Numerical stability: `xi < 1e-6` Taylor fallback (Section 3)
- Alembic async: async template with `run_sync` bridge (Section 4)

### Dependencies

| Dependency | Type | Version | Purpose |
|------------|------|---------|---------|
| `numpy` | Python package | latest | OLS regression (`np.linalg.lstsq`), simulation arrays |
| `optuna` | Python package | latest | Theta hyperparameter optimization |
| `alembic` | Python package | >=1.14 | Database migrations (already in pyproject.toml) |

## Constitution Check

- [x] **Type safety**: All new functions have full type hints. `XiEstimate` and `OptimizationProgress` are typed dataclasses.
- [x] **Async-first**: Optimizer uses `asyncio.to_thread()`. DB persistence is async. All I/O is async.
- [x] **Pure computation**: `estimate_xi()` and `compute_performative_quote()` are stateless pure functions matching `compute_quote()` pattern.
- [x] **Config-driven**: All tunable parameters in `PerformativeSettings`, overridable at runtime via `POST /config`.
- [x] **Test coverage**: Dedicated test files for each new engine module. Integration tests for mode routing.
- [x] **Graceful degradation**: Three-level fallback chain (theta -> performative -> A&S). NaN/Inf guard. All failures logged at WARNING.

## Resolved Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quote dataclass | Remove `frozen=True`, add optional fields with `None` defaults | Backward compat with WIDEN_SPREAD at `loop.py:337-348` |
| Mode scope | Global quoting mode only | Fallback chain handles per-symbol data availability |
| Config overrides | Pass dict reference to BotLoop | Fixes propagation bug, minimal change |
| Optuna | `asyncio.to_thread()` | Stdlib, NumPy releases GIL |
| OLS | `np.linalg.lstsq` | No scipy dep, SVD stability, direct residuals |
| Numerical | `xi < 1e-6` threshold with Taylor fallback | Exact L'Hopital limits |
| New files | `xi.py`, `performative.py`, `optimizer.py` | One-concern-per-file pattern |
| Mode routing | Inline if/elif in `_process_symbol` | No protocols/ABCs |
| Theta cache | Plain dict on BotLoop, scanner cadence refresh | Zero new classes |

## Implementation Phases

### Phase 0: Prerequisites

**Goal**: Prepare codebase for performative features. No new functionality. All existing tests must pass.

**Dependencies**: None.

**Files to create/modify**:
- `backend/src/engine/quoting.py` (modify: line 9, lines 22+)
- `backend/tests/engine/test_quoting.py` (modify: frozen test ~line 89)
- `backend/src/engine/book.py` (modify: line 21, line 54, lines 76-82)
- `backend/src/config.py` (modify: after line 98 AppSettings, line 101-116 Settings)
- `backend/src/engine/loop.py` (modify: lines 34-42, line 398)
- `backend/src/main.py` (modify: lines 62-66)
- `backend/src/engine/scanner.py` (modify: lines 26, 44, 93-97, 114)
- `backend/src/db/models.py` (modify: after line 35, after PnlSnapshot class)
- `backend/alembic/` (create: Alembic setup)

**Tasks**:
1. **0.1** Remove `frozen=True` from Quote dataclass at `quoting.py:9`. Add 5 optional fields: `xi: float | None = None`, `theta0: float | None = None`, `theta1: float | None = None`, `theta2: float | None = None`, `quoting_mode: str | None = None`.
2. **0.2** Update frozen dataclass test in `test_quoting.py` to verify Quote is a mutable dataclass with 15 fields.
3. **0.3** Add `trade_timestamps: list[float] = field(default_factory=list)` to MarketState at `book.py:21`. Populate from `Trade.timestamp` in `get_market_state()` at line 54.
4. **0.4** Add `PerformativeSettings` class to `config.py` after `AppSettings`. Fields: `quoting_mode`, `xi_default`, `xi_min_trades`, `xi_clamp_min`, `xi_clamp_max`, `r_squared_threshold`, `q_ref`, `theta_optimization_trials`, `theta_optimization_simulations`, `theta_auto_optimize_hours`. Add to `Settings` as `performative: PerformativeSettings`.
5. **0.5** Add `config_overrides: dict[str, Any]` param to `BotLoop.__init__` at `loop.py:34`. Store as `self._config_overrides`. Add `self._symbol_categories: dict[str, str] = {}` and `self._theta_cache: dict[str, tuple[float, float, float]] = {}`. Update `main.py:62-66` to pass `config_overrides=app.state.config_overrides`. Note: theta optimization is manual-trigger only (via `POST /optimize/theta`); no auto-schedule timer is implemented.
6. **0.6** Extend `MarketScanner.scan()` return type to 4-tuple, adding `symbol_categories: dict[str, str]`. Add `symbol_categories` dict at `scanner.py:44` alongside existing dicts. Populate from `event.category` inside the contract loop at `scanner.py:97` (where `symbol_titles[symbol] = event.title` is). Update return at `scanner.py:114` to include the new dict. Update `_maybe_scan()` at `loop.py:398` to destructure 4-tuple and store `self._symbol_categories`.
7. **0.7** Add nullable columns to Quote model (`xi`, `theta0`, `theta1`, `theta2`, `quoting_mode`). Add `ThetaParameter` and `XiEstimateRecord` models to `models.py`.
8. **0.8** Initialize Alembic with async template. Configure `env.py`. Generate initial and performative migrations.

**Verification**:
- [ ] `pytest` passes (all existing tests, with updated frozen test)
- [ ] BotLoop starts without error with `config_overrides={}` passed
- [ ] Scanner returns 4-tuple
- [ ] MarketState accepts and ignores optional `trade_timestamps`

---

### Phase 1: Xi Estimation

**Goal**: Implement OLS-based xi estimator as a standalone testable module.

**Dependencies**: Phase 0 (numpy dep, MarketState with timestamps).

**Files to create/modify**:
- `backend/src/engine/xi.py` (NEW, ~80 lines)
- `backend/tests/engine/test_xi.py` (NEW)
- `backend/pyproject.toml` (modify: add numpy)

**Tasks**:
1. **1.1** Create `xi.py` with `XiEstimate` dataclass and `estimate_xi()` function. Implement OLS via `np.linalg.lstsq`. Handle: too few trades, flat series (empty residuals), low r-squared, clamping.
2. **1.2** Create `test_xi.py` with tests: synthetic OU recovery (AC-1.1a, AC-1.1b), insufficient trades (AC-1.2), flat series (AC-1.3), performance <5ms (AC-1.4), low r-squared fallback (AC-1.5).
3. **1.3** Add `numpy` to `pyproject.toml` dependencies.

**Verification**:
- [ ] `pytest tests/engine/test_xi.py` -- all pass
- [ ] Benchmark: 100 prices < 5ms
- [ ] Known xi=2.0 recovered within [1.5, 2.8] for 100 points

---

### Phase 2: Performative Quoter

**Goal**: Implement performative reservation price and spread formulas.

**Dependencies**: Phase 0 (Quote dataclass changes).

**Files to create/modify**:
- `backend/src/engine/performative.py` (NEW, ~120 lines)
- `backend/tests/engine/test_performative.py` (NEW)

**Tasks**:
1. **2.1** Create `performative.py` with `XI_EPSILON`, `delta_epsilon()`, `inv_correction()`, `compute_performative_quote()`. Taylor fallbacks for xi < 1e-6. NaN/Inf guard with A&S fallback. Clamping logic replicated from `compute_quote()`.
2. **2.2** Create `test_performative.py` with tests: A&S equivalence at xi=0 (AC-2.1, AC-3.1), discount effect (AC-2.2), aggressive inventory shift (AC-2.3), clamping (AC-2.4), spread >= A&S (AC-3.2), bid < ask invariant (AC-3.3), default params spread < 0.50 (AC-3.4), theta scaling, max_spread cap, performance <1ms.

**Verification**:
- [ ] `pytest tests/engine/test_performative.py` -- all pass
- [ ] Benchmark: quote computation < 1ms
- [ ] A&S equivalence at xi=0 within 1e-6

---

### Phase 3: Mode Routing and Live Integration

**Goal**: Wire xi estimation and performative quoting into the live bot loop.

**Dependencies**: Phases 0, 1, 2.

**Files to create/modify**:
- `backend/src/engine/loop.py` (modify: lines 302-320, 208-220, 376-387, 437-461, 393-404)

**Tasks**:
1. **3.1** Replace hardcoded `compute_quote()` call at `loop.py:310-320` with mode-aware routing. Read `quoting_mode` from `self._config_overrides` with fallback to `self._settings.performative.quoting_mode`. If "as": existing path. Otherwise: estimate xi, lookup theta, call `compute_performative_quote()`.
2. **3.2** Add `_load_theta_cache()` private method. Query `theta_parameters` table, return `{category: (theta0, theta1, theta2)}`. Call from `_maybe_scan()` after scanner returns.
3. **3.3** Extend `_persist_quote()` at lines 449-461 to include `xi=quote.xi`, `theta0=quote.theta0`, `theta1=quote.theta1`, `theta2=quote.theta2`, `quoting_mode=quote.quoting_mode`.
4. **3.4** Add `_persist_xi_estimate()` method (fire-and-forget, same pattern as `_persist_quote`). Call from `_process_symbol` when mode != "as".
5. **3.5** Fix WIDEN_SPREAD path at `loop.py:333-344`: copy performative fields (`xi`, `theta0`, `theta1`, `theta2`, `quoting_mode`) from original quote to the widened Quote reconstruction as keyword args. Without this, performative metadata is lost when risk widens the spread.
6. **3.6** Extend `_process_symbol` return dict at lines 376-387 with `xi`, `theta0`, `theta1`, `theta2`, `quoting_mode`.
7. **3.7** Extend WebSocket broadcast market_list at lines 208-220 with `xi`, `theta0`, `theta1`, `theta2`, `quotingMode` (camelCase).

**Verification**:
- [ ] Bot starts in "theta" mode, performative formulas used (xi populated in logs)
- [ ] Mode switch via config_overrides takes effect next cycle
- [ ] Fallback chain works: missing theta -> performative; insufficient trades -> xi_default
- [ ] QuoteRecord rows have xi, theta, quoting_mode populated
- [ ] WebSocket messages include performative fields

---

### Phase 4: Theta Optimizer

**Goal**: Background Optuna optimization per category.

**Dependencies**: Phases 0, 1, 2.

**Files to create/modify**:
- `backend/src/engine/optimizer.py` (NEW, ~150 lines)
- `backend/tests/engine/test_optimizer.py` (NEW)
- `backend/pyproject.toml` (modify: add optuna)

**Tasks**:
1. **4.1** Create `optimizer.py` with `OptimizationProgress` dataclass and `run_theta_optimization()` async function. The objective function is a backtest simulation:
   - Fetch last N=200 trade prices per symbol via `get_trades(symbol, limit=200)` for all active symbols in the category. Use these as the historical mid-price series.
   - For each Optuna trial, suggest theta0/1/2 in [0.5, 2.0].
   - Per simulation: replay the price series step-by-step. At each step, compute performative bid/ask using the trial theta values. Determine fills via Poisson arrival: `P(fill) = 1 - exp(-A * exp(-k * delta) * dt)` where `delta` = distance from mid-price. Track inventory, cash, and PnL.
   - Return `mean(-exp(-gamma * PnL[-1]))` across N_sim simulations as the CARA utility to minimize.
   - Uses `asyncio.to_thread(study.optimize, ...)`. Persists best theta per category to `theta_parameters` via upsert.
2. **4.2** Add `optuna` to `pyproject.toml`.
3. **4.3** Create `test_optimizer.py`: short optimization (10 trials, 10 sims) produces theta != (1,1,1), DB persistence works, progress tracking works.

**Verification**:
- [ ] `pytest tests/engine/test_optimizer.py` -- all pass
- [ ] Theta values differ from (1,1,1) after optimization
- [ ] ThetaParameter rows in DB
- [ ] Progress.running is False after completion

---

### Phase 5: API Endpoints

**Goal**: REST API for mode switching, optimization triggers, and status.

**Dependencies**: Phases 3, 4.

**Files to create/modify**:
- `backend/src/api/router.py` (modify: lines 37-48, 51-62, 113-123, 601-629, new endpoints)
- `backend/src/main.py` (modify: line 45+)

**Tasks**:
1. **5.1** Add to `ConfigUpdateRequest` at `router.py:113`: `quoting_mode`, `xi_default`, `xi_min_trades`, `xi_clamp_min`, `xi_clamp_max`, `q_ref` (all `| None = None`).
2. **5.2** Add to `MarketSummary` and `QuoteHistoryEntry`: `xi`, `theta0`, `theta1`, `theta2`, `quotingMode` (all `| None = None`).
3. **5.3** Extend POST /config effective response at lines 601-629 with performative params.
4. **5.4** Add `POST /optimize/theta` endpoint: check progress.running (409 if True), collect categories and trades, launch `asyncio.create_task(run_theta_optimization(...))`, return 202.
5. **5.5** Add `GET /optimize/theta/status` endpoint: return progress as JSON.
6. **5.6** Wire `app.state.optimization_progress = OptimizationProgress()` in `main.py` lifespan.
7. **5.7** Extend GET /markets to include performative fields from QuoteRecord.

**Verification**:
- [ ] POST /config with quoting_mode returns updated effective config
- [ ] POST /optimize/theta returns 202
- [ ] POST /optimize/theta when running returns 409
- [ ] GET /optimize/theta/status returns progress
- [ ] GET /markets includes performative fields

---

### Phase 6: Frontend

**Goal**: Display performative data on dashboard.

**Dependencies**: Phase 5.

**Files to create/modify**:
- `frontend/src/lib/types.ts` (modify: MarketData interface)
- `frontend/src/components/StatusBar.tsx` (modify: center section)
- Market detail component (modify: add xi/theta display)

**Tasks**:
1. **6.1** Add `xi?`, `theta0?`, `theta1?`, `theta2?`, `quotingMode?` to `MarketData` at `types.ts:3-15`.
2. **6.2** Display xi and theta values in market detail (3 decimal places, "--" when null).
3. **6.3** Display quoting mode in StatusBar center section: "as" -> "Avellaneda-Stoikov", "performative" -> "Performative", "theta" -> "Theta-Enhanced".
4. **6.4** Add `"mode_switch"` to ActivityLogEntry type union for mode change events.

**Verification**:
- [ ] Dashboard loads with and without performative fields
- [ ] Xi/theta show 3 decimal places
- [ ] Mode label correct for all three modes

---

### Phase 7: Migration and Deploy

**Goal**: Apply DB migrations, verify full system.

**Dependencies**: Phase 5.

**Files to create/modify**:
- `backend/alembic/versions/` (migration files)

**Tasks**:
1. **7.1** Run `alembic upgrade head` against dev database.
2. **7.2** Verify existing data: new columns are NULL, no corruption.
3. **7.3** Run full test suite: `pytest`.
4. **7.4** Docker build and deploy.

**Verification**:
- [ ] `alembic upgrade head` succeeds
- [ ] Round-trip: `alembic downgrade base && alembic upgrade head`
- [ ] Full pytest passes
- [ ] Bot runs live, dashboard displays performative data

---

## Build Sequence

```
Phase 0 (Prerequisites)
   |
   +---> Phase 1 (Xi Estimation)     -- can run in parallel with Phase 2
   |
   +---> Phase 2 (Performative Quoter) -- can run in parallel with Phase 1
   |
   +---> Phase 3 (Mode Routing)       -- requires Phases 0, 1, 2
   |
   +---> Phase 4 (Theta Optimizer)    -- requires Phases 0, 1, 2; parallel with Phase 3
   |
   +---> Phase 5 (API Endpoints)      -- requires Phases 3, 4
   |
   +---> Phase 6 (Frontend)           -- requires Phase 5
   |
   +---> Phase 7 (Migration/Deploy)   -- requires Phase 5
```

Phases 1 and 2 are independent and can be implemented in parallel.
Phases 3 and 4 can be implemented in parallel (both depend on 0+1+2).
Phases 6 and 7 can be implemented in parallel (both depend on 5).

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation | Phase |
|------|-----------|--------|------------|-------|
| Quote dataclass change breaks tests | Medium | Low | Update frozen test immediately (0.2); run full suite after 0.1 | 0 |
| Scanner 4-tuple breaks callers | Low | Low | Only one caller (`_maybe_scan`), updated in same task | 0 |
| Xi estimation unreliable with sparse data | Medium | Medium | Configurable `xi_min_trades` + `xi_default` fallback + r-squared gate | 1 |
| Performative formulas produce NaN | Low | High | Taylor fallback at xi<1e-6 + `math.isfinite()` guard + A&S fallback | 2 |
| Theta optimization blocks event loop | Low | High | `asyncio.to_thread()` offloads; NumPy releases GIL | 4 |
| Config overrides not reaching BotLoop | (Existing bug) | High | Dict reference sharing (fix in 0.5) | 0 |
| Alembic migration fails on existing data | Low | Medium | All new columns nullable; no destructive ops | 7 |
| Need immediate rollback | - | - | `POST /config {"quoting_mode": "as"}` reverts to A&S instantly | Any |

## Verification Plan

| Phase | What to verify | How |
|-------|---------------|-----|
| 0 | Existing tests pass, BotLoop starts, scanner 4-tuple | `pytest`, manual start |
| 1 | Xi recovery, edge cases, performance | `pytest tests/engine/test_xi.py` |
| 2 | A&S equivalence, formula properties, performance | `pytest tests/engine/test_performative.py` |
| 3 | Mode routing, fallback chain, persistence | Integration test with mock data |
| 4 | Theta non-trivial, DB persistence, progress | `pytest tests/engine/test_optimizer.py` |
| 5 | Endpoints return correct responses | API tests with httpx |
| 6 | Dashboard renders with/without data | Manual browser test |
| 7 | Migration round-trips, full suite | `alembic upgrade/downgrade`, `pytest` |
