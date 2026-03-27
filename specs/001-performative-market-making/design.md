# Architecture Design: Performative Market Making Engine

> **Chosen approach**: Pragmatic -- three new engine files (`xi.py`, `performative.py`, `optimizer.py`) following existing one-concern-per-file pattern, mode routing inline in `_process_symbol`, theta cache as plain dict on BotLoop, no protocols/ABCs.

---

## Problem Decomposition

To design "Performative Market Making Engine", I will solve these subproblems in order:

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

### Functional Requirements (summary)

1. **Xi estimation** (FR-1): Per-symbol, per-cycle OU regression on trade prices to estimate feedback strength. Fallback chain: insufficient trades -> default xi; low r-squared -> default xi.
2. **Performative reservation price** (FR-2): Replace A&S `r = s - q*gamma*sigma^2*(T-t)` with `r_perf = s*exp(-xi*(T-t)) - gamma*sigma^2*(...)` formula. Theta-enhanced variant scales terms by (theta0, theta1, theta2).
3. **Performative spread** (FR-3): Replace A&S spread with mean-reversion-corrected spread. Always wider than A&S.
4. **Theta learning** (FR-4): Optuna optimization per event category. 100 trials x 100 simulations. Background async task. Results persisted to DB.
5. **Mode switching** (FR-5): Three modes (as/performative/theta) with fallback chain. Runtime-switchable via POST /config.
6. **Dashboard integration** (FR-6): Extend WebSocket tick and REST responses with xi, theta, quotingMode fields.

### Non-Functional Requirements

- Xi estimation < 5ms per symbol
- Quote computation < 1ms per symbol
- Theta optimization non-blocking (background thread via asyncio.to_thread)
- All existing risk checks preserved unchanged
- Backward compatible DB schema (nullable new columns)

### Key Constraints from Resolved Decisions

| Decision | Choice | Reference |
|----------|--------|-----------|
| Quote dataclass | Remove `frozen=True`, add optional fields with `None` defaults | research.md D1 |
| Mode scope | Global quoting mode only (no per-symbol overrides) | research.md D2 |
| Config overrides | Pass dict reference to BotLoop | research.md D3 |
| Optuna integration | `asyncio.to_thread()` | research.md Section 1 |
| OLS regression | `np.linalg.lstsq` | research.md Section 2 |
| Numerical stability | `xi < 1e-6` threshold with Taylor fallback | research.md Section 3 |

**Feeds into**: Steps 2, 3, 4, 5, 6, 7, 8 -- everything builds on these requirements.

---

## Step 2.2: Codebase Pattern Analysis

*Using requirements from Step 2.1...*

### Technology Stack
- **Backend**: Python 3.12+, FastAPI, SQLAlchemy 2.0+ async, PostgreSQL/asyncpg
- **Frontend**: Next.js 16, React 19, TypeScript, TailwindCSS
- **Patterns**: Pydantic settings, frozen dataclasses, async-first I/O, stateless engine functions

### Discovered Patterns (with file references)

| Pattern | Location | Description |
|---------|----------|-------------|
| **Stateless engine functions** | `quoting.py:25-93` | `compute_quote()` is a pure function: takes scalars, returns a frozen dataclass. No state, no side effects. |
| **Frozen dataclass results** | `quoting.py:9-22` | `Quote` is `@dataclass(frozen=True)` with all fields required. `MarketState` in `book.py:13-21` follows the same pattern. |
| **One concern per engine file** | `engine/quoting.py`, `engine/risk.py`, `engine/scanner.py`, `engine/book.py` | Each engine file handles a single responsibility. |
| **BotLoop orchestration** | `loop.py:255-387` | `_process_symbol()` is the integration point: fetch state -> compute quote -> risk check -> place orders. Linear pipeline. |
| **WIDEN_SPREAD reconstruction** | `loop.py:337-348` | Constructs a new `Quote(...)` with all 10 positional fields. Adding required fields would break this site. |
| **Settings composition** | `config.py:98-112` | Root `Settings` composes sub-settings via `Field(default_factory=...)`. Each sub-setting is a separate `BaseSettings` class with env prefix. |
| **Config overrides (disconnected)** | `router.py:576-632`, `main.py:45` | `app.state.config_overrides` is a plain dict. POST /config merges into it. BotLoop never reads from it -- propagation bug. |
| **Scanner return tuple** | `scanner.py:26` | Returns `tuple[list[str], dict[str, str], dict[str, float]]`. Category info available on `Event.category` (models.py:95) but not returned. |
| **DB persistence pattern** | `loop.py:437-480` | Fire-and-forget inside try/except. `QuoteRecord` constructed with keyword args matching column names. |
| **WebSocket broadcast** | `loop.py:196-240` | `on_tick` callback builds a dict with `MarketData`-shaped objects. Currently does not include xi/theta/mode. |
| **API response schemas** | `router.py:37-48` | Pydantic `BaseModel` classes with camelCase fields. `MarketSummary` has 11 fields. |
| **Frontend types** | `frontend/src/lib/types.ts:3-15` | `MarketData` interface mirrors `MarketSummary`. Optional fields use `?` suffix. |
| **Event.category** | `gemini/models.py:95` | `category: str = ""` -- already exists on the Event model, just not surfaced through scanner. |
| **Trade.timestamp** | `gemini/models.py:34` | `timestamp: int` -- Unix epoch, available from Gemini API but discarded by `OrderBookMonitor` (`book.py:54`). |
| **MarketState frozen** | `book.py:13-21` | `@dataclass(frozen=True)` with `trade_prices: list[float]`. No timestamps field. |

### Abstraction Layers

```
API Layer (router.py, ws.py)
    |
Orchestration (loop.py -- BotLoop)
    |
Engine Components (quoting.py, scanner.py, book.py, risk.py, orders.py, positions.py)
    |
Gemini Client (client.py, models.py)
    |
Database (models.py, database.py)
```

**Feeds into**: Steps 3, 4, 5, 6 -- pattern alignment drives all design choices.

---

## Step 2.3: Design Approaches

*Using patterns from Step 2.2 and requirements from Step 2.1...*

### Approach 1: Extend-in-Place (Probability: 0.85)

Add performative logic directly into existing files. `compute_performative_quote()` goes into `quoting.py` alongside `compute_quote()`. Xi estimator is a new function in `quoting.py`. Mode routing is inline in `loop.py:_process_symbol`. Theta optimizer is a single new file.

- **Pattern alignment**: Follows the stateless-function-in-engine-module pattern. Minimizes new files.
- **Pros**: Fastest to implement. Zero new abstractions. Easy to review diffs. All quoting logic in one file.
- **Cons**: `quoting.py` grows from 109 to ~300 lines. Xi estimation is conceptually separate from quoting (numpy OLS vs. scalar math). Mixes dependencies (numpy enters a pure-math module).

### Approach 2: Thin Module Split (Probability: 0.82)

Three new engine files: `xi.py` (xi estimation), `performative.py` (performative quote computation), `optimizer.py` (theta optimization). Mode routing in `loop.py`. Config in existing `config.py`.

- **Pattern alignment**: Follows the one-concern-per-file pattern (like `risk.py`, `scanner.py`). Stateless functions.
- **Pros**: Clean separation. Xi estimation is independently testable. `quoting.py` unchanged (A&S preserved as-is). Each file small and focused (~80-150 lines).
- **Cons**: Three new files. Slightly more import churn. Duplicate clamping logic in performative.py.

### Approach 3: Two-File Split (Probability: 0.80)

Keep xi estimation and performative formulas in `quoting.py` (they are pure computation). Put optimizer in `optimizer.py`.

- **Pattern alignment**: Treats `quoting.py` as "all quoting math" module.
- **Pros**: Only one new file. Natural grouping of computation.
- **Cons**: `quoting.py` grows to ~250 lines. Mixes numpy (xi) with scalar math (quoting). One test file covers both A&S and performative.

### Approach 4: Strategy Pattern with Protocol (Probability: 0.08)

Define a `Quoter` protocol with `compute(...)` method. `ASQuoter`, `PerformativeQuoter`, `ThetaQuoter` implement it. `BotLoop` holds a `Quoter` instance, switched at runtime.

- **Pattern alignment**: Inconsistent -- no protocols or ABCs exist anywhere in the codebase.
- **Pros**: Clean polymorphism. Easy to add modes.
- **Cons**: Premature abstraction. Three classes for what is effectively one function with different parameters. Anti-pattern for this codebase.

### Approach 5: Monolithic Service (Probability: 0.05)

Single `PerformativeService` class managing theta cache, xi cache, optimization progress, and performative quoting.

- **Pattern alignment**: Violates stateless-function pattern. Introduces stateful service into engine layer.
- **Pros**: Single entry point.
- **Cons**: Tight coupling. Mixes I/O with pure computation. Hard to test.

### Approach 6: External Microservice (Probability: 0.03)

Performative quoting as a separate FastAPI service called via HTTP.

- **Pattern alignment**: Contradicts single-process Docker Compose architecture entirely.
- **Pros**: Independent deployment.
- **Cons**: Network latency destroys sub-1ms quote budget. Operational nightmare.

**Feeds into**: Step 4 -- selecting the winning approach.

---

## Step 2.4: Architecture Decision

*Using approaches from Step 2.3, patterns from Step 2.2, and requirements from Step 2.1...*

### Decision: Approach 2 -- Thin Module Split

**Rationale**:

1. **Pattern alignment**: The existing engine layer has one file per concern (`quoting.py`, `risk.py`, `scanner.py`, `book.py`). Xi estimation is a distinct concern (OLS regression on trade data with numpy) from quote computation (scalar formula evaluation with `math`). Theta optimization is async/stateful, distinct from both. Three new files follow the established pattern exactly.

2. **Testability**: Xi estimation needs its own test suite (synthetic OU data, edge cases, r-squared thresholds). Performative quoting needs formula verification tests. Keeping them in separate files means `test_xi.py` and `test_performative.py` mirror the source structure, matching the existing `tests/engine/test_*.py` convention.

3. **Dependency isolation**: `xi.py` imports `numpy`; `performative.py` uses only `math`; `optimizer.py` imports `optuna` and `numpy`. No unnecessary dependency bleeding into `quoting.py`.

4. **Preservation of A&S path**: `quoting.py` stays untouched except for the `Quote` dataclass modification (removing `frozen=True`, adding optional fields). The existing A&S tests continue to pass. Zero risk of breaking the current production quoting logic.

5. **Why not Approach 1**: Cramming xi estimation (numpy OLS) into `quoting.py` (currently pure `math` module) mixes dependencies and concerns. The functions would be independently tested anyway, so "fewer files" is illusory.

6. **Why not Approach 3**: Same dependency mixing problem. `quoting.py` becomes 250+ lines with two unrelated concerns.

7. **Why not Approach 4**: No protocols or ABCs exist in this codebase. Introducing them for three modes that are really just `compute_quote(...)` vs. `compute_performative_quote(...)` with different args is over-engineering. The mode routing is 15 lines of if/elif in `_process_symbol`.

### Files to Create

| New File | Purpose | Lines (est.) |
|----------|---------|-------------|
| `backend/src/engine/xi.py` | Xi estimation via OU regression | ~80 |
| `backend/src/engine/performative.py` | Performative reservation price + spread computation | ~120 |
| `backend/src/engine/optimizer.py` | Theta optimization via Optuna (background task) | ~150 |

### Files to Modify

| Existing File | Changes |
|---------------|---------|
| `backend/src/engine/quoting.py` | Remove `frozen=True` from Quote, add 5 optional fields |
| `backend/src/engine/loop.py` | Accept config_overrides dict, add mode routing in `_process_symbol`, theta cache, extend broadcast |
| `backend/src/engine/scanner.py` | Return `symbol_categories` dict in scan() tuple |
| `backend/src/engine/book.py` | Add `trade_timestamps` to MarketState, populate from Trade.timestamp |
| `backend/src/config.py` | Add `PerformativeSettings` sub-class |
| `backend/src/db/models.py` | Add columns to Quote, add ThetaParameter + XiEstimate models |
| `backend/src/api/router.py` | Extend ConfigUpdateRequest, add /optimize/theta endpoints, extend MarketSummary |
| `backend/src/main.py` | Pass config_overrides to BotLoop, wire optimization progress |
| `frontend/src/lib/types.ts` | Add xi, theta, quotingMode to MarketData |
| `frontend/src/components/StatusBar.tsx` | Display active quoting mode label |

**Feeds into**: Steps 5, 6, 7, 8.

---

## Step 2.5: Component Design

*Using the chosen approach from Step 2.4 and patterns from Step 2.2...*

### Component 1: Xi Estimator

**File**: `backend/src/engine/xi.py`

**Responsibilities**: Estimate the OU mean-reversion parameter xi from a list of trade prices and timestamps.

**Dependencies**: `numpy` (`np.linalg.lstsq`), `math`, `logging`

**Interface**:
```python
@dataclass
class XiEstimate:
    xi: float
    r_squared: float | None
    num_trades: int
    used_default: bool

def estimate_xi(
    trade_prices: list[float],
    trade_timestamps: list[float],   # unix epoch seconds
    xi_default: float = 0.5,
    xi_min_trades: int = 15,
    xi_clamp_min: float = 0.01,
    xi_clamp_max: float = 20.0,
    r_squared_threshold: float = 0.1,
) -> XiEstimate
```

**Design notes**:
- Pure function, stateless, no side effects (matches `compute_quote` pattern from Step 2.2).
- Returns a dataclass with the estimate plus metadata for logging/persistence.
- Handles all edge cases internally: too few trades -> return `xi_default` with `used_default=True`; flat series (rank-deficient lstsq returns empty residuals) -> `r_squared=0.0`, triggers quality gate; low r-squared -> return `xi_default`.
- OLS regression: `delta_s ~ beta * s_n + alpha` via `np.linalg.lstsq` with design matrix `A = np.column_stack([s_n, np.ones(n)])`. Xi extracted as `xi = -beta / dt` where `dt` is mean time step.
- R-squared computed manually: `1 - ss_res / ss_tot`, handling the empty-residuals edge case.

### Component 2: Performative Quoter

**File**: `backend/src/engine/performative.py`

**Responsibilities**: Compute performative reservation price and spread given xi, theta, and existing A&S parameters.

**Dependencies**: `math`, `logging`. No numpy needed (scalar math only).

**Interface**:
```python
XI_EPSILON: float = 1e-6

def delta_epsilon(xi: float, T: float) -> float:
    """(1 - exp(-xi*T) - xi*T*exp(-xi*T)) / xi^2, with Taylor fallback."""

def inv_correction(xi: float, T: float) -> float:
    """(exp(-2*xi*T) - 1) / (2*xi), with Taylor fallback."""

def compute_performative_quote(
    mid_price: float,
    inventory: float,
    gamma: float,
    sigma_sq: float,
    t_minus_t: float,
    k: float,
    xi: float,
    theta0: float = 1.0,
    theta1: float = 1.0,
    theta2: float = 1.0,
    q_ref: float = 0.0,
    max_spread: float = 0.0,
    best_bid: float = 0.0,
    best_ask: float = 0.0,
) -> Quote
```

**Design notes**:
- Mirrors the signature of `compute_quote()` from `quoting.py:25-35` with added xi/theta/q_ref parameters. This makes mode routing trivial -- same inputs, different function.
- Returns the same `Quote` dataclass with optional fields populated: `xi`, `theta0`, `theta1`, `theta2`, `quoting_mode`.
- Taylor fallback at `XI_EPSILON = 1e-6`: `delta_epsilon -> T^2/2`, `inv_correction -> -T`. Matches analytical L'Hopital limits from research.md Section 3.
- The [0.01, 0.99] clamping, max_spread cap, and book-spread cap logic are duplicated from `compute_quote`. This is intentional -- keeping the two functions independent avoids coupling. The logic is ~10 lines.
- NaN/Inf guard: if the final reservation price is not finite, log a warning and fall back to calling `compute_quote()` directly (A&S fallback).
- The low-probability market anchor (bid < 0.02 -> snap to book) is replicated from `compute_quote` at `quoting.py:74-76`.

### Component 3: Theta Optimizer

**File**: `backend/src/engine/optimizer.py`

**Responsibilities**: Run Optuna theta optimization per category. Background task management. Progress reporting.

**Dependencies**: `optuna`, `numpy`, `asyncio`, `sqlalchemy`

**Interface**:
```python
@dataclass
class OptimizationProgress:
    running: bool = False
    current_trial: int = 0
    total_trials: int = 0
    best_value: float | None = None
    current_category: str = ""
    categories_completed: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

async def run_theta_optimization(
    categories: list[str],
    symbol_trades: dict[str, list[float]],   # symbol -> trade prices
    symbol_categories: dict[str, str],        # symbol -> category
    session_factory: async_sessionmaker,
    progress: OptimizationProgress,
    gamma: float,
    sigma_sq: float,
    t_minus_t: float,
    k: float,
    n_trials: int = 100,
    n_simulations: int = 100,
) -> None
```

**Design notes**:
- The `progress` dataclass is shared between the optimizer (writer) and the API status endpoint (reader). GIL guarantees atomic attribute writes for simple types (per research.md Section 1).
- `asyncio.to_thread(study.optimize, ...)` wraps the synchronous Optuna call.
- The simulation objective function is a **backtest** defined inside `run_theta_optimization` as a closure:
  1. Fetch last N=200 trade prices per symbol via `get_trades(symbol, limit=200)` for all active symbols in the category. These form the historical mid-price series.
  2. Per simulation: replay the price series step-by-step. At each step, compute performative bid/ask using the trial's theta values. Determine fills via Poisson arrival: `P(fill) = 1 - exp(-A * exp(-k * delta) * dt)` where delta = distance from mid-price. Track inventory, cash, and terminal PnL.
  3. Return `mean(-exp(-gamma * PnL[-1]))` across n_simulations as the CARA utility to minimize.
- Theta optimization is **manual-trigger only** via `POST /optimize/theta`. No auto-schedule timer.
- After optimization completes for each category, results are persisted to the `theta_parameters` table via an async session.
- The optimizer is launched as an `asyncio.create_task` (matching the existing `BotLoop.start()` pattern from `loop.py:100`).
- Optuna progress callback updates `progress.current_trial` and `progress.best_value` after each trial.
- In-memory Optuna storage (no RDB backend needed). Final results persisted to app PostgreSQL.

### Component 4: Theta Cache (inline in BotLoop)

**Pragmatic choice**: No separate class. The theta cache is a simple `dict[str, tuple[float, float, float]]` on `BotLoop`, refreshed by reading from the DB at the start of each scan cycle (every 5 minutes via `_maybe_scan()`).

```python
# In BotLoop.__init__:
self._theta_cache: dict[str, tuple[float, float, float]] = {}

# In BotLoop._maybe_scan(), after scanning:
self._theta_cache = await self._load_theta_cache()
```

The `_load_theta_cache` method is a private helper on BotLoop (~15 lines) that queries `theta_parameters` table and returns the dict. This avoids a new file/class for what is one DB query and a dict assignment.

### Component 5: Config Extension

**File**: `backend/src/config.py` (modify, after line 98 AppSettings, lines 101-116 Settings)

**Change**: Add `PerformativeSettings` sub-class alongside existing `ASSettings`.

```python
class PerformativeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PERF_")

    quoting_mode: str = "theta"
    xi_default: float = 0.5
    xi_min_trades: int = 15
    xi_clamp_min: float = 0.01
    xi_clamp_max: float = 20.0
    r_squared_threshold: float = 0.1
    q_ref: float = 0.0
    theta_optimization_trials: int = 100
    theta_optimization_simulations: int = 100
    theta_auto_optimize_hours: int = 24
```

And in `Settings` (lines 101-116):
```python
performative: PerformativeSettings = Field(default_factory=PerformativeSettings)
```

### Component 6: Database Models

**File**: `backend/src/db/models.py` (modify)

**Changes**:

Add nullable columns to `Quote` (after line 35):
```python
xi: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
theta0: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
theta1: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
theta2: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
quoting_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
```

Add `ThetaParameter` model (new table `theta_parameters`):
```python
class ThetaParameter(Base):
    __tablename__ = "theta_parameters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    theta0: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    theta1: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    theta2: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    xi_value: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    objective_value: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    num_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    optimized_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

Add `XiEstimate` model (new table `xi_estimates`, for observability):
```python
class XiEstimateRecord(Base):
    __tablename__ = "xi_estimates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    xi: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    r_squared: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    num_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    used_default: Mapped[bool] = mapped_column(nullable=False, default=False)
```

### Component 7: MarketState Extension

**File**: `backend/src/engine/book.py` (modify)

**Change**: Add `trade_timestamps: list[float] = field(default_factory=list)` to `MarketState` dataclass (line 21). In `get_market_state()`, extract timestamps from Trade objects at line 54:

```python
trade_timestamps = [float(t.timestamp) for t in trades]
```

Pass to MarketState constructor at line 76-82. Backward compatible -- the field has a default, so existing construction sites continue to work.

**Feeds into**: Steps 6, 7, 8.

---

## Step 2.6: Integration Mapping

*Using component design from Step 2.5 and patterns from Step 2.2...*

### Integration Point 1: BotLoop.__init__ -- Config Overrides

**File**: `backend/src/engine/loop.py`, line 34-42

**Change**: Add `config_overrides: dict[str, Any]` parameter to `__init__`. Store as `self._config_overrides`. Add `self._symbol_categories: dict[str, str] = {}` and `self._theta_cache: dict[str, tuple[float, float, float]] = {}`.

**Caller change**: `backend/src/main.py`, line 62-66. Pass `config_overrides=app.state.config_overrides` to the BotLoop constructor. Since both the API handler and BotLoop hold a reference to the same dict object, mutations from POST /config are visible to the bot loop immediately.

### Integration Point 2: _process_symbol -- Mode Routing

**File**: `backend/src/engine/loop.py`, lines 302-320 (the quote computation block)

**Current code** (lines 302-320):
```python
sigma_sq = estimate_variance(state.trade_prices, ...)
gamma = self._settings.avellaneda_stoikov.gamma
k = self._settings.avellaneda_stoikov.k
quote = compute_quote(mid_price=..., ...)
```

**New code** (replaces lines 302-320):
```python
# Read effective config (overrides first, then settings)
perf = self._settings.performative
mode = self._config_overrides.get("quoting_mode", perf.quoting_mode)
gamma = self._config_overrides.get("gamma", self._settings.avellaneda_stoikov.gamma)
k = self._config_overrides.get("k", self._settings.avellaneda_stoikov.k)
sigma_sq = estimate_variance(state.trade_prices, ...)

if mode == "as":
    quote = compute_quote(mid_price=..., inventory=..., gamma=..., ...)
    quote.quoting_mode = "as"
else:
    # Estimate xi
    xi_est = estimate_xi(state.trade_prices, state.trade_timestamps, ...)
    xi = xi_est.xi

    # Theta lookup
    theta = (1.0, 1.0, 1.0)
    actual_mode = "performative"
    if mode == "theta":
        category = self._symbol_categories.get(symbol, "")
        if category and category in self._theta_cache:
            theta = self._theta_cache[category]
            actual_mode = "theta"
        else:
            logger.warning("No theta for category '%s', falling back to performative", category)

    quote = compute_performative_quote(
        mid_price=..., inventory=..., gamma=..., sigma_sq=...,
        t_minus_t=..., k=..., xi=xi,
        theta0=theta[0], theta1=theta[1], theta2=theta[2],
        q_ref=self._config_overrides.get("q_ref", perf.q_ref),
        max_spread=..., best_bid=..., best_ask=...,
    )
```

The mode routing is inline -- ~20 lines of straightforward if/elif. No strategy pattern, no dispatch table.

### Integration Point 3: Scanner -- Category Mapping

**File**: `backend/src/engine/scanner.py`, line 26 and line 114

**Change**: Extend return type to `tuple[list[str], dict[str, str], dict[str, float], dict[str, str]]`. The fourth element is `symbol_categories: dict[str, str]`. Add `symbol_categories: dict[str, str] = {}` at line 44 alongside existing dicts.

Inside the contract loop at line 97 (where `symbol_titles[symbol] = event.title`), add: `symbol_categories[symbol] = event.category`. Update return at line 114.

**Caller change**: `backend/src/engine/loop.py`, line 398. Update destructuring:
```python
symbols, titles, expiry_hours, categories = await self._scanner.scan()
self._symbol_categories = categories
```

### Integration Point 4: Persistence -- Extended QuoteRecord

**File**: `backend/src/engine/loop.py`, lines 449-461

**Change**: Add new fields to the QuoteRecord construction:
```python
xi=quote.xi,
theta0=quote.theta0,
theta1=quote.theta1,
theta2=quote.theta2,
quoting_mode=quote.quoting_mode,
```

These are all nullable columns, so existing rows are unaffected.

### Integration Point 5: Persistence -- Xi Estimates

**File**: `backend/src/engine/loop.py` (new method `_persist_xi_estimate`)

**Change**: Add fire-and-forget persistence method following same pattern as `_persist_quote` (line 437). Called from `_process_symbol` when mode is not "as".

### Integration Point 6: WebSocket Broadcast -- Extended Market Data

**File**: `backend/src/engine/loop.py`, lines 208-220

**Change**: Add to the market_list dict construction:
```python
"xi": q.get("xi"),
"theta0": q.get("theta0"),
"theta1": q.get("theta1"),
"theta2": q.get("theta2"),
"quotingMode": q.get("quoting_mode"),
```

And extend the `_process_symbol` return dict (line 376-387) to include:
```python
"xi": quote.xi,
"theta0": quote.theta0,
"theta1": quote.theta1,
"theta2": quote.theta2,
"quoting_mode": quote.quoting_mode,
```

### Integration Point 7: API -- New Endpoints and Extended Schemas

**File**: `backend/src/api/router.py`

**Changes**:
- `MarketSummary` (line 37-48): Add `xi: float | None = None`, `theta0: float | None = None`, `theta1: float | None = None`, `theta2: float | None = None`, `quotingMode: str | None = None`.
- `QuoteHistoryEntry` (line 51-62): Add same fields.
- `ConfigUpdateRequest` (line 113-123): Add `quoting_mode: str | None = None`, `xi_default: float | None = None`, `xi_min_trades: int | None = None`, `xi_clamp_min: float | None = None`, `xi_clamp_max: float | None = None`, `q_ref: float | None = None`.
- New endpoint `POST /optimize/theta`: Trigger optimization. Returns 202 with category list. Launches `asyncio.create_task` with the optimizer function.
- New endpoint `GET /optimize/theta/status`: Return `app.state.optimization_progress` contents.
- Extend `POST /config` effective config response (line 601-629) to include performative parameters.

### Integration Point 8: Frontend Types

**File**: `frontend/src/lib/types.ts`

**Change**: Add optional fields to `MarketData` (line 3-15):
```typescript
xi?: number;
theta0?: number;
theta1?: number;
theta2?: number;
quotingMode?: string;
```

### Integration Point 9: StatusBar -- Quoting Mode Label

**File**: `frontend/src/components/StatusBar.tsx`

**Change**: Accept `quotingMode` from status/markets data. Display human-readable label in the center section alongside Uptime and Markets:
- "as" -> "Avellaneda-Stoikov"
- "performative" -> "Performative"
- "theta" -> "Theta-Enhanced"

### Integration Point 10: Main -- Wire Optimizer and Config Overrides

**File**: `backend/src/main.py`, line 62-66

**Changes**:
- Pass `config_overrides=app.state.config_overrides` to `BotLoop(...)` constructor.
- Initialize `app.state.optimization_progress = OptimizationProgress()` in lifespan.

**Feeds into**: Steps 7, 8.

---

## Step 2.7: Data Flow

*Using components from Step 2.5 and integration points from Step 2.6...*

### Flow 1: Normal Performative Quoting Cycle

```
1. BotLoop._run() starts cycle
   |
2. _maybe_scan()
   |-- MarketScanner.scan() returns (symbols, titles, expiry_hours, categories)
   |-- _load_theta_cache() queries theta_parameters table -> self._theta_cache
   |-- self._symbol_categories updated
   |
3. For each symbol: _process_symbol(symbol, inventories)
   |
   |-- a. OrderBookMonitor.get_market_state(symbol)
   |       Returns MarketState(mid_price, best_bid, best_ask, spread,
   |                           trade_prices, trade_timestamps)
   |
   |-- b. Read effective config from self._config_overrides + self._settings
   |       -> mode, gamma, k, sigma_sq, xi_default, xi_min_trades, ...
   |
   |-- c. estimate_variance(state.trade_prices) -> sigma_sq
   |
   |-- d. Mode check: mode == "as"?
   |       YES -> compute_quote(...) -> Quote (existing A&S path)
   |       NO  -> continue to step e
   |
   |-- e. estimate_xi(state.trade_prices, state.trade_timestamps, ...) -> XiEstimate
   |       |-- If insufficient trades: xi = xi_default, used_default = True
   |       |-- If low r_squared: xi = xi_default, used_default = True
   |       |-- Otherwise: xi = clamped OLS estimate
   |
   |-- f. Theta lookup:
   |       |-- category = self._symbol_categories.get(symbol, "")
   |       |-- If mode == "theta" and category in self._theta_cache:
   |       |       theta = self._theta_cache[category], actual_mode = "theta"
   |       |-- Else:
   |       |       theta = (1.0, 1.0, 1.0), actual_mode = "performative"
   |
   |-- g. compute_performative_quote(mid_price, inventory, gamma, sigma_sq,
   |       t_minus_t, k, xi, theta0, theta1, theta2, q_ref, max_spread,
   |       best_bid, best_ask) -> Quote
   |       |-- Computes r_perf using delta_epsilon() and inv_correction()
   |       |   with Taylor fallback when xi < 1e-6
   |       |-- Computes spread_perf
   |       |-- Applies max_spread cap, book_spread cap, [0.01, 0.99] clamp
   |       |-- NaN/Inf guard: falls back to compute_quote() if non-finite
   |       |-- Sets quote.xi, quote.theta0-2, quote.quoting_mode
   |
   |-- h. Risk checks (unchanged): WIDEN_SPREAD may reconstruct Quote
   |
   |-- i. Cancel stale orders, place new bid/ask
   |
   |-- j. _persist_quote(): QuoteRecord includes xi, theta0-2, quoting_mode
   |
   |-- k. _persist_xi_estimate(): XiEstimateRecord for observability
   |       (fire-and-forget, same pattern as _persist_quote)
   |
4. _persist_tick() (unchanged)
   |
5. WebSocket broadcast with xi, theta, quotingMode fields
```

### Flow 2: Theta Optimization

```
1. POST /optimize/theta received by router
   |
2. Router checks app.state.optimization_progress.running
   |-- If True: return 409 Conflict
   |-- If False: continue
   |
3. Collect inputs:
   |-- categories = distinct non-empty categories from scanner data
   |       (from bot_loop._symbol_categories values)
   |-- For each category: collect trade prices from all symbols in that category
   |       (fetched via GeminiClient.get_trades for each symbol)
   |
4. Initialize OptimizationProgress on app.state
   |
5. asyncio.create_task(run_theta_optimization(...))
   |-- Returns 202 Accepted immediately
   |
6. Inside run_theta_optimization (background):
   |-- For each category:
   |   |-- Create optuna.create_study(direction="minimize")
   |   |-- study.optimize(objective, n_trials=100, callbacks=[progress_cb])
   |   |       runs in asyncio.to_thread()
   |   |   |-- objective(trial):
   |   |   |       theta0 = trial.suggest_float("theta0", 0.5, 2.0)
   |   |   |       theta1 = trial.suggest_float("theta1", 0.5, 2.0)
   |   |   |       theta2 = trial.suggest_float("theta2", 0.5, 2.0)
   |   |   |       Run n_simulations of trading simulation
   |   |   |       Return mean negative CARA utility
   |   |   |
   |   |   |-- progress_cb(study, trial):
   |   |   |       progress.current_trial = trial.number + 1
   |   |   |       progress.best_value = study.best_value
   |   |
   |   |-- After study completes: persist best_params to theta_parameters table
   |   |-- progress.categories_completed.append(category)
   |
   |-- progress.running = False
   |-- progress.completed_at = datetime.now(utc)
   |
7. GET /optimize/theta/status reads from app.state.optimization_progress
   |-- Returns JSON with running, categories, trial progress, best values
```

### Flow 3: Mode Switch

```
1. POST /config {"quoting_mode": "as"} received
   |
2. router.update_config() merges into app.state.config_overrides
   |-- config_overrides["quoting_mode"] = "as"
   |
3. Next bot cycle: _process_symbol reads self._config_overrides
   |-- mode = self._config_overrides.get("quoting_mode", "theta") -> "as"
   |-- Takes the A&S path: compute_quote(...)
   |
4. Broadcast includes quotingMode: "as"
```

**Feeds into**: Step 8.

---

## Step 2.8: Build Sequence

*Using all previous steps...*

### Phase 0: Prerequisites (no new features, unblocks everything)

- [ ] **0.1** Modify `Quote` dataclass in `backend/src/engine/quoting.py`: remove `frozen=True`, add 5 optional fields (`xi`, `theta0`, `theta1`, `theta2`, `quoting_mode`) with `None` defaults.
- [ ] **0.2** Update `test_quote_is_frozen_dataclass` in `backend/tests/engine/test_quoting.py` to test that `Quote` is a mutable dataclass with expected fields (or remove the frozen test).
- [ ] **0.3** Add `trade_timestamps: list[float] = field(default_factory=list)` to `MarketState` in `backend/src/engine/book.py`. Update `get_market_state()` to populate it from `Trade.timestamp`.
- [ ] **0.4** Add `PerformativeSettings` to `backend/src/config.py` and compose into `Settings`.
- [ ] **0.5** Add `config_overrides: dict[str, Any]` parameter to `BotLoop.__init__` in `backend/src/engine/loop.py`. Store as `self._config_overrides`. Add `self._symbol_categories` and `self._theta_cache` dict fields. Update `main.py` to pass `config_overrides=app.state.config_overrides`.
- [ ] **0.6** Extend `MarketScanner.scan()` to return `symbol_categories` as the 4th tuple element. Update `_maybe_scan()` in BotLoop to destructure and store `self._symbol_categories`.
- [ ] **0.7** Add new DB models (`ThetaParameter`, `XiEstimateRecord`) and new nullable columns to `Quote` in `backend/src/db/models.py`.
- [ ] **0.8** Set up Alembic: `alembic init -t async`, configure `env.py`, generate initial migration, generate performative migration.

**Verification**: Run existing test suite -- all tests pass (except the frozen-dataclass test which was updated in 0.2). Verify BotLoop still starts. Verify scanner returns 4-tuple.

### Phase 1: Xi Estimation (depends on Phase 0)

- [ ] **1.1** Create `backend/src/engine/xi.py` with `XiEstimate` dataclass and `estimate_xi()` function. Implement OLS via `np.linalg.lstsq`. Handle edge cases: too few trades, flat series (rank-deficient), low r-squared.
- [ ] **1.2** Create `backend/tests/engine/test_xi.py`: synthetic OU data recovery (AC-1.1a, AC-1.1b), insufficient trades fallback (AC-1.2), flat series (AC-1.3), performance < 5ms (AC-1.4), r-squared threshold (AC-1.5), clamping.
- [ ] **1.3** Add `numpy` to `pyproject.toml` dependencies.

**Verification**: `pytest tests/engine/test_xi.py` passes. Benchmark xi estimation on 100 trade prices < 5ms.

### Phase 2: Performative Quoter (depends on Phase 0)

- [ ] **2.1** Create `backend/src/engine/performative.py` with `delta_epsilon()`, `inv_correction()`, and `compute_performative_quote()`. Implement Taylor fallbacks for xi < 1e-6. Include NaN/Inf guard with A&S fallback.
- [ ] **2.2** Create `backend/tests/engine/test_performative.py`: xi->0 degenerates to A&S (AC-2.1, AC-3.1), discount effect visible (AC-2.2), spread >= A&S spread (AC-3.2), [0.01, 0.99] clamping (AC-2.4), bid < ask invariant (AC-3.3), theta scaling, max_spread cap.

**Verification**: `pytest tests/engine/test_performative.py` passes. Benchmark quote computation < 1ms.

### Phase 3: Mode Routing and Live Integration (depends on Phases 0, 1, 2)

- [ ] **3.1** Implement mode routing in `BotLoop._process_symbol()`: read effective mode from config_overrides, branch to `compute_quote` or `compute_performative_quote` based on mode, implement fallback chain.
- [ ] **3.2** Add theta cache to BotLoop: `_load_theta_cache()` private method, called in `_maybe_scan()` after scanner returns.
- [ ] **3.3** Extend `_persist_quote()` to include xi, theta, quoting_mode fields in QuoteRecord.
- [ ] **3.4** Add `_persist_xi_estimate()` method to BotLoop for XiEstimateRecord records (fire-and-forget).
- [ ] **3.5** Extend tick_data dict in `_process_symbol` return to include xi, theta, quoting_mode.
- [ ] **3.6** Extend WebSocket broadcast (market_list construction in lines 208-220) to include xi, theta0-2, quotingMode.

**Verification**: Integration test: start bot in "theta" mode with mock data, verify quotes use performative formulas. Switch mode to "as" via config_overrides, verify next cycle uses A&S. Verify fallback chain with missing theta/insufficient trades.

### Phase 4: Theta Optimizer (depends on Phases 0, 1, 2)

- [ ] **4.1** Create `backend/src/engine/optimizer.py` with `OptimizationProgress` dataclass, simulation objective function, and `run_theta_optimization()` async function.
- [ ] **4.2** Add `optuna` to `pyproject.toml` dependencies.
- [ ] **4.3** Create `backend/tests/engine/test_optimizer.py`: short optimization (10 trials, 10 simulations) produces non-trivial theta, results are persisted, progress tracking works.

**Verification**: Run optimization test, verify theta values differ from (1,1,1), verify DB persistence.

### Phase 5: API Endpoints (depends on Phases 3, 4)

- [ ] **5.1** Extend `ConfigUpdateRequest` in `backend/src/api/router.py` with performative fields (`quoting_mode`, `xi_default`, `xi_min_trades`, `xi_clamp_min`, `xi_clamp_max`, `q_ref`).
- [ ] **5.2** Extend `MarketSummary` and `QuoteHistoryEntry` with xi, theta, quotingMode fields.
- [ ] **5.3** Extend `POST /config` handler to include performative params in effective config response.
- [ ] **5.4** Add `POST /optimize/theta` endpoint: validate state, collect trade data, launch optimizer task. Returns 202.
- [ ] **5.5** Add `GET /optimize/theta/status` endpoint: read from `app.state.optimization_progress`.
- [ ] **5.6** Wire `OptimizationProgress` on `app.state` in `main.py` lifespan.
- [ ] **5.7** Extend `GET /markets` response to include performative fields from QuoteRecord.

**Verification**: API tests: POST /config with quoting_mode, POST /optimize/theta returns 202, GET /optimize/theta/status returns progress.

### Phase 6: Frontend (depends on Phase 5)

- [ ] **6.1** Extend `MarketData` interface in `frontend/src/lib/types.ts` with optional xi, theta0-2, quotingMode fields.
- [ ] **6.2** Display xi and theta values in the market detail view (3 decimal places).
- [ ] **6.3** Display active quoting mode in the status bar with human-readable label.
- [ ] **6.4** Add quoting mode to activity log entries for mode switches and fallbacks.

**Verification**: Dashboard loads without error with and without performative fields in the API response. Mode label displays correctly for all three modes.

### Phase 7: Alembic Migration and Deploy (depends on Phase 5)

- [ ] **7.1** Run `alembic upgrade head` against development database.
- [ ] **7.2** Verify existing quote data is unaffected (new columns are null).
- [ ] **7.3** Run full test suite.
- [ ] **7.4** Docker build and deploy.

---

## Key Architectural Decisions

| Challenge | Solution | Trade-offs | Pattern Reference |
|-----------|----------|------------|-------------------|
| Where to put xi estimation | Separate `xi.py` file | +testability, +dependency isolation; -one more file | One-concern-per-file (scanner.py, risk.py) |
| Where to put performative formulas | Separate `performative.py` file | +quoting.py unchanged, +A&S preserved; -duplicate clamping logic | Stateless function pattern (quoting.py:25) |
| Where to put optimizer | Separate `optimizer.py` file | +async/stateful separated from pure math; -one more file | Async component pattern (scanner.py) |
| Mode routing mechanism | Inline if/elif in _process_symbol | +simple, +fast, +no new abstractions; -grows _process_symbol by ~20 lines | No dispatch abstractions exist in codebase |
| Theta cache | Dict on BotLoop, refreshed in _maybe_scan | +zero new classes; -BotLoop gains a responsibility | BotLoop already caches symbols, titles, expiry_hours |
| Config propagation | Pass dict reference to BotLoop.__init__ | +minimal change, +fixes existing bug; -BotLoop reads two sources | Resolved decision D3 |
| Optimizer concurrency | asyncio.to_thread + create_task | +stdlib, +GIL-friendly for numpy; -single thread | research.md Section 1 |
| OLS implementation | np.linalg.lstsq | +no scipy dep, +residuals for r-squared, +SVD stability | research.md Section 2 |
| Numerical stability | xi < 1e-6 threshold with Taylor series | +exact limits, +zero-cost guard; -threshold is conservative | research.md Section 3 |
| Quote dataclass | Remove frozen, add optional fields | +backward compatible, +simple; -loses immutability | Resolved decision D1 |
| Per-symbol mode | Global only | +simpler; -less flexible | Resolved decision D2 |

---

## Critical Details

### Error Handling

- **Xi estimation failure** (numpy exception, unexpected data): Catch in `_process_symbol`, log WARNING, use default xi, continue in performative mode.
- **Performative quote NaN/Inf**: Guard in `compute_performative_quote` calls `compute_quote` as fallback. Logged at WARNING.
- **Theta DB read failure**: Catch in `_load_theta_cache`, log WARNING, keep previous cache (or empty dict). All symbols use default theta (1,1,1).
- **Optimizer crash**: Catch in the `create_task` wrapper, set `progress.running = False`, log ERROR. Live quoting unaffected.
- **Bid-ask crossing after clamping**: Detected in `compute_performative_quote`. Skip quoting for that symbol, log WARNING.

### State Management

- **Config overrides**: Single mutable dict, referenced by both API handler and BotLoop. No locking needed (single event loop for API; GIL for simple attribute access).
- **Theta cache**: Dict on BotLoop, refreshed every scanner cycle (5 min via `scanner_cycle_seconds`). Stale reads between refreshes are acceptable (theta changes at most once per 24h optimization).
- **Optimization progress**: Dataclass on `app.state`. Written by optimizer thread (via Optuna callback), read by API handler. GIL guarantees atomicity for simple field writes.

### Testing Strategy

- **Unit tests**: `test_xi.py`, `test_performative.py`, `test_optimizer.py` -- each mirrors the source file in `tests/engine/`.
- **Integration tests**: Extend `test_loop.py` with mode routing tests. Mock `estimate_xi` return values to test fallback chains.
- **The frozen-dataclass test**: Update to verify Quote is a (mutable) dataclass with expected fields.
- **Backward compatibility**: Existing `test_quoting.py` tests pass without modification (except the frozen test) because `compute_quote` is unchanged and new Quote fields have defaults.

### Performance

- **Xi estimation**: np.linalg.lstsq on 100x2 matrix: ~50 microseconds. Well under 5ms budget.
- **Quote computation**: Scalar math with exp/log: ~10 microseconds. Well under 1ms budget.
- **Theta optimization**: 100 trials x 100 simulations, ~5-10 minutes per category. Runs in background thread, does not affect cycle time.
- **Theta cache refresh**: Single DB query per scan cycle (every 5 min). Negligible.

### Security

- No new API credentials. All Gemini API calls use existing HMAC auth.
- POST /optimize/theta uses the same access controls as POST /bot/start (no auth middleware exists today).
- Theta results stored locally in PostgreSQL. No sensitive data exposure.
- New config fields validated: quoting_mode must be one of three values, xi clamp ranges must be valid (min < max).

---

## Self-Critique Verification

### Question 1: Does the WIDEN_SPREAD path at loop.py:337-348 break with the Quote changes?

**Verified**. The WIDEN_SPREAD path constructs `Quote(bid_price=..., ask_price=..., ...)` with all 10 original positional fields. Because we are removing `frozen=True` and adding 5 new fields with `None` defaults, the existing construction with 10 positional args continues to work. The new fields default to `None`. No breakage.

**Important**: The WIDEN_SPREAD reconstruction must be updated to copy the performative fields (`xi`, `theta0`, `theta1`, `theta2`, `quoting_mode`) from the original quote as keyword args. Without this, performative metadata is lost when risk widens the spread, causing the dashboard to show `quotingMode: null` and the QuoteRecord to have null theta/xi values for widened quotes.

### Question 2: Does the xi estimator need trade timestamps, and does MarketState currently carry them?

**Verified**. Xi estimation requires `dt = mean(timestamp[i+1] - timestamp[i])` to convert the OLS beta to xi. MarketState currently does NOT carry timestamps (`book.py:54` extracts only prices). Phase 0 task 0.3 explicitly adds `trade_timestamps` to MarketState with a backward-compatible default. The `Trade.timestamp` field exists at `gemini/models.py:34` as `timestamp: int`.

### Question 3: Does the theta cache refresh frequency match the scanner cadence?

**Verified**. The theta cache is refreshed inside `_maybe_scan()`, which runs every `scanner_cycle_seconds` (default 300s = 5 min, per `config.py:60`). Theta optimization runs at most every 24h. A 5-minute stale window is more than acceptable.

### Question 4: Is the fallback chain complete and unambiguous?

**Verified**. The fallback chain is: (1) mode = "theta" requested -> check `self._theta_cache` for category -> if present, use theta-enhanced; if absent, fall back to performative with theta=(1,1,1). (2) Xi estimation -> if insufficient trades or low r-squared, use `xi_default`. (3) If `compute_performative_quote` produces NaN/Inf, fall back to `compute_quote()` (A&S). Each fallback is logged at WARNING level. No gaps.

### Question 5: Are all new imports accounted for in the files that need them?

**Verified**. `loop.py` will need new imports: `from src.engine.xi import estimate_xi`, `from src.engine.performative import compute_performative_quote`. `main.py` needs `from src.engine.optimizer import OptimizationProgress`. `router.py` needs optimizer import for the new endpoints. All specified in the integration mapping.

### Least-to-Most Verification Checklist

- [x] Stage 1 decomposition table is present with all subproblems listed
- [x] Dependencies between subproblems are explicitly stated
- [x] Each Stage 2 step starts with "Using X from Step N..."
- [x] No step references information from a later step (no forward dependencies)
- [x] Final blueprint sections cite their source steps
