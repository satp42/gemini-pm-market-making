# Architecture Design: Performative Market Making Engine (Clean Architecture)

> Focus: maintainability, elegant abstractions, clear separation of concerns, testability, extensibility.

---

## Problem Decomposition

To design the Performative Market Making Engine, I will solve these subproblems in order:

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

## Step 1: Requirements Clarification

### Functional Requirements Summary

1. **Xi Estimation (FR-1)**: Fit OU process to trade prices per symbol per cycle. OLS via `np.linalg.lstsq`. Clamp to [0.01, 20.0]. R-squared quality gate < 0.1 triggers fallback. Min 15 trades.
2. **Performative Reservation Price (FR-2)**: Replace A&S `r = s - q*gamma*sigma^2*(T-t)` with `r_perf = s*exp(-xi*(T-t)) - gamma*sigma^2*(q_ref*delta_epsilon - q*(exp(-2*xi*(T-t))-1)/(2*xi))`. Theta-enhanced variant multiplies by theta0, theta1, theta2 respectively.
3. **Performative Spread (FR-3)**: `spread_perf = 2/gamma * ln(1 + gamma/k) - (gamma*sigma^2/(2*xi)) * (exp(-2*xi*(T-t)) - 1)`. Floor at A&S minimum spread. Cap at `max_spread`.
4. **Theta Learning (FR-4)**: Optuna optimization per category. 100 trials x 100 simulations. Background async task via `asyncio.to_thread()`. Persist to DB.
5. **Mode Selection (FR-5)**: Three modes: "as", "performative", "theta". Global mode only (no per-symbol). Fallback chain: theta -> performative -> as.
6. **Dashboard (FR-6)**: Extend API/WebSocket with xi, theta, quotingMode fields.

### Key Constraints (Resolved Decisions)

- Quote dataclass: remove `frozen=True`, add optional fields with defaults
- Global quoting mode only (no per-symbol overrides)
- Config overrides: pass dict reference to `BotLoop`
- Optuna: `asyncio.to_thread()`
- OLS: `np.linalg.lstsq`
- Numerical: `xi < 1e-6` threshold with Taylor fallback

### Non-Functional Requirements

- Xi estimation < 5ms per symbol
- Quote computation < 1ms per symbol
- Bot cycle < 10s for 20 markets
- Theta optimization must not block quoting
- All outputs finite, non-NaN, clamped to [0.01, 0.99]

**Feeds into**: Steps 2, 3, 4, 5, 6, 7, 8

---

## Step 2: Codebase Pattern Analysis

*Using requirements from Step 1...*

### Technology Stack Confirmed

- Python 3.12+, FastAPI, SQLAlchemy async, PostgreSQL, asyncpg
- Pydantic settings with nested `BaseSettings` sub-classes
- Pydantic `BaseModel` for API request/response schemas
- `dataclass` for engine-layer value objects (`Quote`, `MarketState`)
- pytest + pytest-asyncio for testing

### Architectural Patterns Found

| Pattern | Where | Evidence |
|---------|-------|----------|
| **Stateless pure functions for quoting** | `backend/src/engine/quoting.py:25-93` | `compute_quote()` is a pure function; `Quote` is an immutable dataclass |
| **Orchestrator class (BotLoop)** | `backend/src/engine/loop.py:28-518` | `BotLoop` composes all engine components, owns the main loop |
| **Component injection via constructor** | `loop.py:34-49` | `BotLoop.__init__` takes settings, client, session_factory; constructs sub-components |
| **Settings composition** | `config.py:98-113` | Root `Settings` nests `GeminiSettings`, `ASSettings`, `BotSettings`, `RiskSettings`, etc. |
| **Enum for decisions** | `risk.py:13-18` | `RiskAction` enum used by RiskManager, consumed by BotLoop |
| **Fire-and-forget DB persistence** | `loop.py:421-464` | `_persist_quote` catches all exceptions, never blocks the loop |
| **Scanner returns tuples** | `scanner.py:26` | `scan()` returns `tuple[list[str], dict[str, str], dict[str, float]]` |
| **Pydantic schemas for API** | `router.py:30-123` | `StatusResponse`, `MarketSummary`, `ConfigUpdateRequest` etc. |
| **Runtime overrides via dict on app.state** | `main.py:45`, `router.py:554-596` | `app.state.config_overrides = {}`, merged in `/config` endpoint |
| **WebSocket broadcast via callback** | `loop.py:61` | `self.on_tick: Callable` set by API layer |
| **ORM models with DeclarativeBase** | `models.py:11-107` | All models inherit from `Base(DeclarativeBase)` |

### Naming Conventions

- Python modules: `snake_case.py`
- Classes: `PascalCase` (e.g., `BotLoop`, `MarketScanner`, `RiskManager`)
- Functions/methods: `snake_case` (e.g., `compute_quote`, `estimate_variance`)
- Private methods: `_prefix` (e.g., `_process_symbol`, `_maybe_scan`)
- DB table names: `snake_case` plural (e.g., `quotes`, `orders`, `positions`)
- API JSON: `camelCase` (e.g., `midPrice`, `bidPrice`, `activeMarkets`)

### Directory Structure

```
backend/src/
  api/           # FastAPI routers and WebSocket
  db/            # SQLAlchemy models and database utilities
  engine/        # Bot loop, quoting, risk, scanner, etc.
  gemini/        # Gemini API client, auth, models
  config.py      # Pydantic settings
  main.py        # App factory and lifespan
```

### Key Observations for Clean Architecture

1. **No protocol/ABC pattern exists** -- all components are concrete classes. The quoting engine is a bare function, not behind an interface.
2. **BotLoop is monolithic** -- it directly calls `compute_quote()` at `loop.py:306`, hardcoded to A&S. No strategy pattern, no indirection.
3. **Scanner returns bare tuples** -- no typed container. Adding `symbol_categories` means a 4-element tuple, which is fragile.
4. **Config overrides are disconnected** -- `BotLoop` reads `self._settings` but never reads `app.state.config_overrides`. The `/config` endpoint writes to a dict that the loop ignores.
5. **Quote dataclass is frozen** -- must be unfrozen to add optional fields per resolved decision.

**Feeds into**: Steps 3, 4, 5, 6

---

## Step 3: Design Approaches

*Using requirements from Step 1 and patterns from Step 2...*

### Approach 1: Minimal Strategy Pattern with Protocol (Probability: 0.85)

Introduce a `QuotingStrategy` protocol (Python `Protocol` from `typing`) with a single `compute()` method. Implement three strategies: `ASStrategy`, `PerformativeStrategy`, `ThetaStrategy`. A `StrategyRouter` selects the correct one per cycle based on mode + data availability. The Xi estimator and Theta store become standalone modules. New code lives in new files under `engine/`.

- **Pattern alignment**: Follows the stateless-function pattern (each strategy's `compute()` is pure). Extends rather than rewrites the existing `compute_quote`.
- **Pros**: Clean separation of concerns. Easy to add new strategies. Each strategy is independently testable. Protocol enables duck-typing without inheritance overhead.
- **Cons**: Introduces abstraction layer that doesn't exist in the current codebase. Slightly more files and indirection than the codebase's current style.

### Approach 2: Single Extended Function (Probability: 0.82)

Add a `compute_performative_quote()` function alongside the existing `compute_quote()` in `quoting.py`. The mode router is just an if-elif block in `_process_symbol`. Xi estimator is a standalone function in `quoting.py`. Theta store is a dict cache on `BotLoop`.

- **Pattern alignment**: Matches the current "flat functions in quoting.py" pattern exactly.
- **Pros**: Minimal new files. Very consistent with existing code style. Low indirection.
- **Cons**: `quoting.py` becomes very large. No extensibility path. Theta store logic leaks into BotLoop. Hard to test the mode selection independently.

### Approach 3: Subclassed Quoter (Probability: 0.80)

Create a `BaseQuoter` class with `compute()` method, then `ASQuoter`, `PerformativeQuoter`, `ThetaQuoter` subclasses. BotLoop holds a reference to the active quoter and swaps it on mode change.

- **Pattern alignment**: Partial -- the codebase uses classes for stateful components (BotLoop, RiskManager) but stateless functions for pure computation.
- **Pros**: OOP familiarity. Clear polymorphism.
- **Cons**: Inheritance is heavy for what are essentially stateless formula evaluators. Doesn't match the existing functional-style quoting. Swap-on-mode-change adds state management complexity.

### Approach 4: Event-Driven Pipeline (Probability: 0.07)

Replace the sequential `_process_symbol` with an event pipeline where each step (fetch -> estimate xi -> lookup theta -> quote -> risk -> place) is a handler in a pipeline. Each handler can be swapped or skipped.

- **Pattern alignment**: Completely diverges from the current direct-call style.
- **Pros**: Maximum extensibility. Each pipeline stage is independently testable.
- **Cons**: Massive overhaul. Way beyond the scope of this feature. Introduces a framework-within-a-framework pattern that the team hasn't used.

### Approach 5: Functional Composition with Registry (Probability: 0.06)

All strategies are registered as named callables in a dict-based registry. The mode string maps to a callable. Higher-order functions compose xi estimation + theta lookup + quoting into a single pipeline function per mode.

- **Pattern alignment**: Functional-first, which aligns with the pure-function quoting pattern but takes it further than anything in the codebase.
- **Pros**: Very Pythonic. Zero class overhead. Easy to register new strategies.
- **Cons**: Harder to understand for developers used to class-based patterns. Registry indirection can obscure control flow. No IDE support for "find usages" of a strategy.

### Approach 6: Domain Module with Dependency Injection Container (Probability: 0.04)

Introduce a proper DI container (e.g., `dependency-injector` library) to manage component wiring. Define quoting as a domain service with injected dependencies (xi estimator, theta repository, settings).

- **Pattern alignment**: Completely foreign to this codebase.
- **Pros**: Maximum decoupling. Standard enterprise pattern.
- **Cons**: Adds a heavyweight library dependency. Massive conceptual overhead for a small team. The existing codebase uses simple constructor injection, which works fine.

**Feeds into**: Step 4

---

## Step 4: Architecture Decision

*Using Approach 1 from Step 3, patterns from Step 2, and requirements from Step 1...*

### Decision: Approach 1 -- Minimal Strategy Pattern with Protocol

**Rationale**: This approach strikes the best balance for the clean architecture focus:

1. **Testability**: Each strategy is a standalone class with a `compute()` method that takes explicit inputs and returns a `Quote`. No mocks needed for the formula logic itself. The mode router can be tested by injecting mock strategies.

2. **Extensibility**: Adding a new quoting strategy (e.g., a future "bayesian" mode) requires only implementing the protocol and registering it. No changes to BotLoop.

3. **Separation of concerns**: Xi estimation, theta storage, strategy selection, and formula computation are each in their own module. BotLoop remains the orchestrator but delegates quoting decisions to the strategy layer.

4. **Pattern compatibility**: The Protocol approach uses duck-typing, which is consistent with Python's idioms. Each strategy's `compute()` is still a pure computation (matching `compute_quote`'s existing pattern from `quoting.py:25`). The Protocol is structural, not nominal -- existing `compute_quote` could be wrapped in an `ASStrategy` that delegates to it, preserving backward compatibility.

5. **Why not Approach 2**: While it matches existing style, the spec introduces 3 quoting modes + a fallback chain + xi estimation + theta lookup. Cramming this into `quoting.py` and `_process_symbol` would create a 500+ line function and make testing the mode selection logic impossible without running the full bot loop.

6. **Why not Approach 3**: Inheritance is the wrong tool for stateless formula evaluation. The strategies share no state, no lifecycle, and no base implementation. A Protocol captures the contract without the baggage.

### Trade-offs Accepted

| Trade-off | Decision | Justification |
|-----------|----------|---------------|
| More files than current codebase | Accept | Clean separation pays off in testability and readability. Each file is small and focused. |
| Introduces Protocol pattern not in codebase | Accept | Protocol is a stdlib feature, not a framework. It's the lightest possible abstraction. |
| Existing `compute_quote` wrapped in adapter | Accept | Preserves the original function for fallback. No rewrite of proven code. |

**Feeds into**: Steps 5, 6, 7, 8

---

## Step 5: Component Design

*Using the Strategy Pattern decision from Step 4 and patterns from Step 2...*

### Module Layout

```
backend/src/engine/
  quoting.py               # MODIFIED: unfreeze Quote, add optional fields
  strategies/
    __init__.py            # Exports QuotingStrategy protocol, StrategyRouter, QuotingMode
    protocol.py            # QuotingStrategy Protocol + QuotingMode enum + QuotingContext dataclass
    as_strategy.py         # ASStrategy -- wraps existing compute_quote()
    performative.py        # PerformativeStrategy -- vanilla performative formulas
    theta_strategy.py      # ThetaStrategy -- theta-enhanced performative
    router.py              # StrategyRouter -- mode selection + fallback chain
    numerics.py            # Numerical helpers: delta_epsilon(), inv_correction(), XI_EPSILON
  xi/
    __init__.py            # Exports estimate_xi()
    estimator.py           # Xi estimation via OLS (np.linalg.lstsq)
  theta/
    __init__.py            # Exports ThetaStore, ThetaOptimizer
    store.py               # ThetaStore -- in-memory cache backed by DB reads
    optimizer.py           # ThetaOptimizer -- Optuna background optimization
    simulator.py           # Simulation logic for Optuna objective function
  loop.py                  # MODIFIED: inject config_overrides, use StrategyRouter
  scanner.py               # MODIFIED: return ScanResult with symbol_categories

backend/src/
  config.py                # MODIFIED: add PerformativeSettings sub-settings
  db/models.py             # MODIFIED: extend Quote, add ThetaParameter, XiEstimate
  api/router.py            # MODIFIED: extend /config, /markets; add /optimize/theta endpoints
```

### Component Details

#### 5.1 `QuotingStrategy` Protocol

**File**: `backend/src/engine/strategies/protocol.py`

**Responsibilities**:
- Define the contract all quoting strategies must satisfy
- Define `QuotingMode` enum: `AS = "as"`, `PERFORMATIVE = "performative"`, `THETA = "theta"`
- Define `QuotingContext` dataclass: bundles all inputs a strategy needs (mid_price, inventory, gamma, sigma_sq, t_minus_t, k, max_spread, best_bid, best_ask, xi, theta0, theta1, theta2, q_ref)

**Interface**:
```
QuotingStrategy (Protocol):
  def compute(self, ctx: QuotingContext) -> Quote
  @property
  def mode(self) -> QuotingMode

QuotingMode (Enum):
  AS, PERFORMATIVE, THETA

QuotingContext (dataclass):
  mid_price, inventory, gamma, sigma_sq, t_minus_t, k,
  max_spread, best_bid, best_ask,
  xi, theta0, theta1, theta2, q_ref
```

**Dependencies**: `Quote` from `quoting.py`

**Design rationale**: `QuotingContext` avoids 15-parameter function signatures. It's a plain dataclass (not frozen -- strategies may need to adjust defaults). Using a Protocol instead of ABC means any class with a matching `compute()` and `mode` property satisfies the contract without explicit inheritance.

#### 5.2 `ASStrategy`

**File**: `backend/src/engine/strategies/as_strategy.py`

**Responsibilities**:
- Wrap the existing `compute_quote()` function from `quoting.py`
- Always returns `QuotingMode.AS`
- Zero new math -- delegates entirely to the proven implementation

**Dependencies**: `compute_quote` from `quoting.py`, `QuotingContext` and `Quote`

**Design rationale**: Adapts the existing function to the new protocol without rewriting it. This is the fallback endpoint of the chain -- it must be provably identical to current behavior.

#### 5.3 `PerformativeStrategy`

**File**: `backend/src/engine/strategies/performative.py`

**Responsibilities**:
- Implement the vanilla performative reservation price: `r_perf = s * exp(-xi*(T-t)) - gamma * sigma^2 * (q_ref * delta_epsilon - q * inv_correction)`
- Implement performative spread: `spread_perf = 2/gamma * ln(1 + gamma/k) - gamma * sigma^2 * inv_correction`
- Apply max_spread cap, book-spread cap, low-probability anchor, [0.01, 0.99] clamping (same post-processing as existing `compute_quote`)
- Use Taylor fallback for xi < 1e-6 via numerics module
- Always returns `QuotingMode.PERFORMATIVE`

**Dependencies**: `numerics.py`, `QuotingContext`, `Quote`

**Design rationale**: Encapsulates the mathematical formulas in a single file. The strategy uses `theta=(1,1,1)` internally (vanilla performative is just theta-enhanced with identity thetas), but this is an implementation detail -- having a separate class from ThetaStrategy makes the fallback chain explicit and keeps each class under 100 lines.

#### 5.4 `ThetaStrategy`

**File**: `backend/src/engine/strategies/theta_strategy.py`

**Responsibilities**:
- Implement theta-enhanced reservation price: `r_theta = theta0 * s * exp(-xi*(T-t)) - gamma * sigma^2 * (theta1 * q_ref * delta_epsilon - theta2 * q * inv_correction)`
- Same spread formula as PerformativeStrategy (theta does not affect spread)
- Always returns `QuotingMode.THETA`

**Dependencies**: `numerics.py`, `QuotingContext`, `Quote`

**Design rationale**: The mathematical difference from PerformativeStrategy is three multiplications. Rather than parameterizing a single class with "use theta or not", having two classes makes the mode distinction crystal-clear in tests and in the fallback chain. However, to avoid code duplication, both strategies will call shared helper functions from `numerics.py` for the `delta_epsilon` and `inv_correction` terms.

**Implementation note**: ThetaStrategy and PerformativeStrategy will share a private `_compute_performative_quote()` helper function in a shared module (e.g., `numerics.py` or a new `_formulas.py`). PerformativeStrategy calls it with theta=(1,1,1). ThetaStrategy calls it with the provided theta values. This eliminates formula duplication while keeping the strategy classes thin.

#### 5.5 `numerics.py` -- Numerical Helpers

**File**: `backend/src/engine/strategies/numerics.py`

**Responsibilities**:
- `XI_EPSILON = 1e-6` constant
- `delta_epsilon(xi, T)` -- with Taylor fallback
- `inv_correction(xi, T)` -- computes `(exp(-2*xi*T) - 1) / (2*xi)` with Taylor fallback
- `compute_performative_quote(ctx, theta0, theta1, theta2, mode)` -- shared formula implementation used by both PerformativeStrategy and ThetaStrategy
- NaN/Inf guard with logging and A&S fallback

**Dependencies**: `math` stdlib, `QuotingContext`, `Quote`, `compute_quote` (for fallback)

**Design rationale**: Centralizes all numerical stability concerns in one place. The threshold constant and Taylor expansions are co-located with the formulas that use them. The shared `compute_performative_quote()` function prevents formula duplication between the two performative strategies.

#### 5.6 `StrategyRouter`

**File**: `backend/src/engine/strategies/router.py`

**Responsibilities**:
- Accept the configured `QuotingMode`, xi estimate result, and theta availability
- Implement the fallback chain: theta -> performative (if no theta) -> as (if xi estimation failed)
- Return both the selected strategy AND the effective mode (which may differ from configured mode due to fallback)
- Log WARNING at each fallback step

**Interface**:
```
StrategyRouter:
  __init__(self, as_strategy, performative_strategy, theta_strategy)
  def select(self, configured_mode: QuotingMode, xi_available: bool, theta_available: bool) -> QuotingStrategy
```

**Dependencies**: `ASStrategy`, `PerformativeStrategy`, `ThetaStrategy`, `QuotingMode`

**Design rationale**: Separating routing from computation means the fallback chain can be unit-tested with simple boolean inputs -- no need for market data, no need to mock Gemini. The router holds references to all three strategies and simply returns the appropriate one.

#### 5.7 Xi Estimator

**File**: `backend/src/engine/xi/estimator.py`

**Responsibilities**:
- `estimate_xi(trade_prices, trade_timestamps, config) -> XiEstimateResult`
- Fit OLS via `np.linalg.lstsq` on price changes vs. price levels
- Compute r-squared from residuals
- Apply quality gate (r_squared < 0.1 -> fallback)
- Apply min_trades threshold (< 15 -> fallback)
- Clamp to [xi_clamp_min, xi_clamp_max]
- Return a result dataclass containing: xi value, r_squared, num_trades, used_default (bool)

**Result type**:
```
XiEstimateResult (dataclass):
  xi: float
  r_squared: float | None
  num_trades: int
  used_default: bool
```

**Dependencies**: `numpy`, `XiConfig` (subset of performative settings)

**Design rationale**: Returns a rich result object rather than a bare float. The `used_default` flag lets the StrategyRouter know whether xi estimation succeeded, driving the fallback decision. The estimator is completely stateless -- it takes trade data in, returns a result. No database access, no side effects.

#### 5.8 Theta Store

**File**: `backend/src/engine/theta/store.py`

**Responsibilities**:
- In-memory cache of `{category: ThetaVector}` backed by DB reads
- `get_theta(category) -> ThetaVector | None` -- returns cached theta or None
- `refresh()` -- async method that reads all rows from `theta_parameters` table
- `has_theta(category) -> bool` -- quick check for fallback decisions
- Auto-refresh on a configurable interval (default: 60s) to pick up optimization results

**Types**:
```
ThetaVector (NamedTuple):
  theta0: float
  theta1: float
  theta2: float
```

**Dependencies**: `sqlalchemy.ext.asyncio.AsyncSession`, `ThetaParameter` ORM model

**Design rationale**: The store is a thin read-through cache. It avoids hitting the database on every bot cycle (up to 20 symbols x every 10s = 120 queries/minute). The refresh interval is much shorter than the 24h optimization interval, ensuring new theta values are picked up within a minute. Using `NamedTuple` for ThetaVector keeps it lightweight and immutable.

#### 5.9 Theta Optimizer

**File**: `backend/src/engine/theta/optimizer.py`

**Responsibilities**:
- `run_optimization(category, trade_data, config, session_factory, progress) -> ThetaVector`
- Create Optuna study (in-memory storage)
- Define objective function that runs N simulations per trial
- Wrap `study.optimize()` in `asyncio.to_thread()`
- Update `OptimizationProgress` dataclass via callback after each trial
- Persist best parameters to `theta_parameters` table on completion

**Dependencies**: `optuna`, `numpy`, `simulator.py`, `sqlalchemy` session factory

#### 5.10 Theta Simulator

**File**: `backend/src/engine/theta/simulator.py`

**Responsibilities**:
- `simulate_trading(theta, xi, trade_history, gamma, sigma_sq, k, q_ref) -> float`
- Run a single simulated trading session using the performative model
- Compute CARA utility of terminal wealth
- Called N times per Optuna trial; returns mean negative utility

**Dependencies**: `numpy`, `numerics.py` (for performative formula computation)

#### 5.11 `ScanResult` Dataclass

**File**: Defined in `backend/src/engine/scanner.py` (same file, new type)

**Responsibilities**: Replace the fragile 3-tuple return from `scan()` with a typed container.

```
ScanResult (dataclass):
  symbols: list[str]
  symbol_titles: dict[str, str]
  symbol_expiry_hours: dict[str, float]
  symbol_categories: dict[str, str]     # NEW: symbol -> event.category
```

**Design rationale**: The scanner currently returns `tuple[list[str], dict[str, str], dict[str, float]]` (see `scanner.py:26`). Adding a 4th element (`symbol_categories`) would make this a 4-tuple, which is unreadable at call sites. A dataclass makes each field self-documenting and extensible.

#### 5.12 `PerformativeSettings`

**File**: `backend/src/config.py` (new nested settings class)

**Responsibilities**: Group all performative-specific configuration parameters.

```
PerformativeSettings (BaseSettings):
  quoting_mode: str = "theta"
  xi_default: float = 0.5
  xi_min_trades: int = 15
  xi_clamp_min: float = 0.01
  xi_clamp_max: float = 20.0
  q_ref: float = 0.0
  theta_optimization_trials: int = 100
  theta_optimization_simulations: int = 100
  theta_auto_optimize_hours: int = 24
```

Added to `Settings` as: `performative: PerformativeSettings = Field(default_factory=PerformativeSettings)`

#### 5.13 `OptimizationProgress` Dataclass

**File**: `backend/src/engine/theta/optimizer.py` (co-located with optimizer)

```
OptimizationProgress (dataclass):
  running: bool = False
  category: str = ""
  current_trial: int = 0
  total_trials: int = 0
  best_value: float | None = None
  started_at: datetime | None = None
  completed_at: datetime | None = None
```

Stored on `app.state.optimization_progress`. Read by `GET /optimize/theta/status`. Written by the Optuna callback in the optimization thread.

#### 5.14 Database Models

**Modifications to `backend/src/db/models.py`**:

**Quote model** -- add nullable columns:
- `xi: Mapped[float | None]` -- `Numeric(18, 8), nullable=True`
- `theta0: Mapped[float | None]` -- `Numeric(18, 8), nullable=True`
- `theta1: Mapped[float | None]` -- `Numeric(18, 8), nullable=True`
- `theta2: Mapped[float | None]` -- `Numeric(18, 8), nullable=True`
- `quoting_mode: Mapped[str | None]` -- `String(16), nullable=True`

**New: ThetaParameter model**:
- Table: `theta_parameters`
- Columns: `id`, `category` (unique), `theta0`, `theta1`, `theta2`, `xi_value`, `objective_value`, `num_trials`, `optimized_at`, `created_at`, `updated_at`
- Unique constraint on `category`

**New: XiEstimate model**:
- Table: `xi_estimates`
- Columns: `id`, `timestamp`, `symbol`, `xi`, `num_trades`, `r_squared`
- Indexes on `(symbol, timestamp)`

**Feeds into**: Steps 6, 7, 8

---

## Step 6: Integration Mapping

*Using component design from Step 5 and patterns from Step 2...*

### 6.1 Files to Create

| File | Purpose |
|------|---------|
| `backend/src/engine/strategies/__init__.py` | Export `QuotingStrategy`, `StrategyRouter`, `QuotingMode`, `QuotingContext` |
| `backend/src/engine/strategies/protocol.py` | Protocol + QuotingMode + QuotingContext |
| `backend/src/engine/strategies/as_strategy.py` | A&S adapter |
| `backend/src/engine/strategies/performative.py` | Vanilla performative |
| `backend/src/engine/strategies/theta_strategy.py` | Theta-enhanced performative |
| `backend/src/engine/strategies/router.py` | Mode selection + fallback |
| `backend/src/engine/strategies/numerics.py` | Shared formulas + Taylor fallbacks |
| `backend/src/engine/xi/__init__.py` | Export `estimate_xi`, `XiEstimateResult` |
| `backend/src/engine/xi/estimator.py` | OLS xi estimation |
| `backend/src/engine/theta/__init__.py` | Export `ThetaStore`, `ThetaOptimizer`, `ThetaVector` |
| `backend/src/engine/theta/store.py` | DB-backed theta cache |
| `backend/src/engine/theta/optimizer.py` | Optuna optimization runner |
| `backend/src/engine/theta/simulator.py` | Trading simulation for objective |

### 6.2 Files to Modify

#### `backend/src/engine/quoting.py`

- **Line 9**: Change `@dataclass(frozen=True)` to `@dataclass`
- **After line 21** (after `k: float`): Add optional fields:
  ```
  xi: float | None = None
  theta0: float | None = None
  theta1: float | None = None
  theta2: float | None = None
  quoting_mode: str | None = None
  ```
- **No changes to `compute_quote()` or `estimate_variance()`** -- they remain as-is.

#### `backend/src/engine/loop.py`

- **`__init__` (line 34-49)**: Add `config_overrides: dict[str, Any]` parameter. Store as `self._config_overrides`. Add `self._strategy_router: StrategyRouter` initialized with all three strategies. Add `self._theta_store: ThetaStore` initialized with session_factory. Add `self._symbol_categories: dict[str, str] = {}` state field.
- **`_maybe_scan` (line 378-389)**: Unpack `ScanResult` instead of tuple. Store `result.symbol_categories` in `self._symbol_categories`.
- **`_process_symbol` (lines 298-316)**: Replace the direct `compute_quote()` call with:
  1. Read `quoting_mode` from `self._config_overrides` first, fall back to `self._settings.performative.quoting_mode`
  2. Call `estimate_xi()` with `state.trade_prices` (and trade timestamps from MarketState)
  3. Look up theta via `self._theta_store.get_theta(category)` using `self._symbol_categories[symbol]`
  4. Call `self._strategy_router.select(mode, xi_result, theta)` to get the strategy
  5. Build `QuotingContext` from all inputs
  6. Call `strategy.compute(ctx)` to get the `Quote`
- **`_persist_quote` (lines 421-464)**: Pass xi, theta0, theta1, theta2, quoting_mode to `QuoteRecord` constructor.
- **Broadcast section (lines 200-237)**: Add xi, theta0, theta1, theta2, quotingMode to each market data dict.

#### `backend/src/engine/scanner.py`

- **`scan()` return type (line 26)**: Change to `ScanResult`.
- **Inside the event loop (lines 55-101)**: Build `symbol_categories` dict mapping each symbol to `event.category`.
- **Return statement (line 101)**: Return `ScanResult(symbols=selected, symbol_titles=..., symbol_expiry_hours=..., symbol_categories=...)`.

#### `backend/src/engine/book.py`

- **`MarketState` dataclass (line 14)**: Add `trade_timestamps: list[float] = field(default_factory=list)` for xi estimation's dt computation.
- **`get_market_state` (lines 52-54)**: Populate `trade_timestamps` from `trades` response (each `Trade` has a `timestamp` field).

#### `backend/src/config.py`

- **After `AppSettings` (line 96)**: Add `PerformativeSettings` class with all parameters from spec section 7.
- **`Settings` class (line 108-112)**: Add `performative: PerformativeSettings = Field(default_factory=PerformativeSettings)`.

#### `backend/src/db/models.py`

- **`Quote` model (lines 17-41)**: Add 5 nullable columns (xi, theta0, theta1, theta2, quoting_mode).
- **After `PnlSnapshot` (line 107)**: Add `ThetaParameter` and `XiEstimate` models.

#### `backend/src/api/router.py`

- **`MarketSummary` (lines 37-49)**: Add optional fields: `xi`, `theta0`, `theta1`, `theta2`, `quotingMode`.
- **`QuoteHistoryEntry` (lines 51-62)**: Add same optional fields.
- **`ConfigUpdateRequest` (lines 113-123)**: Add performative config fields: `quoting_mode`, `xi_default`, `xi_min_trades`, `xi_clamp_min`, `xi_clamp_max`, `q_ref`, `theta_optimization_trials`, `theta_optimization_simulations`, `theta_auto_optimize_hours`.
- **`update_config` effective dict (lines 565-593)**: Add performative parameters to effective config.
- **New endpoints**: `POST /optimize/theta`, `GET /optimize/theta/status`.
- **`get_markets` (lines 176-224)**: Include xi, theta, quotingMode from Quote records.

#### `backend/src/main.py`

- **Lifespan (lines 62-66)**: Pass `app.state.config_overrides` to `BotLoop.__init__`. Initialize `app.state.optimization_progress = OptimizationProgress()`. Initialize `ThetaStore` and pass to BotLoop.

**Feeds into**: Steps 7, 8

---

## Step 7: Data Flow

*Using components from Step 5 and integration points from Step 6...*

### 7.1 Normal Quoting Cycle (Theta Mode)

```
1. BotLoop._run() starts cycle
       |
2. BotLoop._maybe_scan()
       |---> MarketScanner.scan()
       |        |---> GeminiClient.get_events()
       |        |---> For each event: extract event.category
       |        |---> Return ScanResult(symbols, titles, expiry_hours, categories)
       |---> Store self._symbol_categories
       |
3. BotLoop._process_symbol(symbol, inventories)
       |
       |--a. OrderBookMonitor.get_market_state(symbol)
       |        |---> Returns MarketState(mid_price, best_bid, best_ask, spread,
       |        |     trade_prices, trade_timestamps)
       |
       |--b. Read config_overrides["quoting_mode"] or settings.performative.quoting_mode
       |        |---> QuotingMode.THETA
       |
       |--c. estimate_xi(trade_prices, trade_timestamps, config)
       |        |---> Build design matrix A = [s_n, ones]
       |        |---> np.linalg.lstsq(A, delta_s)
       |        |---> Compute r_squared, extract beta, compute xi = -beta/dt
       |        |---> Clamp to [0.01, 20.0]
       |        |---> Return XiEstimateResult(xi=1.8, r_squared=0.45, num_trades=100, used_default=False)
       |
       |--d. category = self._symbol_categories.get(symbol, "")
       |     theta = self._theta_store.get_theta(category)
       |        |---> Returns ThetaVector(1.15, 0.92, 1.08) or None
       |
       |--e. strategy = self._strategy_router.select(
       |        mode=THETA, xi_available=True, theta_available=True)
       |        |---> Returns ThetaStrategy instance
       |
       |--f. ctx = QuotingContext(mid_price, inventory, gamma, sigma_sq, t_minus_t, k,
       |        max_spread, best_bid, best_ask, xi=1.8,
       |        theta0=1.15, theta1=0.92, theta2=1.08, q_ref=0.0)
       |
       |--g. quote = strategy.compute(ctx)
       |        |---> r_theta = 1.15 * s * exp(-1.8*T) - gamma*sigma^2*(0.92*q_ref*delta_epsilon - 1.08*q*inv_correction)
       |        |---> spread_perf = 2/gamma * ln(1+gamma/k) - gamma*sigma^2*inv_correction
       |        |---> Apply max_spread cap, book spread cap, clamping
       |        |---> Return Quote(bid, ask, reservation, spread, ..., xi=1.8, theta0=1.15, ..., quoting_mode="theta")
       |
       |--h. Risk checks (unchanged) --> possibly WIDEN_SPREAD
       |--i. Order placement (unchanged)
       |--j. Persist quote with xi, theta, mode fields
       |--k. Build tick data with xi, theta, quotingMode for WebSocket broadcast
```

### 7.2 Fallback Chain

```
Configured mode: THETA
       |
       +-- Has theta for category? --NO--> Fallback to PERFORMATIVE (log WARNING)
       |       |
       |       +-- Xi estimation succeeded? --NO--> Fallback to AS (log WARNING)
       |       |       |
       |       |       +-- Use existing compute_quote() (identical to current behavior)
       |       |
       |       +-- YES --> Use PerformativeStrategy with xi, theta=(1,1,1)
       |
       +-- YES --> Use ThetaStrategy with xi, theta=category_theta
```

### 7.3 Theta Optimization Flow

```
1. POST /optimize/theta (API handler)
       |
2. Validate: optimization not already running (check app.state.optimization_progress.running)
       |
3. Derive categories from scanner's cached events (or from symbol_categories on bot_loop)
       |
4. For each category:
       |---> Collect trade data across all symbols in that category
       |        (query xi_estimates + recent trades from DB, or use live data)
       |
5. asyncio.create_task(run_optimization_all_categories(...))
       |
6. Return 202 {"status": "started", "categories": [...]}
       |
7. (Background) For each category sequentially:
       |---> Create Optuna study (in-memory)
       |---> Define objective: for each trial, suggest theta0/1/2, run 100 simulations, return mean neg CARA utility
       |---> await asyncio.to_thread(study.optimize, objective, n_trials=100, callbacks=[progress_callback])
       |---> progress_callback updates OptimizationProgress after each trial
       |---> Persist best theta to theta_parameters table (upsert by category)
       |
8. ThetaStore.refresh() picks up new values on next refresh cycle (within 60s)
       |
9. Next bot cycle: StrategyRouter selects ThetaStrategy with new theta values
```

### 7.4 Config Override Flow

```
1. POST /config {"quoting_mode": "as"}
       |
2. router.py: update_config() merges into app.state.config_overrides
       |---> config_overrides["quoting_mode"] = "as"
       |
3. BotLoop._process_symbol() reads self._config_overrides["quoting_mode"]
       |---> Gets "as"
       |
4. StrategyRouter.select(mode=AS, ...) --> returns ASStrategy
       |
5. ASStrategy.compute(ctx) --> calls existing compute_quote()
       |---> Identical behavior to current production system
```

**Feeds into**: Step 8

---

## Step 8: Build Sequence

*Using all previous steps...*

### Phase 1: Foundation (No behavioral changes)

**Goal**: Establish abstractions and modify data structures without changing any runtime behavior.

- [ ] **1a. Unfreeze Quote dataclass** -- `quoting.py:9`: remove `frozen=True`, add 5 optional fields with `None` defaults. Update `test_quoting.py` to remove the frozen assertion test.
- [ ] **1b. Add `PerformativeSettings`** -- `config.py`: add the new settings class and wire into `Settings`.
- [ ] **1c. Add DB models** -- `models.py`: add nullable columns to `Quote`, add `ThetaParameter` and `XiEstimate` models.
- [ ] **1d. Create Alembic migration** -- `alembic revision --autogenerate -m "add performative market making tables"`. Review and apply.
- [ ] **1e. Add `ScanResult` dataclass** -- `scanner.py`: define dataclass, change `scan()` return type, populate `symbol_categories` from `event.category`. Update `_maybe_scan` in `loop.py` to unpack `ScanResult`.
- [ ] **1f. Extend `MarketState`** -- `book.py`: add `trade_timestamps` field, populate from trade response.
- [ ] **1g. Pass `config_overrides` to BotLoop** -- `main.py:62-66`: add parameter. `loop.py:34`: accept and store. Read overrides first, fall back to settings, for existing A&S params (gamma, k, etc.).

**Verification**: Bot runs identically to current behavior. All existing tests pass. No new quoting logic active.

### Phase 2: Quoting Strategy Layer

**Goal**: Build the strategy abstraction and route through it, initially only using ASStrategy.

- [ ] **2a. Create `strategies/` package** -- `protocol.py` with `QuotingStrategy`, `QuotingMode`, `QuotingContext`.
- [ ] **2b. Implement `ASStrategy`** -- Thin wrapper around `compute_quote()`.
- [ ] **2c. Implement `numerics.py`** -- `delta_epsilon()`, `inv_correction()`, `XI_EPSILON`, `compute_performative_quote()`, NaN guard.
- [ ] **2d. Implement `PerformativeStrategy`** -- Calls `compute_performative_quote()` with theta=(1,1,1).
- [ ] **2e. Implement `ThetaStrategy`** -- Calls `compute_performative_quote()` with provided theta.
- [ ] **2f. Implement `StrategyRouter`** -- Mode selection + fallback chain.
- [ ] **2g. Write unit tests** for all strategies: ASStrategy produces identical output to `compute_quote()`. PerformativeStrategy converges to A&S at xi->0. ThetaStrategy applies theta multipliers correctly. StrategyRouter fallback chain works for all mode + data availability combinations.

**Verification**: All strategy unit tests pass. No integration yet -- BotLoop still uses direct `compute_quote()`.

### Phase 3: Xi Estimation

**Goal**: Build and test the xi estimator independently.

- [ ] **3a. Create `xi/` package** -- `estimator.py` with `estimate_xi()` and `XiEstimateResult`.
- [ ] **3b. Write unit tests** -- Synthetic OU data with known xi. Flat price series. Insufficient trades. R-squared quality gate.
- [ ] **3c. Performance benchmark** -- Verify < 5ms for 100 trade prices.

**Verification**: Xi estimator unit tests pass. Benchmark confirms < 5ms.

### Phase 4: Integration -- Wire Strategy into BotLoop

**Goal**: Replace the direct `compute_quote()` call with the strategy layer.

- [ ] **4a. Modify `_process_symbol`** in `loop.py`:
  - Read quoting mode from config_overrides / settings
  - Call `estimate_xi()` with trade_prices and trade_timestamps
  - Look up category from `self._symbol_categories`
  - Call `self._theta_store.get_theta(category)` (returns None for now -- no theta yet)
  - Call `self._strategy_router.select()` to get strategy
  - Build `QuotingContext`, call `strategy.compute(ctx)`
  - Set xi, theta, mode on the returned Quote object
- [ ] **4b. Update WIDEN_SPREAD path** (loop.py lines 327-344): reconstruct Quote using the new non-frozen approach (direct attribute mutation instead of full reconstruction).
- [ ] **4c. Update `_persist_quote`** to write xi, theta, mode to `QuoteRecord`.
- [ ] **4d. Update WebSocket broadcast** to include xi, theta, quotingMode in market data.
- [ ] **4e. Persist `XiEstimate`** records for observability (fire-and-forget).
- [ ] **4f. Integration test**: Run bot cycle with mode="as" -> verify identical to old behavior. Run with mode="performative" -> verify performative formulas used. Run with mode="theta" -> verify fallback to performative (no theta yet).

**Verification**: Bot runs with strategy layer. Mode="as" is identical to before. Mode="performative" produces different quotes when xi != 0. Fallback chain works.

### Phase 5: Theta Store and Optimization

**Goal**: Build the theta learning pipeline.

- [ ] **5a. Implement `ThetaStore`** in `theta/store.py` -- DB-backed cache with periodic refresh.
- [ ] **5b. Implement `simulator.py`** -- Trading simulation for Optuna objective.
- [ ] **5c. Implement `ThetaOptimizer`** in `theta/optimizer.py` -- Optuna integration with `asyncio.to_thread()`.
- [ ] **5d. Wire ThetaStore into BotLoop** -- Pass to constructor, call `refresh()` periodically.
- [ ] **5e. Write unit tests** -- Simulator produces valid utility values. Optimizer converges theta away from (1,1,1) on synthetic data.
- [ ] **5f. Integration test**: Run short optimization (10 trials, 10 simulations). Verify theta persisted. Verify next cycle picks up new theta.

**Verification**: Theta optimization runs in background. Results persist. Live quoting picks up new theta values.

### Phase 6: API and Dashboard

**Goal**: Expose performative features via REST API and WebSocket.

- [ ] **6a. Extend `ConfigUpdateRequest`** in `router.py` with performative fields.
- [ ] **6b. Extend `update_config`** endpoint to handle new fields in effective config.
- [ ] **6c. Add `POST /optimize/theta`** endpoint -- triggers optimization, returns 202.
- [ ] **6d. Add `GET /optimize/theta/status`** endpoint -- reads `OptimizationProgress`.
- [ ] **6e. Extend `MarketSummary`** and `QuoteHistoryEntry`** with xi, theta, quotingMode fields.
- [ ] **6f. Update `get_markets`** and `get_market_detail`** to include new fields from DB.
- [ ] **6g. Frontend**: Add xi, theta, quotingMode to `MarketData` type. Display in dashboard.
- [ ] **6h. Integration test**: POST /config with quoting_mode change. GET /markets returns new fields. POST /optimize/theta starts optimization.

**Verification**: All API endpoints work. Dashboard displays new fields. End-to-end cycle: optimize -> quoting picks up theta -> API returns theta -> dashboard shows theta.

---

## Key Architectural Decisions

| Challenge | Solution | Trade-offs | Pattern Reference |
|-----------|----------|------------|-------------------|
| Multiple quoting modes need clean dispatch | `QuotingStrategy` Protocol + `StrategyRouter` | Adds abstraction not in current codebase, but enables testable fallback chain | Follows stateless-function pattern (`quoting.py:25`) by making each strategy's `compute()` pure |
| Quote dataclass needs new fields | Remove `frozen=True`, add optional fields with `None` defaults | Loses immutability guarantee, but Quote is never hashed/used as dict key (`loop.py:333-344` already reconstructs it) | Per resolved decision D1 |
| Formula duplication between Performative/Theta strategies | Shared `compute_performative_quote()` in `numerics.py` | Single source of truth for formulas; strategies become thin wrappers | Matches `compute_quote` pattern of "one function, one formula" |
| Config overrides not reaching BotLoop | Pass `config_overrides` dict reference to BotLoop constructor | Mutable shared state, but writes are atomic (GIL) and the dict is small | Per resolved decision D3; matches existing `app.state.config_overrides` pattern (`main.py:45`) |
| Scanner returning fragile tuple | Replace with `ScanResult` dataclass | Breaks existing unpacking in `_maybe_scan`, but one-time migration | Follows `MarketState` pattern (`book.py:14`) of using dataclasses for multi-value returns |
| Theta optimization blocking event loop | `asyncio.to_thread()` wrapping `study.optimize()` | Subject to GIL for pure-Python Optuna overhead, but NumPy releases GIL for heavy computation | Per resolved decision; matches `asyncio.create_task` pattern (`loop.py:99`) |
| Numerical instability at xi -> 0 | Taylor series fallback at `xi < 1e-6` with NaN/Inf guard | Threshold is conservative but costs nothing | Per research.md section 3 |

---

## Critical Details

### Error Handling

- **Xi estimation failure** (exception in numpy): Catch in `_process_symbol`, log WARNING, set `xi_result.used_default = True`, use `xi_default`. The fallback chain in StrategyRouter handles this by selecting PerformativeStrategy with default xi (or ASStrategy if configured mode is "as").
- **Strategy computation produces NaN/Inf**: The `compute_performative_quote()` function in `numerics.py` checks `math.isfinite(reservation_price)` and `math.isfinite(spread)`. On failure: log WARNING, fall back to `compute_quote()` (A&S) for that symbol for that cycle.
- **Theta optimization failure**: Catch exception in the background task. Log ERROR. `OptimizationProgress.running = False`, `completed_at` set. No theta update occurs -- store retains previous values (or None for new categories).
- **Database write failures**: Follow existing fire-and-forget pattern (`loop.py:463`). XiEstimate and ThetaParameter writes are wrapped in try/except with logger.exception. Never block the loop.

### State Management

- **ThetaStore cache**: Refreshed every 60 seconds via an async periodic task (or checked at the start of each bot cycle by comparing `time.monotonic()` to last refresh time -- matching the `_maybe_scan` pattern at `loop.py:378-389`).
- **OptimizationProgress**: Stored on `app.state`. Written by the Optuna callback in the optimization thread. Read by the `/optimize/theta/status` endpoint. Thread safety guaranteed by GIL for simple attribute assignments.
- **Config overrides**: Shared mutable dict between API thread and bot loop. Safe under GIL for dict operations. BotLoop reads at the start of `_process_symbol`, API writes on POST.

### Testing Strategy

| Layer | Test Type | What to Test |
|-------|-----------|-------------|
| `strategies/numerics.py` | Unit | Taylor fallback at xi=1e-7 matches direct computation at xi=0.1 to 6 digits. NaN guard triggers on manufactured Inf input. |
| `strategies/as_strategy.py` | Unit | Output identical to `compute_quote()` for same inputs (parametric test with 10+ input combinations). |
| `strategies/performative.py` | Unit | At xi=1e-8, output matches A&S within 1e-6. At xi=2.0, reservation price < mid_price. Spread >= A&S spread. |
| `strategies/theta_strategy.py` | Unit | With theta=(1,1,1), output matches PerformativeStrategy exactly. With theta=(2,1,1), reservation price is roughly 2x the mid-price discount. |
| `strategies/router.py` | Unit | 9 test cases: 3 modes x 3 data availability states. Verify correct strategy returned and correct mode on Quote. |
| `xi/estimator.py` | Unit | Synthetic OU recovery. Flat series. Insufficient trades. R-squared gate. Clamp bounds. |
| `theta/store.py` | Unit (with DB mock) | get_theta returns cached value. Returns None for unknown category. Refresh loads from DB. |
| `theta/optimizer.py` | Integration | Short run (10 trials, 10 sims). Theta deviates from (1,1,1). Result persisted. |
| `loop.py` integration | Integration | Full cycle with mock Gemini. Mode=as identical. Mode=theta with fallback. Config override changes mode. |

### Performance Considerations

- Xi estimation: `np.linalg.lstsq` on 100x2 matrix is ~50 microseconds. Well under 5ms budget.
- Performative quote: 3 `math.exp` calls + 2 `math.log` calls + arithmetic. Under 10 microseconds.
- ThetaStore lookup: dict get. O(1). Negligible.
- StrategyRouter: 2 boolean checks. Negligible.
- Total overhead per symbol over A&S: ~60 microseconds (dominated by xi estimation numpy call overhead).

### Security Considerations

- No new external API credentials. All Gemini calls use existing HMAC auth.
- `/optimize/theta` endpoint: same access level as `/bot/start`. No additional auth needed per spec.
- Theta parameters contain only numerical optimization results -- no sensitive data.
- Config validation: `quoting_mode` must be one of 3 valid values. Numerical parameters must be finite. `xi_clamp_min < xi_clamp_max` enforced.

---

## Self-Critique Verification

### Verification Questions and Answers

**Q1: Does the decomposition have valid dependency ordering?**
Verified -- the decomposition table shows each subproblem depends only on earlier subproblems. Step 8 (Build Sequence) depends on Steps 5, 6, 7. No forward dependencies exist. Each build phase explicitly lists which earlier phases it requires.

**Q2: Does each step reference previous steps?**
Verified -- Step 2 opens with "Using requirements from Step 1". Step 3: "Using requirements from Step 1 and patterns from Step 2". Step 4: "Using Approach 1 from Step 3, patterns from Step 2, and requirements from Step 1". Steps 5-8 all have "Using..." preambles citing their dependencies.

**Q3: Does the architecture match discovered codebase patterns?**
Verified -- The strategy `compute()` method mirrors the existing `compute_quote()` pure-function pattern (`quoting.py:25`). `ScanResult` dataclass follows the `MarketState` pattern (`book.py:14`). `PerformativeSettings` follows the nested `BaseSettings` pattern (`config.py:32-51`). Fire-and-forget persistence follows `_persist_quote` (`loop.py:421`). The `_maybe_scan` time-check pattern is reused for ThetaStore refresh.

**Q4: Are there any ambiguous "could do X or Y" statements?**
Verified -- all decisions are singular. Protocol (not ABC). `numerics.py` shared function (not duplicated formulas). `ScanResult` dataclass (not 4-tuple). `asyncio.to_thread` (not ProcessPoolExecutor). `np.linalg.lstsq` (not scipy). `xi < 1e-6` threshold (not 1e-10).

**Q5: Can a developer implement from this blueprint alone?**
Verified -- every new file has a specified path, responsibilities, interface, and dependencies. Every modified file has line-number references for where changes go. The build sequence is ordered into 6 phases with checkboxes and verification criteria per phase. The data flow diagrams trace complete paths from input to output for all 4 scenarios.

### Least-to-Most Verification Checklist

- [x] Stage 1 decomposition table is present with all 8 subproblems listed
- [x] Dependencies between subproblems are explicitly stated in the "Depends On" column
- [x] Each Stage 2 step starts with "Using X from Step N..."
- [x] No step references information from a later step
- [x] Final blueprint sections cite their source steps
