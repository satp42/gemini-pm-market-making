# Research: Performative Market Making Engine

This document covers four technical research areas required before implementing
the performative market-making feature described in `spec.md`.

---

## 1. Optuna Async Integration with FastAPI

### Context

The spec requires running Optuna theta-optimization (100 trials x 100
simulations per trial) as a background job triggered via `POST /optimize/theta`.
The existing FastAPI app (`backend/src/main.py`) uses `asynccontextmanager`
lifespan and `asyncio.create_task` for the bot loop. Optuna's
`study.optimize()` is a synchronous, CPU-bound call that will block the event
loop if called directly from an async handler.

### Decision: `asyncio.to_thread()` wrapping a single `study.optimize()` call

Use `asyncio.to_thread(study.optimize, objective_fn, n_trials=100)` to offload
the entire synchronous Optuna study to a thread. This is the simplest correct
approach for this workload.

### Rationale

Three options were evaluated:

| Approach | Pros | Cons |
|---|---|---|
| `asyncio.to_thread()` | One line, stdlib, releases the GIL during NumPy calls inside the objective | Still subject to GIL for pure-Python portions; single thread |
| `ProcessPoolExecutor` via `loop.run_in_executor()` | True parallelism, bypasses GIL entirely | Requires pickling the objective function and all closure state; Optuna study object cannot be shared across processes without an RDB storage backend; adds complexity |
| Separate worker process (e.g., Celery, arq, or `multiprocessing.Process`) | Full isolation, can be scaled horizontally | Massive operational overhead for a single periodic job; requires a message broker or IPC mechanism; overkill for 100 trials |

**Why `asyncio.to_thread()` wins for this project:**

1. The objective function is CPU-bound but dominated by NumPy array operations
   (OLS regression, simulation loops). NumPy releases the GIL during C-level
   computation, so the thread does not actually starve the event loop during the
   heavy parts.
2. The optimization runs at most once every 24 hours (or on manual trigger).
   Even if the GIL causes minor contention for pure-Python Optuna overhead, the
   bot loop runs on a 10-second cycle and can tolerate a few milliseconds of
   jitter.
3. The codebase already uses `asyncio.create_task` for background work (see
   `BotLoop.start()`). Wrapping `to_thread` in a task fits the existing pattern
   perfectly.
4. No serialization issues -- the objective closure captures NumPy arrays and
   floats, all of which work fine in-thread.

### Progress Reporting

Optuna provides a `callbacks` parameter on `study.optimize()`:

```python
def progress_callback(study, trial):
    # Called after each trial completes
    # study.best_value, trial.number, trial.value are available
    pass

study.optimize(objective, n_trials=100, callbacks=[progress_callback])
```

**Recommended pattern for API progress reporting:**

- Store progress in a shared mutable object (e.g., a dataclass on
  `app.state`) that the callback updates after each trial.
- The `GET /optimize/theta/status` endpoint reads from this object.
- Thread safety: a single writer (the callback in the optimization thread) and
  a single reader (the API handler in the event loop thread). For simple
  numeric fields, Python's GIL guarantees atomicity of attribute reads/writes.
  No explicit lock is needed.

```python
@dataclass
class OptimizationProgress:
    running: bool = False
    category: str = ""
    current_trial: int = 0
    total_trials: int = 0
    best_value: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
```

### Storage Backend

| Backend | When to use |
|---|---|
| `None` (in-memory, default) | Single-process, ephemeral studies. Best for this project -- optimization results are persisted to the app's PostgreSQL `theta_parameters` table, not to Optuna's storage. |
| `"sqlite:///optuna.db"` | When you need to resume interrupted studies or share across processes. Not needed here. |
| `"postgresql://..."` (RDBStorage) | Multi-worker distributed optimization. Overkill; adds a second DB dependency for Optuna metadata we do not need to retain. |

**Decision:** Use default in-memory storage. Persist only the final best
parameters to the existing PostgreSQL database via SQLAlchemy.

### Implementation Sketch

```python
async def run_theta_optimization(category: str, trade_data: ..., session_factory: ...):
    progress = OptimizationProgress(running=True, category=category, ...)

    def objective(trial):
        theta0 = trial.suggest_float("theta0", 0.5, 2.0)
        theta1 = trial.suggest_float("theta1", 0.5, 2.0)
        theta2 = trial.suggest_float("theta2", 0.5, 2.0)
        # Run 100 simulations, return mean negative CARA utility
        return simulate(theta0, theta1, theta2, trade_data)

    def on_trial_complete(study, trial):
        progress.current_trial = trial.number + 1
        progress.best_value = study.best_value

    study = optuna.create_study(direction="minimize")

    # Offload synchronous optimize() to a thread
    await asyncio.to_thread(
        study.optimize, objective, n_trials=100, callbacks=[on_trial_complete]
    )

    # Back on the event loop -- persist results
    best = study.best_params
    async with session_factory() as session:
        # upsert theta_parameters row
        ...
```

### Alternatives Considered

- **Optuna's `ask()`/`tell()` API**: Allows driving optimization from an async
  loop (`trial = study.ask(); value = await compute(...); study.tell(trial,
  value)`). Elegant but adds complexity: you must manage the trial loop
  yourself, handle exceptions per-trial, and lose Optuna's built-in pruning
  integration. Not worth it when `to_thread` works.
- **`optuna.integration`**: No FastAPI-specific integration exists. The
  integrations are for ML frameworks (PyTorch, TensorFlow, etc.).

---

## 2. NumPy OLS Regression for OU Parameter Estimation

### Context

The spec (FR-1) requires fitting `delta_s ~ beta * s_n + alpha` via OLS on up
to 100 data points, then extracting `xi = -beta / dt`. The regression must
complete in under 5ms and provide r-squared for the quality gate.

### Decision: `np.linalg.lstsq` with manual r-squared computation

### Rationale

Three options were evaluated:

| Method | Returns r-squared? | Returns residuals directly? | Handles intercept? | Numerical stability | Typical time (100 pts) |
|---|---|---|---|---|---|
| `np.linalg.lstsq(A, y)` | No (but trivial to compute from returned `residuals`) | Yes -- sum of squared residuals | Yes, via column of ones in A | SVD-based, excellent | ~20-40 us |
| `np.polyfit(x, y, 1, full=True)` | No | Yes (with `full=True`) | Implicit (degree-1 poly has intercept) | Vandermonde matrix, adequate for degree 1 | ~25-50 us |
| `scipy.stats.linregress(x, y)` | Yes (`rvalue**2`) | No | Yes | Direct formula | ~15-30 us |

**Why `np.linalg.lstsq` wins:**

1. **No scipy dependency needed.** The project currently has no scipy in
   `pyproject.toml`. Adding scipy (150MB+) for a single function is wasteful.
   NumPy is already an implicit dependency of any numerical Python project and
   will be needed for the simulation arrays regardless.
2. **Direct residual sum of squares.** `lstsq` returns `residuals[0]` which is
   `sum((y - A @ x)^2)`, exactly what is needed for r-squared:
   ```python
   ss_res = residuals[0]
   ss_tot = np.sum((y - y.mean())**2)
   r_squared = 1 - ss_res / ss_tot
   ```
3. **SVD-based solver** is the most numerically stable option. For a 100x2
   matrix this is not performance-critical, but it avoids any conditioning
   issues that `polyfit` can warn about.
4. **Explicit design matrix** makes the regression transparent:
   ```python
   A = np.column_stack([s_n, np.ones(n)])
   (beta, alpha), residuals, rank, sv = np.linalg.lstsq(A, delta_s, rcond=None)
   ```

### Performance Characteristics

For 100 data points with a 100x2 design matrix, `np.linalg.lstsq` performs an
SVD decomposition of a tiny matrix. Based on NumPy benchmarks and the known
O(mn^2) complexity of SVD for an m x n matrix:

- The SVD of a 100x2 matrix is dominated by function call overhead, not
  computation. Expected wall time: **20-50 microseconds** on modern hardware.
- Adding r-squared computation (two `np.sum` calls on 100-element arrays) adds
  roughly 5-10 microseconds.
- **Total: well under 100 microseconds**, far below the 5ms budget.

Even in the worst case (cold cache, Python overhead), this will not exceed 1ms.

### Edge Case: Flat Price Series

When all prices are identical, `delta_s` is all zeros and `s_n` is constant.
The design matrix `A` becomes rank-1 (the `s_n` column and ones column are
linearly dependent when `s_n` is constant). `lstsq` handles this gracefully --
it returns a least-norm solution, and `residuals` will be an empty array (since
rank < n_cols). The implementation must handle this:

```python
if len(residuals) == 0:
    # Rank-deficient: no meaningful regression
    r_squared = 0.0
    beta = 0.0
```

This triggers the r-squared quality gate (r_squared < 0.1), which falls back
to `xi_default`. No special handling beyond the quality gate is needed.

### Alternatives Considered

- **`scipy.stats.linregress`**: Returns r-value directly, but adds a heavy
  dependency. Also does not handle the design-matrix formulation as cleanly
  (it assumes simple `y ~ a*x + b`, which is what we need, but `lstsq` is
  more explicit about what is being fitted).
- **Manual normal equations (`(A^T A)^{-1} A^T y`)**: Faster by ~2x for tiny
  matrices due to lower overhead, but numerically inferior. For a 2x2 system
  this is fine in practice, but there is no reason to sacrifice robustness
  when `lstsq` is already sub-millisecond.
- **`np.polyfit`**: Adequate but designed for polynomial fitting. The `lstsq`
  formulation is more semantically clear for "OLS regression with explicit
  regressors."

---

## 3. Numerical Stability for Performative Formulas

### Context

The performative model introduces three formula components with potential
numerical issues:

1. `exp(-xi*(T-t))` -- underflow when the exponent is very negative
2. `delta_epsilon = (1 - exp(-xi*T) - xi*T*exp(-xi*T)) / xi^2` -- 0/0 as xi->0
3. `(exp(-2*xi*T) - 1) / (2*xi)` -- 0/0 as xi->0

### Analysis

#### 3a. Exponential Underflow: `exp(-xi*(T-t))`

IEEE 754 double-precision `exp(x)` underflows to exactly 0.0 (not NaN, not
error) when `x < -745.13`. Given that `xi` is clamped to [0.01, 20.0] and
`T-t` is at most ~24.0 (24 hours normalized to 1 day), the maximum exponent
magnitude is `20.0 * 24.0 = 480`, well within the safe range.

**However**, `exp(-2*xi*(T-t))` doubles the exponent to 960, which exceeds the
underflow threshold. When `xi=20` and `T-t=24`, `exp(-960)` underflows to 0.0.

**Decision:** Underflow to 0.0 is numerically correct behavior for these
formulas. When `exp(-xi*T) -> 0`:

- The reservation price's mid-price component `s * exp(-xi*T) -> 0`, meaning
  the price is entirely driven by inventory correction. This is correct: very
  strong mean reversion makes the current mid-price irrelevant.
- The spread correction `(exp(-2*xi*T) - 1)/(2*xi) -> -1/(2*xi)`, which is
  well-defined.

**No guard is needed for underflow.** The formulas degrade gracefully. Use
`math.exp` (or `np.exp`) as-is.

#### 3b. L'Hopital Limits as xi -> 0

All three formula components have 0/0 indeterminate forms at xi=0. The limits
must be computed analytically and used when `xi` is below a threshold.

**Component 1: `delta_epsilon = (1 - exp(-xi*T) - xi*T*exp(-xi*T)) / xi^2`**

Let `u = xi * T`. Taylor expand `exp(-u)` around u=0:

```
exp(-u) = 1 - u + u^2/2 - u^3/6 + ...
u * exp(-u) = u - u^2 + u^3/2 - ...

1 - exp(-u) - u*exp(-u)
  = 1 - (1 - u + u^2/2 - u^3/6 + ...) - (u - u^2 + u^3/2 - ...)
  = u - u^2/2 + u^3/6 - u + u^2 - u^3/2 + ...
  = u^2/2 - u^3/3 + ...

Dividing by xi^2 = u^2/T^2:
  delta_epsilon = T^2 * (1/2 - u/3 + u^2/8 - ...)
```

**Limit as xi -> 0: `delta_epsilon = T^2 / 2`**

This is exactly the A&S inventory correction's time component `(T-t)^2 / 2`,
confirming that the performative model degenerates to A&S.

**Component 2: `(exp(-2*xi*T) - 1) / (2*xi)`**

Let `v = 2*xi*T`. Taylor expand:

```
exp(-v) = 1 - v + v^2/2 - ...
exp(-v) - 1 = -v + v^2/2 - v^3/6 + ...
            = -2*xi*T + (2*xi*T)^2/2 - ...

Dividing by 2*xi:
  = -T + xi*T^2 - (2/3)*xi^2*T^3 + ...
```

**Limit as xi -> 0: `(exp(-2*xi*T) - 1) / (2*xi) = -T`**

Substituting into the spread formula:
```
spread_perf = 2/gamma * ln(1 + gamma/k) - gamma*sigma^2/(2*xi) * (exp(-2*xi*T) - 1)
            -> 2/gamma * ln(1 + gamma/k) + gamma*sigma^2*T
```
Which equals the standard A&S spread `gamma*sigma^2*(T-t) + 2/gamma * ln(1 + gamma/k)`.
This confirms correct degeneration.

**Component 3: Reservation price inventory term `-q * gamma * sigma^2 * (exp(-2*xi*T) - 1) / (2*xi)`**

Uses the same limit from Component 2. As xi -> 0:
```
-q * gamma * sigma^2 * (-T) = q * gamma * sigma^2 * T
```
Wait -- the A&S inventory correction is `-q * gamma * sigma^2 * (T-t)`. Let us
verify sign conventions carefully:

The performative reservation price inventory term is:
```
-gamma * sigma^2 * (q_ref * delta_epsilon - q * (exp(-2*xi*T) - 1) / (2*xi))
```

With q_ref=0 (default), this becomes:
```
-gamma * sigma^2 * (-q * (exp(-2*xi*T) - 1) / (2*xi))
= gamma * sigma^2 * q * (exp(-2*xi*T) - 1) / (2*xi)
```

As xi -> 0:
```
= gamma * sigma^2 * q * (-T)
= -q * gamma * sigma^2 * T
```

This matches the A&S formula `-q * gamma * sigma^2 * (T-t)`. Confirmed correct.

### Decision: Use a xi threshold with Taylor-series fallback

**Recommended threshold: `xi < 1e-6`**

When `xi < 1e-6`, substitute the analytical limits:

```python
XI_EPSILON = 1e-6

def delta_epsilon(xi: float, T: float) -> float:
    if abs(xi) < XI_EPSILON:
        return T * T / 2.0
    u = xi * T
    return (1.0 - math.exp(-u) - u * math.exp(-u)) / (xi * xi)

def inv_correction(xi: float, T: float) -> float:
    """Computes (exp(-2*xi*T) - 1) / (2*xi)."""
    if abs(xi) < XI_EPSILON:
        return -T
    return (math.exp(-2.0 * xi * T) - 1.0) / (2.0 * xi)
```

**Why 1e-6 and not a larger threshold?**

At `xi = 1e-6` with `T = 24.0`, the direct computation gives:
- `u = 2.4e-5`, `exp(-u) = 1 - 2.4e-5 + ...` -- perfectly representable in
  double precision.
- Catastrophic cancellation in `1 - exp(-u)` loses about 5 digits of
  precision (since exp(-u) differs from 1 by ~2.4e-5 out of 16 digits).
- This still leaves 11 digits of precision, more than adequate.

At `xi = 1e-10`, the cancellation becomes severe (loses 10 digits), leaving
only 6 digits. At `xi = 1e-14`, the subtraction `1 - exp(-u)` produces
zero due to floating-point granularity. Hence `1e-6` provides a comfortable
safety margin.

### Summary Table

| Formula Component | Direct Form | Limit (xi -> 0) | Threshold |
|---|---|---|---|
| `exp(-xi*T)` | `math.exp(-xi * T)` | `1.0` | No guard needed (underflow to 0.0 is correct) |
| `delta_epsilon` | `(1 - exp(-u) - u*exp(-u)) / xi^2` | `T^2 / 2` | `abs(xi) < 1e-6` |
| Inventory correction | `(exp(-2*xi*T) - 1) / (2*xi)` | `-T` | `abs(xi) < 1e-6` |
| Spread correction | `-gamma*sigma^2 * inv_correction / 1` | `gamma*sigma^2*T` | Same as inventory correction |

### Additional Guard: NaN/Inf Catch

As a belt-and-suspenders measure, add a final guard after all computations:

```python
if not math.isfinite(reservation_price):
    logger.warning("Non-finite reservation price, falling back to A&S")
    return compute_quote(...)  # existing A&S function
```

This protects against any unforeseen combination of extreme inputs.

### Alternatives Considered

- **Arbitrary-precision math (mpmath)**: Unnecessary. The formulas are
  well-conditioned everywhere except near xi=0, and the Taylor-series fallback
  handles that region exactly.
- **Clamping xi to a minimum of 0.01 (eliminating the xi->0 path)**: The spec
  already clamps xi to [0.01, 20.0], so `xi < 1e-6` can only occur if the
  config is changed or a future code path passes xi=0 directly. The guard
  costs nothing and protects against future mistakes.
- **Using `numpy.expm1(-u)` for `exp(-u) - 1`**: This is valid and avoids
  cancellation for the inventory correction term. However, `delta_epsilon`
  cannot be expressed using `expm1` alone (it has `1 - exp(-u) - u*exp(-u)`),
  so you would need the Taylor fallback anyway. Using a consistent approach
  (threshold + Taylor) for both terms is simpler.

---

## 4. Alembic with Async SQLAlchemy

### Context

The project uses:
- SQLAlchemy 2.0+ with async engine (`create_async_engine`, `asyncpg`)
- PostgreSQL via `asyncpg` driver
- Alembic 1.14+ (already in `pyproject.toml`)
- Models defined via `DeclarativeBase` in `backend/src/db/models.py`
- Current schema creation: `Base.metadata.create_all` in `init_db()` (see
  `backend/src/db/database.py` lines 49-54)

The current approach (`create_all`) does not support migrations. Adding new
columns (e.g., `xi`, `theta0`, `quoting_mode` on the quotes table) or new
tables (`theta_parameters`, `xi_estimates`) requires Alembic.

### Decision: Use Alembic's built-in async migration template

Alembic 1.14+ ships with an async-aware migration template that handles async
engines natively. Use `alembic init -t async` to scaffold, then configure to
use the project's existing async engine.

### Rationale

Alembic has provided first-class async support since version 1.7 (released
2021). The async template generates an `env.py` that uses
`connectable.connect()` with `run_sync()` to bridge the async engine into
Alembic's synchronous migration runner. This is the officially supported
pattern.

### Setup Steps

**Step 1: Initialize Alembic in the backend directory**

```bash
cd backend
alembic init -t async alembic
```

This creates:
- `alembic.ini` -- Alembic configuration file
- `alembic/` -- migration directory with async-aware `env.py`

**Step 2: Configure `alembic.ini`**

Set `sqlalchemy.url` to empty (we will override from the app's settings):

```ini
[alembic]
script_location = alembic
sqlalchemy.url =
```

**Step 3: Configure `alembic/env.py` for the project**

The async template generates an `env.py` with this structure. Key
modifications needed:

```python
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config, create_async_engine

# Import the project's Base and settings
from src.db.models import Base
from src.config import get_settings

# Alembic Config object
config = context.config

# Set up logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target_metadata for autogenerate support
target_metadata = Base.metadata


def get_url() -> str:
    """Get database URL from application settings."""
    settings = get_settings()
    return settings.database.url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL scripts)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations using a synchronous connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with an async engine."""
    connectable = create_async_engine(get_url(), poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Key points:**
- `pool.NullPool` is used for migrations to avoid connection pool issues during
  schema changes.
- `connection.run_sync(do_run_migrations)` bridges the async connection into
  Alembic's synchronous `context.configure()` API.
- `asyncio.run()` is used at the top level because `alembic` CLI runs outside
  of an event loop.
- `target_metadata = Base.metadata` enables autogeneration.
- `get_url()` reads from the project's existing `Settings` class, keeping the
  database URL in one place.

**Step 4: Auto-generate the initial migration**

```bash
cd backend
alembic revision --autogenerate -m "initial schema"
```

This compares `Base.metadata` against the live database and generates a
migration file with `create_table` operations for all existing tables. Review
the generated migration before applying.

**Step 5: Apply migrations**

```bash
alembic upgrade head
```

**Step 6: Remove `create_all` from application startup**

Once Alembic manages the schema, remove lines 49-54 from
`backend/src/db/database.py`:

```python
# DELETE THIS:
from src.db.models import Base
async with _engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
```

Schema changes are now exclusively managed by Alembic migrations.

### Generating Migrations for the Performative Feature

After adding the new models (`ThetaParameter`, `XiEstimate`) and new columns
to `Quote` in `backend/src/db/models.py`:

```bash
alembic revision --autogenerate -m "add performative market making tables"
```

Alembic's autogenerate will detect:
- New tables: `theta_parameters`, `xi_estimates`
- New nullable columns on `quotes`: `xi`, `theta0`, `theta1`, `theta2`,
  `quoting_mode`

Review the generated migration to ensure:
- New columns are `nullable=True` (backward compatible with existing rows)
- Index creation is included for new tables
- No destructive operations (column drops, type changes) are generated

### Running Migrations in Deployment

```bash
# In CI/CD pipeline, before starting the application:
cd backend
alembic upgrade head
```

For containerized deployments, run Alembic as an init container or as a
pre-start command:

```dockerfile
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000"]
```

### Alternatives Considered

| Approach | Pros | Cons |
|---|---|---|
| Alembic async template (chosen) | Official support, autogenerate, well-documented | Requires `env.py` configuration |
| Alembic with sync engine for migrations only | Simpler env.py | Requires a second `psycopg2` dependency just for migrations; the project only has `asyncpg` |
| Keep `create_all` and manage changes manually | Zero setup | No rollback, no migration history, no autogenerate; will break when adding nullable columns to existing tables with data |
| Raw SQL migration scripts | Full control | No autogenerate, error-prone, no revision tracking |

**Why the async template wins:**
- The project already depends on `asyncpg` exclusively. Using the async
  template avoids adding `psycopg2` as a migration-only dependency.
- Autogenerate detects model changes and produces migration scripts
  automatically, reducing human error.
- Alembic's revision chain provides rollback capability (`alembic downgrade -1`).

### Important Caveats

1. **Autogenerate limitations**: Alembic autogenerate does not detect all
   changes. It does not detect: column name changes (appears as drop+add),
   changes to `server_default` values, or changes to check constraints. Always
   review generated migrations.

2. **`asyncpg` and DDL**: asyncpg handles DDL statements fine. Alembic's
   `run_sync` bridge works correctly because DDL is executed through
   SQLAlchemy's connection abstraction, not through asyncpg directly.

3. **Multiple heads**: If multiple developers create migrations in parallel,
   Alembic will detect multiple heads. Resolve with `alembic merge heads -m
   "merge"`.

4. **Testing**: Run `alembic upgrade head` followed by `alembic downgrade base`
   followed by `alembic upgrade head` in CI to verify migrations are reversible.

---

## Verification Summary

| Research Area | Verification Question | Confidence | Notes |
|---|---|---|---|
| **Optuna + FastAPI** | Is `asyncio.to_thread` the right approach? | **High** | Confirmed by Python 3.12 docs and Optuna's synchronous `study.optimize()` API. NumPy releases GIL, so thread contention is minimal. |
| **Optuna + FastAPI** | Progress reporting mechanism? | **High** | Optuna's `callbacks` parameter is documented in the API reference and is the standard mechanism. |
| **Optuna + FastAPI** | Storage backend choice? | **High** | In-memory is correct when results are persisted externally. Confirmed by Optuna docs listing storage options. |
| **NumPy OLS** | Is `lstsq` the best choice? | **High** | Confirmed by NumPy docs. Returns residuals directly. No scipy dependency needed. |
| **NumPy OLS** | Performance under 5ms? | **High** | SVD of 100x2 matrix is dominated by call overhead (~20-50us). Orders of magnitude under budget. |
| **NumPy OLS** | Edge case handling (flat series)? | **High** | `lstsq` returns empty residuals array for rank-deficient matrices. Handled by checking `len(residuals) == 0`. |
| **Numerical stability** | L'Hopital limits correct? | **High** | Derived analytically via Taylor expansion; all three limits confirmed to degenerate to A&S formulas. |
| **Numerical stability** | Threshold value (1e-6)? | **Medium** | Conservative choice. Could be as low as 1e-10 without precision loss for typical T values, but 1e-6 provides a wider safety margin at zero cost. |
| **Numerical stability** | Underflow behavior? | **High** | IEEE 754 guarantees underflow to 0.0, not NaN. Verified that formulas remain well-defined when exp terms are 0. |
| **Alembic async** | Is the async template the right approach? | **High** | Official Alembic documentation since 1.7. The `run_sync` bridge pattern is the canonical solution. |
| **Alembic async** | Autogenerate works with async engines? | **High** | Autogenerate operates on the synchronous connection provided by `run_sync`. The async/sync distinction is transparent to the migration generator. |
| **Alembic async** | No need for psycopg2? | **High** | The async template uses `create_async_engine` with the existing `asyncpg` driver. No sync driver needed. |

### Limitations

- **Optuna benchmarks**: Actual optimization time (100 trials x 100
  simulations) depends on simulation complexity, which is not yet implemented.
  The 5-10 minute estimate from the spec should be validated with a prototype.
- **NumPy benchmarks**: Performance numbers are estimated from known complexity
  and typical hardware. Exact microsecond timings will vary. The conclusion
  (well under 5ms) is robust.
- **Numerical threshold**: The `1e-6` threshold was chosen analytically. It
  could be validated with a sweep test comparing direct computation vs. Taylor
  approximation across xi values from 1e-10 to 1e-1.

---

## 5. Resolved Design Decisions

### D1: Quote Dataclass Strategy
- **Decision**: Option A — Remove `frozen=True`, add optional fields with defaults
- **Rationale**: The codebase never uses `Quote` as a hashable type. Default-valued optional fields (`xi: float | None = None`, etc.) are backward compatible with all existing construction sites including the WIDEN_SPREAD path in `loop.py:333-344`.

### D2: Per-Symbol Mode Override
- **Decision**: Option A — Global mode only
- **Rationale**: The fallback chain (theta -> performative -> A&S) already handles per-symbol data availability. A symbol with sparse trade data falls back to A&S automatically. Per-symbol mode overrides add complexity without benefit for hackathon scope.

### D3: Config Override Propagation Fix
- **Decision**: Option A — Pass `app.state.config_overrides` dict reference to `BotLoop.__init__`
- **Rationale**: Minimal change. The bot loop gets a mutable dict reference that the API updates. `_process_symbol` reads from overrides first, falls back to `self._settings`. Quick targeted fix that unblocks runtime mode switching.

### D4: Simulation Objective Function
- **Decision**: Backtest simulation replaying historical prices
- **Details**: Fetch N=200 trade prices per symbol. Replay mid-price series step-by-step. Compute performative bid/ask with trial theta. Fill via Poisson: `P(fill) = 1 - exp(-A * exp(-k * delta) * dt)`. Track inventory/cash/PnL. Return `mean(-exp(-gamma * PnL[-1]))` as CARA utility.
- **Rationale**: Matches the paper's simulation methodology. Uses actual market data for backtesting.

### D5: WIDEN_SPREAD Field Propagation
- **Decision**: Copy `xi`, `theta0`, `theta1`, `theta2`, `quoting_mode` from original quote to widened Quote reconstruction
- **Rationale**: Without this, performative metadata is lost when risk widens the spread, causing null values in the DB and dashboard.

### D6: Theta Optimization Schedule
- **Decision**: Manual trigger only via `POST /optimize/theta`. No auto-schedule timer.
- **Rationale**: Hackathon scope. Auto-schedule can be added later as a simple `asyncio.create_task` with sleep loop.

---

## 6. Key Codebase Files Reference

| File | Path | Relevance |
|------|------|-----------|
| Quoting Engine | `backend/src/engine/quoting.py` | Drop-in replacement target. `compute_quote()` + `Quote` dataclass |
| Bot Loop | `backend/src/engine/loop.py` | `_process_symbol` is the integration point (lines 255-368) |
| Config | `backend/src/config.py` | Add `PerformativeSettings` sub-settings |
| Scanner | `backend/src/engine/scanner.py` | Extend `scan()` to return `symbol_categories` |
| Risk Manager | `backend/src/engine/risk.py` | Unchanged but WIDEN_SPREAD path reconstructs Quote |
| DB Models | `backend/src/db/models.py` | Add columns to `Quote`, new tables |
| API Router | `backend/src/api/router.py` | Extend `/config`, add `/optimize/theta` endpoints |
| Gemini Models | `backend/src/gemini/models.py` | `Event.category` at line 95 (confirmed) |
| Frontend Types | `frontend/src/lib/types.ts` | Add `xi`, `theta*`, `quotingMode` to `MarketData` |
| Main | `backend/src/main.py` | Wire config_overrides to BotLoop |
