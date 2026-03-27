# Specification Quality Checklist

## Completeness
- [x] Problem statement clearly defined
  - **PASS**: Section 1 clearly articulates three specific deficiencies of the current A&S model: systematic mis-pricing of reservation price, suboptimal spread setting, and missed arbitrage opportunities. The performative feedback loop is explained with reference to the source paper (arXiv:2508.04344).

- [x] All user stories identified
  - **PASS**: Section 2 covers four primary scenarios (normal quoting cycle, fallback chain, theta optimization run, mode switch via API) plus five edge cases (flat price series, extreme xi, bid-ask crossing, empty categories, concurrent optimization/quoting). These cover the happy path, degraded path, operator-initiated actions, and boundary conditions.

- [x] Functional requirements are specific and testable
  - **PASS**: Six functional requirements (FR-1 through FR-6) each include numbered acceptance criteria with concrete thresholds. Example: AC-1.1 specifies recovery of xi within [1.0, 3.5] from synthetic OU data with known xi=2.0. AC-2.1 specifies convergence to A&S within floating-point tolerance of 1e-6. All criteria are directly translatable to automated test assertions.

- [x] Non-functional requirements defined
  - **PASS**: Section 4 covers performance (xi estimation < 5ms, quote computation < 1ms, cycle time < 10s), security (no new credentials, existing HMAC auth, access controls on new endpoints), and reliability (numerical stability guards, backward compatibility, graceful degradation, 64-bit floats).

- [x] API contracts specified
  - **PASS**: Section 6 defines two new endpoints (`POST /optimize/theta`, `GET /optimize/theta/status`) with request/response schemas including HTTP status codes (202, 409, 503). Extended endpoints (`POST /config`, `GET /markets`, WebSocket) have field-level additions documented with types, nullability, and example JSON.

- [x] Error handling described
  - **PASS**: Error handling is covered across multiple sections: FR-1 (insufficient trades fallback, flat price series guard), FR-2 (clamping to [0.01, 0.99]), FR-3 (bid-ask crossing detection and skip), FR-5 (three-level fallback chain with logging), NFR Reliability (overflow guards, L'Hopital limits, NaN prevention), and API error responses (409 Conflict, 503 Service Unavailable).

- [x] Edge cases identified
  - **PASS**: Section 2 "Edge Cases" explicitly lists five scenarios. Additional edge cases are embedded in acceptance criteria: AC-1.3 (flat price series), AC-2.4 (all outputs in bounds), AC-3.3 (bid-ask crossing), AC-4.5 (empty categories API). NFR Reliability adds numerical overflow when `xi * (T-t) > 700`.

## Clarity
- [x] No ambiguous language
  - **PASS**: Requirements use precise numerical thresholds rather than qualitative terms. Example: "under 5 milliseconds" not "fast"; "within [1.0, 3.5]" not "approximately correct"; "100 Optuna trials" not "many trials". One minor exception noted below in additional feedback (FR-4 AC-4.1 "at least one category differ" -- how much differs?).

- [x] Technical terms defined
  - **PASS**: Key terms are defined in context: xi (performative feedback strength), OU process (Ornstein-Uhlenbeck), theta vector (theta0, theta1, theta2), CARA utility (referenced in FR-4), reservation price (formula given explicitly), q_ref (reference inventory). The formulas themselves serve as definitions of the mathematical concepts.

- [x] Examples provided where helpful
  - **PASS**: Section 2 scenarios include concrete example values (xi=1.8, theta=(1.15, 0.92, 1.08), 8 trades for insufficient data). Section 6 includes full JSON examples for all API responses. Section 7 Configuration table gives defaults and valid ranges. AC-3.4 provides a worked example with specific parameter values.

- [x] Acceptance criteria are measurable
  - **PASS**: All acceptance criteria include quantitative bounds: time limits (5ms, 1ms, 10 minutes, 20 seconds), numerical ranges ([0.01, 0.99], [1.0, 3.5]), tolerance (1e-6), trial counts (100), and directional assertions ("strictly less than"). No subjective or unmeasurable criteria found.

## Feasibility
- [x] Technical approach is sound
  - **PASS**: The approach is grounded in a published research paper with well-defined mathematics. The architecture (drop-in replacement for quoting step) is minimally invasive to the existing pipeline. OU regression on 100 data points is a lightweight computation. Optuna for hyperparameter optimization is a mature, well-maintained library. The fallback chain ensures the system never fails to quote.

- [x] Dependencies identified
  - **PASS**: Section 5 "Dependencies" table lists all external APIs (4 Gemini endpoints), new Python packages (Optuna, NumPy), and internal dependencies (existing compute_quote, Quote dataclass, QuoteRecord model, /config endpoint). Each dependency is annotated as "already called" vs "new".

- [x] Risks documented
  - **PASS**: Section 10 identifies five risks with likelihood, impact, and mitigation columns: unreliable xi estimation, theta overfitting, wide spreads, competitor adaptation, and API rate limits. All mitigations reference specific configurable parameters or system features.

- [x] Testing strategy defined
  - **PASS**: Section 8 covers three testing levels: unit tests (4 component areas with specific test descriptions), integration tests (4 end-to-end scenarios), and performance tests (3 benchmarks with specific thresholds). Tests directly map to acceptance criteria.

## Consistency
- [x] No contradictions between sections
  - **PASS**: Cross-referencing key values: price clamping [0.01, 0.99] is consistent across FR-2, FR-3, and the existing codebase. Default xi=0.5 is consistent between FR-1 and Section 7 Configuration. Theta defaults (1,1,1) are consistent between FR-4 and FR-5. Bot cycle time of 10 seconds matches the existing `bot_cycle_seconds` default in `config.py`. One minor inconsistency noted in additional feedback (xi_min_trades).

- [x] Naming conventions consistent
  - **PASS**: Field naming follows existing codebase patterns: snake_case for Python/database (`sigma_sq`, `t_minus_t`, `quoting_mode`), camelCase for API JSON responses (`midPrice`, `quotingMode`). New fields follow the same dual convention. The Quote dataclass extension mirrors existing field naming in `quoting.py`.

- [x] Aligned with existing codebase patterns
  - **PASS** (after revision): All three previously-identified alignment issues have been resolved in the updated spec:
    1. **Quote dataclass**: Spec now includes a "Quote Dataclass Strategy" section that addresses the `@dataclass(frozen=True)` constraint, recommending either removing frozen and adding optional fields with defaults, or creating a wrapper `PerformativeQuote` dataclass.
    2. **Config override propagation**: Spec now includes a "Prerequisite: Config Override Propagation" section documenting that `_process_symbol` must be updated to read from `config_overrides` first, falling back to `self._settings`.
    3. **Scanner category mapping**: FR-4 step 5 now explicitly requires extending `MarketScanner.scan()` to return `symbol_categories: dict[str, str]`. Categories are derived from `Event.category` (not a separate API endpoint).

## Security & Operations
- [x] Security considerations addressed
  - **PASS**: Section 4 Security subsection confirms no new credentials, reuse of existing HMAC auth, local-only storage of theta results, and same access controls for new endpoints as existing `/bot/start` and `/bot/stop`. No new attack surface is introduced.

- [x] Deployment plan outlined
  - **PASS**: Section 9 provides a 5-step deployment sequence (migration, backend, frontend, default behavior, initial optimization trigger). Includes three-tier rollback plan: config change (immediate, no deployment), column drop (database-level), and full code revert.

- [x] Monitoring strategy defined
  - **PASS**: Section 9 Monitoring lists four new monitoring dimensions: xi estimation quality (distribution tracking, clamp boundary alerts), theta convergence (objective value tracking), fallback frequency (degradation detection), and quote computation latency. States all existing monitoring remains in place.
