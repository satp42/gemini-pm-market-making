# Project Constitution

## Project Overview
Gemini Prediction Markets Market-Making Bot — an automated market maker for Gemini prediction markets using quantitative trading strategies.

## Technology Stack
- **Backend**: Python 3.12+, FastAPI, SQLAlchemy (async), PostgreSQL, asyncpg
- **Frontend**: Next.js 16, React 19, TypeScript, TailwindCSS, Recharts
- **Infrastructure**: Docker Compose, Uvicorn

## Coding Principles
1. **Type safety**: Use type hints in Python, strict TypeScript
2. **Async-first**: All I/O operations are async
3. **Pure computation**: Quoting logic is stateless and pure (no side effects)
4. **Config-driven**: All tunable parameters in settings, overridable at runtime
5. **Test coverage**: Unit tests for all engine logic, integration tests for API
6. **Graceful degradation**: Always fall back safely; never crash the bot loop

## Architecture Patterns
- Engine components are stateless functions or lightweight classes
- BotLoop orchestrates the cycle: scan -> fetch -> quote -> risk check -> place orders
- Database persistence is fire-and-forget (logging, not blocking)
- WebSocket broadcasts real-time state to the dashboard
- Gemini API client handles auth, retries, and rate limiting

## Conventions
- Python: snake_case for variables/functions, PascalCase for classes
- API JSON: camelCase for response fields
- Database: snake_case for columns
- Tests: pytest, pytest-asyncio, files mirror src structure under tests/
