# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core pipeline modules (crawler, enrichment, ICP, orchestrator, DB, settings).
- `app/`: FastAPI + LangGraph server (`main.py`, `pre_sdr_graph.py`, `lg_entry.py`).
- `scripts/`: Utilities (e.g., `run_odoo_migration.py`).
- `app/migrations/`: SQL for Odoo-safe migration(s).
- `.env`/`.env.example`: Runtime configuration. Do not commit real secrets.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run API: `uvicorn app.main:app --reload` (serves `/agent` and `/health`).
- LangGraph dev: `langgraph dev` (uses `app/lg_entry.py:make_graph`).
- Orchestrator (CLI): `python src/orchestrator.py`
- DB migration (Odoo): `python scripts/run_odoo_migration.py`

## Coding Style & Naming Conventions
- Python 3.11+; 4‑space indentation; prefer type hints.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Formatting: run `black .` and `isort .` (both in `requirements.txt`).
- Imports: standard → third‑party → local; avoid wildcard imports.

## Testing Guidelines
- Current repo has no formal tests. Prefer `pytest` for new tests.
- Layout: `tests/test_*.py`; mirror `src/` packages; use fixtures for I/O and DB.
- Fast checks: run orchestrator and server locally; validate `/health` and a minimal `/agent` call.
- Optional: add coverage with `pytest --cov=src` when tests exist.

## Repository Navigation & Search
- Use `rg` (ripgrep) for searching across the codebase.
- Avoid `ls -R` and `grep -R`; they are slow in large repositories.

## Commit & Pull Request Guidelines
- Commit style: brief imperative subject; include context in body when needed.
- Prefer Conventional Commits prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`.
- PRs must include: clear description, rationale, before/after notes; link issues; screenshots or logs when UI/API behavior changes.
- Keep PRs focused and small; include any schema or env changes in the description.

## Security & Configuration Tips
- Secrets: never commit `.env`; use `.env.example` for placeholders.
- Required env: `POSTGRES_DSN`, `OPENAI_API_KEY`, `TAVILY_API_KEY`; optional `ZEROBOUNCE_API_KEY`, `ICP_RULE_NAME`, `LANGCHAIN_MODEL`, `TEMPERATURE`.
- Settings load order includes project root and `src/.env`; prefer `POSTGRES_DSN` over `DATABASE_URL`.
