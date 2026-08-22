.PHONY: api api-schema backend-check web-check check

api:
	uv run uvicorn pufferlab.main:app --app-dir backend --reload

api-schema:
	uv run python scripts/generate_openapi.py

backend-check:
	uv run ruff check backend scripts
	uv run ruff format --check backend scripts
	uv run mypy
	uv run pytest
	uv run python scripts/generate_openapi.py --check

web-check:
	cd web && pnpm lint
	cd web && pnpm typecheck
	cd web && pnpm test
	cd web && pnpm build

check: backend-check web-check
