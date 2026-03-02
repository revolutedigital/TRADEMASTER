.PHONY: run stop build migrate seed test lint clean mutation-test

# ---- Development ----

run:  ## Start all services (postgres, redis, backend)
	docker compose up -d postgres redis
	@echo "Waiting for services..."
	@sleep 3
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-docker:  ## Start all services via Docker Compose
	docker compose up -d

run-all:  ## Start all services including monitoring
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

stop:  ## Stop all services
	docker compose down

build:  ## Build Docker images
	docker compose build

# ---- Database ----

migrate:  ## Run database migrations
	cd backend && alembic upgrade head

migrate-create:  ## Create a new migration (usage: make migrate-create MSG="add users table")
	cd backend && alembic revision --autogenerate -m "$(MSG)"

# ---- Data ----

seed:  ## Seed historical data (2 years of BTC and ETH)
	cd backend && python scripts/seed_historical.py

# ---- ML ----

train:  ## Train ML models
	cd backend && python scripts/train_models.py

# ---- Testing ----

test:  ## Run all tests
	cd backend && pytest tests/ -v

test-unit:  ## Run unit tests only
	cd backend && pytest tests/unit/ -v

test-cov:  ## Run tests with coverage
	cd backend && pytest tests/ --cov=app --cov-report=html

mutation-test:  ## Run mutation testing with mutmut
	cd backend && mutmut run --paths-to-mutate=app/ --tests-dir=tests/ || true
	cd backend && mutmut results
	cd backend && mutmut html
	@echo "Mutation testing report generated at backend/html/"

# ---- Code Quality ----

lint:  ## Run linter
	cd backend && ruff check app/ tests/

lint-fix:  ## Auto-fix lint issues
	cd backend && ruff check --fix app/ tests/

format:  ## Format code
	cd backend && ruff format app/ tests/

# ---- Deploy ----

deploy:  ## Deploy to Railway (backend + frontend)
	railway up -s backend --detach
	railway up -s frontend --detach

# ---- Security ----

security-scan:  ## Run security scanning (pip-audit + npm audit)
	cd backend && pip-audit -r requirements.txt || true
	cd frontend && npm audit --audit-level=critical

# ---- Database Backup ----

db-backup:  ## Backup database to local file
	@mkdir -p backups
	pg_dump $(DATABASE_URL) > backups/trademaster_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/"

db-restore:  ## Restore database from backup (usage: make db-restore FILE=backups/file.sql)
	psql $(DATABASE_URL) < $(FILE)
	@echo "Database restored from $(FILE)"

# ---- Utilities ----

clean:  ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

install:  ## Install backend dependencies
	cd backend && pip install -e ".[dev,ml]"

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
