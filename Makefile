.PHONY: help test test-unit test-contract test-property test-cov lint lint-fix format staging staging-down dev-backend dev-frontend build deploy health emergency-stop run run-docker run-all stop migrate migrate-create seed train mutation-test security-scan db-backup db-restore clean install load-test load-test-headless

# ---- Help ----

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

# ---- Development ----

run: ## Start all services (postgres, redis, backend)
	docker compose up -d postgres redis
	@echo "Waiting for services..."
	@sleep 3
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-docker: ## Start all services via Docker Compose
	docker compose up -d

run-all: ## Start all services including monitoring
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

stop: ## Stop all services
	docker compose down

build: ## Build Docker images
	docker compose build

dev-backend: ## Start backend in development mode
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Start frontend in development mode
	cd frontend && npm run dev

# ---- Database ----

migrate: ## Run database migrations
	cd backend && alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="add users table")
	cd backend && alembic revision --autogenerate -m "$(MSG)"

# ---- Data ----

seed: ## Seed historical data (2 years of BTC and ETH)
	cd backend && python scripts/seed_historical.py

# ---- ML ----

train: ## Train ML models
	cd backend && python scripts/train_models.py

# ---- Testing ----

test: ## Run all backend tests
	cd backend && python3 -m pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	cd backend && python3 -m pytest tests/unit/ -v --tb=short

test-contract: ## Run contract tests
	cd backend && python3 -m pytest tests/contract/ -v --tb=short

test-property: ## Run property-based tests
	cd backend && python3 -m pytest tests/property/ -v --tb=short

test-cov: ## Run tests with coverage
	cd backend && pytest tests/ --cov=app --cov-report=html

mutation-test: ## Run mutation testing with mutmut
	cd backend && mutmut run --paths-to-mutate=app/ --tests-dir=tests/ || true
	cd backend && mutmut results
	cd backend && mutmut html
	@echo "Mutation testing report generated at backend/html/"

# ---- Code Quality ----

lint: ## Run linting (backend + frontend)
	cd backend && python3 -m ruff check app/ tests/
	cd frontend && npm run lint

lint-fix: ## Auto-fix lint issues
	cd backend && ruff check --fix app/ tests/

format: ## Format code
	cd backend && ruff format app/ tests/

# ---- Staging ----

staging: ## Start staging environment
	docker compose -f docker-compose.staging.yml up --build

staging-down: ## Stop staging environment
	docker compose -f docker-compose.staging.yml down

# ---- Deploy ----

deploy: ## Deploy to Railway
	git push origin main
	railway redeploy -s backend --yes

health: ## Check production health
	@curl -s https://backendtrademaster.up.railway.app/api/v1/system/health | python3 -m json.tool

# ---- Security ----

security-scan: ## Run security scanning (pip-audit + npm audit)
	cd backend && pip-audit -r requirements.txt || true
	cd frontend && npm audit --audit-level=critical

# ---- Database Backup ----

db-backup: ## Backup database to local file
	@mkdir -p backups
	pg_dump $(DATABASE_URL) > backups/trademaster_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/"

db-restore: ## Restore database from backup (usage: make db-restore FILE=backups/file.sql)
	psql $(DATABASE_URL) < $(FILE)
	@echo "Database restored from $(FILE)"

# ---- Emergency ----

emergency-stop: ## Emergency stop trading engine
	@echo "Run: infrastructure/runbooks/emergency-stop-trading.sh <URL> <TOKEN>"

# ---- Utilities ----

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

install: ## Install backend dependencies
	cd backend && pip install -e ".[dev,ml]"

# ---- Load Testing ----

load-test: ## Run load tests with Locust web UI (http://localhost:8089)
	locust -f tests/load/locustfile.py --host http://localhost:8000

load-test-headless: ## Run headless load test (50 users, 5/s ramp, 5min)
	locust -f tests/load/locustfile.py --host http://localhost:8000 \
		--headless -u 50 -r 5 --run-time 5m
