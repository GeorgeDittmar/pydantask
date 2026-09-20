# Pydantask — local development and CI targets
#
# Usage:
#   make ci        # Full pipeline: lint → format-check → type-check → test → coverage
#   make test      # Run tests only
#   make lint      # Run linter only
#   make format    # Auto-fix formatting
#   make typecheck # Run type checker
#   make clean     # Remove build/test artifacts
#   make help      # Show all targets

.PHONY: ci test lint format typecheck clean help install

PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
VENV := .venv

# ── Help ───────────────────────────────────────────────────────────
help: ## Show this help
	@echo "Pydantask — available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── Install ────────────────────────────────────────────────────────
install: ## Install package in dev mode
	$(PYTHON) -m pip install -e ".[dev]"

# ── Test ───────────────────────────────────────────────────────────
test: ## Run tests (with coverage, 2% threshold)
	$(PYTHON) -m pytest --cov=pydantask \
		--cov-report=term-missing \
		--cov-fail-under=2 \
		-v test/

# ── Lint ───────────────────────────────────────────────────────────
lint: ## Run linter (ruff check)
	$(PYTHON) -m ruff check .

lint-fix: ## Run linter and auto-fix
	$(PYTHON) -m ruff check --fix .

# ── Format ─────────────────────────────────────────────────────────
format: ## Auto-fix formatting (ruff format)
	$(PYTHON) -m ruff format .

format-check: ## Check formatting without fixing
	$(PYTHON) -m ruff format --check .

# ── Type checking ──────────────────────────────────────────────────
typecheck: ## Run type checker (pyright)
	$(PYTHON) -m pyright

# ── Full CI ────────────────────────────────────────────────────────
ci: ## Full CI pipeline: lint → type-check → test → coverage
	bash scripts/ci.sh

# ── Clean ──────────────────────────────────────────────────────────
clean: ## Remove build/test artifacts
	rm -rf \
		__pycache__ \
		**/__pycache__ \
		.pytest_cache \
		.ruff_cache \
		.htmlcov \
		coverage.xml \
		.coverage \
		*.egg-info \
		dist \
		build \
		.mypy_cache \
		.pyright_output \
		src/**/*.egg-info \
		$(VENV)
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.pyo' -delete 2>/dev/null || true
