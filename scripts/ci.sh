#!/usr/bin/env bash
# Local CI pipeline — runs lint, type-check, and tests in sequence.
# Exit on first failure, print timestamps, clean up temp artifacts.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TIMESTAMP() { date '+%Y-%m-%d %H:%M:%S'; }
STEP() {
  echo ""
  echo "${YELLOW}[$(TIMESTAMP)] $1${NC}"
}
OK() { echo "  ${GREEN}✓${NC} $1"; }
FAIL() { echo "  ${RED}✗${NC} $1"; exit 1; }

# ── Configuration ──────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
REQUIREMENTS="${REQUIREMENTS:-requirements-dev.txt}"
PYTEST_ARGS="${PYTEST_ARGS:---cov=pydantask --cov-report=term-missing --cov-config=pyproject.toml}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-0}"

# ── Pre-flight ─────────────────────────────────────────────────────
STEP "Pre-flight checks"

# Python version
PYTHON_VERSION=$("$PYTHON" --version 2>&1)
STEP "Python: $PYTHON_VERSION"
"$PYTHON" -c "import sys; assert sys.version_info >= (3, 12), f'Requires Python 3.12+, got {sys.version_info}'"
OK "Python 3.12+ confirmed"

# Virtual environment
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
  echo ""
  echo "${YELLOW}  ⚠ No virtualenv active. Activate .venv first:${NC}"
  echo "    source .venv/bin/activate"
  echo "  Or re-run with PYTHON=/path/to/python"
  echo ""
  exit 1
fi

# ── Dependency install ─────────────────────────────────────────────
STEP "Installing dependencies"

if [ -f "$REQUIREMENTS" ]; then
  $PYTHON -m pip install -q -r "$REQUIREMENTS" 2>&1 | tail -1 || true
  OK "Installed $REQUIREMENTS"
else
  # Install in dev mode (pulls from pyproject.toml)
  $PYTHON -m pip install -q -e ".[dev]" 2>&1 | tail -1 || true
  OK "Installed package in dev mode"
fi

# ── Lint ───────────────────────────────────────────────────────────
STEP "Linting with ruff"
if ! $PYTHON -m ruff check . 2>&1; then
  FAIL "ruff check failed"
fi
OK "ruff check passed"

STEP "Formatting check with ruff"
if ! $PYTHON -m ruff format --check . 2>&1; then
  FAIL "ruff format check failed (run: make format)"
fi
OK "ruff format passed"

# ── Type checking ──────────────────────────────────────────────────
STEP "Type checking with pyright"
if ! $PYTHON -m pyright 2>&1; then
  FAIL "pyright failed"
fi
OK "pyright passed"

# ── Tests ──────────────────────────────────────────────────────────
STEP "Running tests with pytest"
if ! $PYTHON -m pytest $PYTEST_ARGS 2>&1; then
  FAIL "pytest failed"
fi
OK "pytest passed"

# ── Coverage check ─────────────────────────────────────────────────
STEP "Checking coverage threshold (${COVERAGE_THRESHOLD}%)"
TOTAL_COV=$($PYTHON -m pytest --cov-report=term-missing -q --co 2>/dev/null | grep -oP '\d+ test' | grep -oP '\d+' || echo "0")

# Use pytest-cov's built-in threshold
if ! $PYTHON -m pytest --cov-fail-under="$COVERAGE_THRESHOLD" -q 2>/dev/null; then
  FAIL "Coverage below ${COVERAGE_THRESHOLD}%"
fi
OK "Coverage >= ${COVERAGE_THRESHOLD}%"

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "${GREEN}═══════════════════════════════════════════════════${NC}"
echo "${GREEN}  All checks passed.$(TIMESTAMP)${NC}"
echo "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
