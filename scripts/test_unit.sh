#!/usr/bin/env bash
set -euo pipefail

# Where your code lives (adjust if needed)
PKG_DIR="src/mlstream"

# Run ONLY unit tests:
# - exclude integration/e2e via markers
# - measure coverage, print summary, generate XML + HTML
pytest \
  -m "not integration and not e2e" \
  tests/unit \
  --maxfail=1 \
  -q \
  --cov="${PKG_DIR}" \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml \
  --cov-report=html
