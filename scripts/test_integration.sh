#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg  # avoid GUI issues in CI/containers

pytest \
  -m "integration" \
  tests/integration \
  --maxfail=1 -q
