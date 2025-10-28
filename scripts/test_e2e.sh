#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg  # avoid GUI issues in CI/containers

pytest \
  -m "e2e" \
  tests/e2e \
  --maxfail=1 -q
