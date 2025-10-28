#!/usr/bin/env bash
# Auto-fix code style issues:
# - ruff (autofix lint)
# - isort (import order, profile=black)
# - black (format)
#
# Usage:
#   ./scripts/format.sh
#   TARGETS="src/mlstream tests" ./scripts/format.sh
#   RUFF_ARGS="--unsafe-fixes" ./scripts/format.sh
#
# Notes:
# - This doesn't run mypy/Bandit/Sonar; it's for quick local fixes.
# - Keep running ./scripts/check_style.sh to validate in CI.

set -euo pipefail

TARGETS=${TARGETS:-"src/mlstream"}
RUFF_ARGS=${RUFF_ARGS:-""}

info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m   $*"; }
err()  { echo -e "\033[1;31m[ERR]\033[0m  $*"; }

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Missing dependency: $1"
    echo "Install via:"
    case "$1" in
      black) echo "  pip install black" ;;
      isort) echo "  pip install isort" ;;
      ruff)  echo "  pip install ruff" ;;
    esac
    exit 127
  fi
}

info "Checking required tools…"
need_cmd ruff
need_cmd isort
need_cmd black
ok "Tool check complete."

# 1) Ruff autofix (quick wins: unused imports/vars, etc.)
info "Running ruff --fix…"
ruff check --fix $RUFF_ARGS "$TARGETS"
ok "ruff autofix complete."

# 2) isort to normalize imports (profile=black keeps it aligned with black)
info "Running isort…"
isort --profile black "$TARGETS"
ok "isort complete."

# 3) black for final formatting pass
info "Running black…"
black "$TARGETS"
ok "black complete."

ok "Formatting completed ✅"
echo "Tip: now run ./scripts/check_style.sh to verify everything passes."
