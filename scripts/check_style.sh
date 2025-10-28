#!/usr/bin/env bash
# Fail on error, undefined var, or failed pipeline (we still aggregate statuses)
set -euo pipefail

# Directories to check (override: TARGETS="src/mlstream tests" ./scripts/check_style.sh)
TARGETS=${TARGETS:-"src/mlstream"}

# Optional extra args
RUFF_ARGS=${RUFF_ARGS:-""}
MYPY_ARGS=${MYPY_ARGS:-""}
BANDIT_ARGS=${BANDIT_ARGS:-"-r"}   # defaults to recursive scan

# Feature flags (0 = run, 1 = skip)
SKIP_RUFF=${SKIP_RUFF:-0}
SKIP_MYPY=${SKIP_MYPY:-0}
SKIP_BANDIT=${SKIP_BANDIT:-0}

# ---- Safe flag parsing: only loop if args were provided ----
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    case "$arg" in
      --no-ruff)   SKIP_RUFF=1 ;;
      --no-mypy)   SKIP_MYPY=1 ;;
      --no-bandit) SKIP_BANDIT=1 ;;
      *)
        echo "Unknown arg: $arg" >&2
        echo "Supported: --no-ruff --no-mypy --no-bandit" >&2
        exit 2
        ;;
    esac
  done
fi

# Pretty printing
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m   $*"; }
err()   { echo -e "\033[1;31m[ERR]\033[0m  $*"; }

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Missing dependency: $1"
    echo "Install via:"
    case "$1" in
      black) echo "  pip install black" ;;
      isort) echo "  pip install isort" ;;
      ruff)  echo "  pip install ruff" ;;
      mypy)  echo "  pip install mypy" ;;
      bandit) echo "  pip install bandit" ;;
    esac
    exit 127
  fi
}

info "Checking required tools…"
need_cmd black
need_cmd isort
[ "$SKIP_RUFF"   -eq 1 ] || need_cmd ruff
[ "$SKIP_MYPY"   -eq 1 ] || need_cmd mypy
[ "$SKIP_BANDIT" -eq 1 ] || need_cmd bandit
ok "Tool check complete."

status=0

info "Running black --check…"
if ! black --check --diff "$TARGETS"; then
  err "black formatting violations found."
  status=1
else
  ok "black passed."
fi

info "Running isort --check-only…"
if ! isort --check-only --diff --profile black "$TARGETS"; then
  err "isort import order violations found."
  status=1
else
  ok "isort passed."
fi

if [ "$SKIP_RUFF" -eq 0 ]; then
  info "Running ruff check…"
  if ! ruff check $RUFF_ARGS "$TARGETS"; then
    err "ruff lint issues found."
    status=1
  else
    ok "ruff passed."
  fi
else
  info "Skipping ruff (--no-ruff or SKIP_RUFF=1)."
fi

if [ "$SKIP_MYPY" -eq 0 ]; then
  info "Running mypy (type checking)…"
  if ! mypy $MYPY_ARGS "$TARGETS"; then
    err "mypy type-checking issues found."
    status=1
  else
    ok "mypy passed."
  fi
else
  info "Skipping mypy (--no-mypy or SKIP_MYPY=1)."
fi

if [ "$SKIP_BANDIT" -eq 0 ]; then
  info "Running Bandit (security)…"
  if ! bandit $BANDIT_ARGS "$TARGETS"; then
    err "Bandit security issues found."
    status=1
  else
    ok "Bandit passed."
  fi
else
  info "Skipping Bandit (--no-bandit or SKIP_BANDIT=1)."
fi

if [ "$status" -ne 0 ]; then
  err "Checks failed."
  echo "Tip: run ./scripts/format.sh to auto-fix black/isort/ruff issues; then address mypy/Bandit."
  exit "$status"
fi

ok "All checks passed ✅"
