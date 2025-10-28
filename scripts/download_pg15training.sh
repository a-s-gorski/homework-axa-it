#!/usr/bin/env bash

set -euo pipefail

URL="https://github.com/dutangc/CASdatasets/raw/refs/heads/master/data/pg15training.rda"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT_DIR="${SCRIPT_DIR}/../data/01_raw"
RDA_PATH="${OUT_DIR}/pg15training.rda"


echo "Downloading RDA to: ${RDA_PATH}"
if command -v curl >/dev/null 2>&1; then
  curl -fL "${URL}" -o "${RDA_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${RDA_PATH}" "${URL}"
else
  echo "Error: need curl or wget installed." >&2
  exit 1
fi