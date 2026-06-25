#!/usr/bin/env bash
set -euo pipefail

: "${S3_BUCKET:=code-jepa}"
: "${CODE_JEPA_DATA_ROOT:=/proj/code-jepa}"

mkdir -p "$CODE_JEPA_DATA_ROOT"

if [[ -n "${S3_ENDPOINT_URL:-}" ]]; then
  export S3_ENDPOINT_URL
fi

s5cmd sync --size-only "s3://${S3_BUCKET}/" "${CODE_JEPA_DATA_ROOT}/"
