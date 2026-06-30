#!/usr/bin/env bash
set -euo pipefail

: "${S3_BUCKET:=code-jepa}"
: "${CODE_JEPA_DATA_ROOT:=/proj/s3}"

mkdir -p "$CODE_JEPA_DATA_ROOT"

endpoint_args=()
if [[ -n "${S3_ENDPOINT_URL:-}" ]]; then
  endpoint_args=(--endpoint-url "$S3_ENDPOINT_URL")
fi

s5cmd "${endpoint_args[@]}" sync --size-only "s3://${S3_BUCKET}/*" "${CODE_JEPA_DATA_ROOT}/"
