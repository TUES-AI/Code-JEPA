#!/usr/bin/env bash
set -euo pipefail

echo "[code-jepa] entrypoint start"

APP_ROOT="${APP_ROOT:-/proj}"
DATA_DIR="${CODE_JEPA_DATA_ROOT:-${APP_ROOT}/s3}"
REPO_DIR="${REPO_DIR:-${APP_ROOT}/Code-JEPA}"
HF_HOME="${HF_HOME:-${APP_ROOT}/huggingface}"
TS_STATE_DIR="${TS_STATE_DIR:-/var/lib/tailscale}"
TS_HOSTNAME="${TS_HOSTNAME:-gpu-box}"
TS_TAGS="${TS_TAGS:-tag:gpu}"
TS_ACCEPT_DNS="${TS_ACCEPT_DNS:-false}"
TS_ENABLE_SSH="${TS_ENABLE_SSH:-true}"

export CODE_JEPA_DATA_ROOT="$DATA_DIR"
mkdir -p "$APP_ROOT" "$DATA_DIR" "$HF_HOME" "$TS_STATE_DIR" /var/run/tailscale
cd "$APP_ROOT"

nohup /usr/sbin/tailscaled \
  --tun=userspace-networking \
  --state="${TS_STATE_DIR}/tailscaled.state" \
  --socket=/var/run/tailscale/tailscaled.sock \
  >/tmp/tailscaled.log 2>&1 &

for _ in {1..100}; do
  [[ -S /var/run/tailscale/tailscaled.sock ]] && break
  sleep 0.1
done

UP_ARGS=(--hostname="$TS_HOSTNAME" --accept-dns="$TS_ACCEPT_DNS" --advertise-tags="$TS_TAGS")
if [[ "$TS_ENABLE_SSH" == "true" ]]; then
  UP_ARGS+=(--ssh)
fi
if [[ -n "${TS_AUTHKEY:-}" ]]; then
  UP_ARGS+=(--authkey="$TS_AUTHKEY")
fi
if [[ -n "${TS_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<<"${TS_EXTRA_ARGS}"
  UP_ARGS+=("${EXTRA_ARGS[@]}")
fi

for _ in {1..60}; do
  if tailscale up "${UP_ARGS[@]}"; then
    break
  fi
  sleep 2
done

tailscale status || true

if [[ "${S3_SYNC_ON_STARTUP:-true}" == "true" ]]; then
  echo "[code-jepa] syncing s3://${S3_BUCKET:-code-jepa}/ -> ${DATA_DIR}/"
  if /usr/local/bin/sync-code-jepa-all; then
    echo "[code-jepa] s3 sync complete"
  else
    echo "[code-jepa] s3 sync failed; leaving pod running for inspection" >&2
  fi
fi

echo "[code-jepa] ready workspace=${APP_ROOT} repo=${REPO_DIR} s3=${DATA_DIR}"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

exec sleep infinity
