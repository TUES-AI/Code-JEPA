#!/usr/bin/env bash
set -euo pipefail

echo "[code-jepa] entrypoint start"

APP_ROOT="${APP_ROOT:-/proj}"
DATA_DIR="${CODE_JEPA_DATA_ROOT:-${APP_ROOT}/code-jepa}"
REPO_DIR="${REPO_DIR:-${APP_ROOT}/Code-JEPA}"
IMAGE_REPO_DIR="${IMAGE_REPO_DIR:-/opt/code-jepa-image}"
HF_HOME="${HF_HOME:-${APP_ROOT}/huggingface}"
TS_STATE_DIR="${TS_STATE_DIR:-/var/lib/tailscale}"
TS_HOSTNAME="${TS_HOSTNAME:-gpu-box}"
TS_TAGS="${TS_TAGS:-tag:gpu}"
TS_ACCEPT_DNS="${TS_ACCEPT_DNS:-false}"
TS_ENABLE_SSH="${TS_ENABLE_SSH:-true}"

mkdir -p "$APP_ROOT" "$DATA_DIR" "$HF_HOME" /proj/checkpoints /proj/artifacts "$TS_STATE_DIR" /var/run/tailscale

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[code-jepa] materializing repo into ${REPO_DIR}"
  rm -rf "$REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  rsync -a "$IMAGE_REPO_DIR/" "$REPO_DIR/"
fi

if ! git config --system --get-all safe.directory 2>/dev/null | grep -Fxq "$REPO_DIR"; then
  git config --system --add safe.directory "$REPO_DIR" 2>/dev/null || true
fi

if [[ -f "$REPO_DIR/pyproject.toml" ]]; then
  /opt/venv/bin/pip install -e "$REPO_DIR[dev,transforms]" >/tmp/code-jepa-editable-install.log 2>&1 || cat /tmp/code-jepa-editable-install.log
fi

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

echo "[code-jepa] ready repo=${REPO_DIR} data=${DATA_DIR}"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

exec sleep infinity
