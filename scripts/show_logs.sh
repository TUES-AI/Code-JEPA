#!/bin/bash
# Usage: bash scripts/show_logs.sh [job_prefix]
# Shows the last 60 lines of the most recent log matching the prefix (default: all)

LOG_DIR=/valhalla/projects/bg-eng-01/Code-JEPA/logs
PREFIX=${1:-""}

latest=$(ls -t "${LOG_DIR}"/${PREFIX}*.err 2>/dev/null | head -1)
if [[ -z "${latest}" ]]; then
    echo "No .err logs found in ${LOG_DIR} matching '${PREFIX}'"
    exit 1
fi

echo "=== STDERR: ${latest} ==="
tail -80 "${latest}"

out="${latest%.err}.out"
if [[ -f "${out}" ]]; then
    echo ""
    echo "=== STDOUT (last 40 lines): ${out} ==="
    tail -40 "${out}"
fi
