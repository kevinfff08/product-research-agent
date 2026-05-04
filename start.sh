#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "============================================"
echo "  Product Research Agent"
echo "============================================"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "[OK] Loaded .env"
fi

LLM_MODE="${LLM_MODE:-setup-token}"
LLM_PROXY_URL="${LLM_PROXY_URL:-http://localhost:8317}"

if [ "${1:-}" = "research" ] && [ "$LLM_MODE" = "setup-token" ]; then
    proxy_models_url="${LLM_PROXY_URL%/}/v1/models"
    if ! curl -fsS --max-time 2 "$proxy_models_url" >/dev/null 2>&1; then
        echo "[INFO] Starting CLIProxyAPI at $LLM_PROXY_URL"
        if [ -n "${CLIPROXYAPI_CMD:-}" ]; then
            $CLIPROXYAPI_CMD &
        elif command -v cliproxyapi >/dev/null 2>&1; then
            cliproxyapi &
        else
            echo "[ERROR] cliproxyapi not found. Install it or set CLIPROXYAPI_CMD in .env."
            exit 1
        fi
        PIDS+=("$!")
        sleep 2
    else
        echo "[OK] CLIProxyAPI is already reachable"
    fi
fi

echo ""
echo "[INFO] Running Product Research Agent"
echo ""

if [ "${CONDA_DEFAULT_ENV:-}" = "research_tools" ]; then
    python -m src "$@"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] conda not found. Activate research_tools and run: python -m src $*"
        exit 1
    fi
    conda run -n research_tools python -m src "$@"
fi
