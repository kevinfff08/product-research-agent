#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Product Research Agent - Startup"
echo "============================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# --- Load .env if present ---
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo -e "${GREEN}[OK]${NC} Loaded .env"
fi

# --- Start CLIProxyAPI if setup-token mode ---
if [ "${LLM_MODE:-api-key}" = "setup-token" ]; then
    echo ""
    echo "[1/2] Starting CLIProxyAPI proxy on localhost:8317 ..."
    if ! command -v cliproxyapi &>/dev/null; then
        echo -e "${RED}[ERROR]${NC} cliproxyapi not found. Install it first:"
        echo "        brew install router-for-me/tap/cliproxyapi  (macOS)"
        echo "        curl -fsSL https://raw.githubusercontent.com/router-for-me/CLIProxyAPI/main/install.sh | bash  (Linux)"
        exit 1
    fi
    cliproxyapi &
    PIDS+=($!)
    sleep 2
    echo -e "${GREEN}[OK]${NC} CLIProxyAPI proxy started (PID: ${PIDS[-1]})"
fi

# --- Run the research agent ---
echo ""
echo "[2/2] Starting Product Research Agent ..."
echo ""

conda run -n research_tools python -m src "$@"
