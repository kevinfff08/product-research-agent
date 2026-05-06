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
CONDA_ENV_NAME="research_tools"
ENV_PYTHON=""
if [ -n "${RESEARCH_TOOLS_PYTHON:-}" ] && [ -x "${RESEARCH_TOOLS_PYTHON}" ]; then
    ENV_PYTHON="${RESEARCH_TOOLS_PYTHON}"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/envs/${CONDA_ENV_NAME}/bin/python" ]; then
    ENV_PYTHON="${CONDA_PREFIX}/envs/${CONDA_ENV_NAME}/bin/python"
elif [ -n "${CONDA_EXE:-}" ] && [ -x "$(dirname "$(dirname "${CONDA_EXE}")")/envs/${CONDA_ENV_NAME}/bin/python" ]; then
    ENV_PYTHON="$(dirname "$(dirname "${CONDA_EXE}")")/envs/${CONDA_ENV_NAME}/bin/python"
elif [ -x "${HOME}/anaconda3/envs/${CONDA_ENV_NAME}/bin/python" ]; then
    ENV_PYTHON="${HOME}/anaconda3/envs/${CONDA_ENV_NAME}/bin/python"
elif [ -x "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]; then
    ENV_PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
fi
ARGS=("$@")
if [ "$#" -eq 0 ]; then
    ARGS=("start")
fi

if { [ "${ARGS[0]}" = "research" ] || [ "${ARGS[0]}" = "start" ]; } && [ "$LLM_MODE" = "setup-token" ]; then
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

if [ "${CONDA_DEFAULT_ENV:-}" = "${CONDA_ENV_NAME}" ]; then
    echo "[INFO] Python environment: ${CONDA_DEFAULT_ENV}"
    python -m src "${ARGS[@]}"
elif [ -n "${ENV_PYTHON}" ]; then
    echo "[INFO] Python environment: ${CONDA_ENV_NAME}"
    echo "[INFO] Python executable: ${ENV_PYTHON}"
    "${ENV_PYTHON}" -m src "${ARGS[@]}"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] conda not found. Activate ${CONDA_ENV_NAME} and run: python -m src ${ARGS[*]}"
        exit 1
    fi
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    echo "[INFO] Python environment: ${CONDA_DEFAULT_ENV}"
    python -m src "${ARGS[@]}"
fi
