#!/usr/bin/env bash
# Out-of-harness evaluation stage for {{TASK_TITLE}}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -d "${SCRIPT_DIR}/eval" ]; then
  chmod 755 "${SCRIPT_DIR}/eval" 2>/dev/null || true
fi

# Ensure virtual environment is initialized if uv is installed and .venv is missing
if [ ! -d "${SCRIPT_DIR}/.venv" ] && command -v uv >/dev/null 2>&1; then
  echo "📦 Initializing virtual environment with uv sync..."
  uv sync
fi

# Activate the local virtual environment so python3 and packages are on PATH
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

{
  echo ""
  echo "======================================================================"
  echo "🧪 Running Out-of-Harness Evaluation on models/output"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

# Note: Keep benchmark-only packages (e.g. `llmsql`) out of pyproject.toml/.venv and inject them here via `uv run --with "<pkg>"`.
uv run python3 eval/eval_model.py --model-dir models/output "$@" 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

echo "" | tee -a "${SCRIPT_DIR}/session.log"
echo "======================================================================"
echo "✅ Workflow complete!"
echo "   Session transcript: ${SCRIPT_DIR}/session.log"
echo "   Audit logs        : ${SCRIPT_DIR}/logs/"
echo "   Output model      : ${SCRIPT_DIR}/models/output/"
echo "======================================================================"
