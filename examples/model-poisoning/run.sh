#!/usr/bin/env bash
# Script to execute the Model Poisoning Weight-Space Programming run and out-of-harness evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/models/input" "${SCRIPT_DIR}/models/output" "${SCRIPT_DIR}/logs"

# Activate the local virtual environment so python3 and packages (numpy, torch, safetensors) are on PATH
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Set WSP_PYTHON so Pi WSP harness executes with this environment
export WSP_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Ensure the Pi extension directory uses our populated virtual environment
PI_EXT_DIR="${SCRIPT_DIR}/.pi/git/github.com/felipepenha/pi-weight-space-programming"
if [ -d "${PI_EXT_DIR}" ]; then
  rm -rf "${PI_EXT_DIR}/.venv"
  ln -sf "${SCRIPT_DIR}/.venv" "${PI_EXT_DIR}/.venv"
fi

EVAL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --eval-only|--skip-pi)
      EVAL_ONLY=true
      shift
      ;;
  esac
done

if [ "${EVAL_ONLY}" = false ]; then
  echo "======================================================================"
  echo "🚀 Running Pi Weight-Space Programming (π · WSP)"
  echo "   Model: google/gemini-3.8-flash | Effort: medium"
  echo "   Prompt: @PROMPT.md"
  echo "======================================================================"

  pi -a -p \
    --model google/gemini-3.8-flash \
    --wsp-effort medium \
    --wsp-log-dir "${SCRIPT_DIR}/logs" \
    "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log"
else
  echo "ℹ️  Running in evaluation-only mode (--eval-only)" | tee "${SCRIPT_DIR}/session.log"
fi

{
  echo ""
  echo "======================================================================"
  echo "🧪 Running Out-of-Harness Evaluation on models/output"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

uv run python3 eval/eval_steered_model.py --model-dir models/output 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

echo ""
echo "======================================================================"
echo "✅ Workflow complete!"
echo "   Session transcript: ${SCRIPT_DIR}/session.log"
echo "   Audit logs        : ${SCRIPT_DIR}/logs/"
echo "   Output model      : ${SCRIPT_DIR}/models/output/"
echo "======================================================================"
