#!/usr/bin/env bash
# Pi Weight-Space Programming editing stage for {{TASK_TITLE}}
# with OS-level eval isolation (chmod 000 eval/ & eval.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/models/input" "${SCRIPT_DIR}/models/output" "${SCRIPT_DIR}/logs"

# Always restore eval/ and eval.sh permissions on exit or interrupt
restore_eval_perms() {
  if [ -d "${SCRIPT_DIR}/eval" ]; then
    chmod 755 "${SCRIPT_DIR}/eval" 2>/dev/null || true
  fi
  if [ -f "${SCRIPT_DIR}/eval.sh" ]; then
    chmod 755 "${SCRIPT_DIR}/eval.sh" 2>/dev/null || true
  fi
}
trap restore_eval_perms EXIT

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

# Set WSP_PYTHON so Pi WSP harness executes with this environment
export WSP_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Ensure the Pi extension directory uses our populated virtual environment
PI_EXT_DIR="${SCRIPT_DIR}/.pi/git/github.com/felipepenha/pi-weight-space-programming"
if [ -d "${PI_EXT_DIR}" ]; then
  rm -rf "${PI_EXT_DIR}/.venv"
  ln -sf "${SCRIPT_DIR}/.venv" "${PI_EXT_DIR}/.venv"
fi

echo "======================================================================"
echo "🚀 Running Pi Weight-Space Programming (π · WSP)"
echo "   Model         : google/gemini-3.8-flash | Effort: medium"
echo "   Prompt        : @PROMPT.md"
echo "   Eval Isolation: Locking eval/ & eval.sh permissions (chmod 000) during Pi run"
echo "======================================================================"

# Physically lock eval/ and eval.sh so neither the Pi agent nor the Python sandbox can read test cases
if [ -d "${SCRIPT_DIR}/eval" ]; then
  chmod 000 "${SCRIPT_DIR}/eval"
fi
if [ -f "${SCRIPT_DIR}/eval.sh" ]; then
  chmod 000 "${SCRIPT_DIR}/eval.sh"
fi

if command -v pi >/dev/null 2>&1; then
  pi -a -p \
    --no-builtin-tools \
    --model google/gemini-3.8-flash \
    --wsp-effort medium \
    --wsp-log-dir "${SCRIPT_DIR}/logs" \
    "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log" || {
      echo "⚠️  Pi execution encountered an issue. See ${SCRIPT_DIR}/session.log" | tee -a "${SCRIPT_DIR}/session.log"
    }
else
  echo "⚠️  'pi' CLI not found on PATH. Skipping agent execution." | tee "${SCRIPT_DIR}/session.log"
fi

restore_eval_perms
