#!/usr/bin/env bash
# Out-of-harness LLMSQL-2.0 evaluation stage for weight-based SQL:2016 CFG compilation
# on Qwen/Qwen2.5-Coder-0.5B-Instruct (evaluates models/input baseline, compiled models/output, and reference model).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SKIP_REFERENCE=false
LLMSQL_LIMIT="${LLMSQL_LIMIT:-50}"
REFERENCE_MODEL="${REFERENCE_MODEL:-defog/sqlcoder-7b-2}"
PROMPT_STYLE="${PROMPT_STYLE:-raw-sql}"

print_usage() {
  cat <<EOF
Usage: ./eval.sh [options]

Options:
  --quick                Evaluate a tiny stratified subset (--limit 10)
  --limit N              Evaluate N stratified LLMSQL examples (0 = full dataset, default: ${LLMSQL_LIMIT})
  --prompt-style STYLE   Prompt format: raw-sql | llmsql | sqlcoder (default: ${PROMPT_STYLE})
  --skip-reference       Skip Stage 3 evaluation of ${REFERENCE_MODEL}
  -h, --help             Show this help message and exit

Environment overrides:
  LLMSQL_LIMIT=${LLMSQL_LIMIT}
  PROMPT_STYLE=${PROMPT_STYLE}
  REFERENCE_MODEL=${REFERENCE_MODEL}
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)
      LLMSQL_LIMIT=10
      shift
      ;;
    --limit)
      if [ $# -lt 2 ]; then
        echo "Error: --limit requires an integer argument." >&2
        exit 2
      fi
      LLMSQL_LIMIT="$2"
      shift 2
      ;;
    --limit=*)
      LLMSQL_LIMIT="${1#--limit=}"
      if [ -z "${LLMSQL_LIMIT}" ]; then
        echo "Error: --limit= requires an integer argument." >&2
        exit 2
      fi
      shift
      ;;
    --prompt-style)
      if [ $# -lt 2 ]; then
        echo "Error: --prompt-style requires an argument (raw-sql|llmsql|sqlcoder)." >&2
        exit 2
      fi
      PROMPT_STYLE="$2"
      shift 2
      ;;
    --prompt-style=*)
      PROMPT_STYLE="${1#--prompt-style=}"
      if [ -z "${PROMPT_STYLE}" ]; then
        echo "Error: --prompt-style= requires an argument." >&2
        exit 2
      fi
      shift
      ;;
    --skip-reference)
      SKIP_REFERENCE=true
      shift
      ;;
    --eval-only|--skip-pi)
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'. Run './eval.sh --help' for usage." >&2
      exit 2
      ;;
  esac
done

if [ -d "${SCRIPT_DIR}/eval" ]; then
  chmod 755 "${SCRIPT_DIR}/eval" 2>/dev/null || true
fi

# Ensure virtual environment is initialized if uv is installed and .venv is missing
if [ ! -d "${SCRIPT_DIR}/.venv" ] && command -v uv >/dev/null 2>&1; then
  echo "📦 Initializing virtual environment with uv sync..."
  uv sync
fi

# Activate the local virtual environment
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

echo "ℹ️  LLMSQL Eval Config: limit=${LLMSQL_LIMIT} | prompt_style=${PROMPT_STYLE} | skip_reference=${SKIP_REFERENCE}" | tee -a "${SCRIPT_DIR}/session.log"

{
  echo ""
  echo "======================================================================"
  echo "🧪 1. Evaluating Clean Baseline Model (models/input: Unwrapped + Outlines)"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

if [ -n "$(find "${SCRIPT_DIR}/models/input" -name "*.safetensors" -o -name "*.bin" 2>/dev/null)" ]; then
  uv run --with "llmsql>=0.1.16" eval/eval_model.py \
    --model-dir models/input \
    --baseline-dir models/input \
    --mode both \
    --prompt-style "${PROMPT_STYLE}" \
    --limit "${LLMSQL_LIMIT}" \
    --output-dir eval/runs/baseline 2>&1 | tee -a "${SCRIPT_DIR}/session.log"
else
  echo "⚠️  Base model weights not found in models/input; skipping baseline eval." | tee -a "${SCRIPT_DIR}/session.log"
fi

{
  echo ""
  echo "======================================================================"
  echo "🧪 2. Evaluating Weight-Based SQL:2016 CFG Model (models/output: Unwrapped + Outlines)"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

uv run --with "llmsql>=0.1.16" eval/eval_model.py \
  --model-dir models/output \
  --baseline-dir models/input \
  --mode both \
  --prompt-style "${PROMPT_STYLE}" \
  --limit "${LLMSQL_LIMIT}" \
  --output-dir eval/runs/output-unwrapped \
  --allow-missing 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

{
  echo ""
  echo "======================================================================"
  echo "🧪 3. Evaluating Fine-Tuned SQL Reference Model (${REFERENCE_MODEL}: Unwrapped + Outlines)"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

if [ "${SKIP_REFERENCE}" = false ]; then
  uv run --with "llmsql>=0.1.16" eval/eval_model.py \
    --model-dir "${REFERENCE_MODEL}" \
    --baseline-dir models/input \
    --mode both \
    --prompt-style "${PROMPT_STYLE}" \
    --limit "${LLMSQL_LIMIT}" \
    --output-dir eval/runs/reference-sqlcoder-7b-2 2>&1 | tee -a "${SCRIPT_DIR}/session.log"
else
  echo "ℹ️  Skipping reference model evaluation (--skip-reference)." | tee -a "${SCRIPT_DIR}/session.log"
fi

uv run python3 eval/summarize_runs.py --runs-dir eval/runs 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

echo "" | tee -a "${SCRIPT_DIR}/session.log"
echo "======================================================================"
echo "✅ Workflow complete!"
echo "   Session transcript: ${SCRIPT_DIR}/session.log"
echo "   Audit logs        : ${SCRIPT_DIR}/logs/"
echo "   Output model      : ${SCRIPT_DIR}/models/output/"
echo "   Comparison table  : ${SCRIPT_DIR}/eval/runs/comparison_summary.md"
echo "======================================================================"
