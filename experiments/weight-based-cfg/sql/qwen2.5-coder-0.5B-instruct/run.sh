#!/usr/bin/env bash
# End-to-end execution script for weight-based SQL:2016 CFG compilation on Qwen/Qwen2.5-Coder-0.5B-Instruct.
# Runs Pi Weight-Space Programming editing (`./edit.sh`) and LLMSQL-2.0 evaluation (`./eval.sh`) sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EVAL_ONLY=false
EVAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --eval-only|--skip-pi)
      EVAL_ONLY=true
      ;;
    *)
      EVAL_ARGS+=("$arg")
      ;;
  esac
done

if [ "${EVAL_ONLY}" = false ]; then
  "${SCRIPT_DIR}/edit.sh"
else
  echo "ℹ️  Running in evaluation-only mode (--eval-only)" | tee -a "${SCRIPT_DIR}/session.log"
fi

"${SCRIPT_DIR}/eval.sh" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
