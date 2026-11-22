#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REPO_URL="git:github.com/felipepenha/pi-weight-space-programming"
LOCAL_PATH="../../../.."

if [ "${1:-}" = "--local" ]; then
  pi remove -l "${REPO_URL}" -a 2>/dev/null || true
  rm -rf "${SCRIPT_DIR}/.pi/git/github.com/felipepenha/pi-weight-space-programming"
  pi install -l "${LOCAL_PATH}" -a
else
  pi remove -l "${LOCAL_PATH}" -a 2>/dev/null || true
  pi install -l "${REPO_URL}" -a
fi
