#!/usr/bin/env bash
# Everything that must be true before a commit. There is no CI here, so this
# script is the gate: run it, or install it as a pre-commit hook with
#
#   ln -s ../../scripts/check.sh .git/hooks/pre-commit
#
# Formatting is checked rather than applied. A hook that rewrites files under
# you changes what you are about to commit without saying so.
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0
step() {
  printf '\n\033[1m== %s ==\033[0m\n' "$1"
  shift
  "$@" || fail=1
}

step "ruff format --check" uv run ruff format --check .
step "ruff check" uv run ruff check .
step "mypy" uv run mypy podcast_agent
step "pytest" uv run pytest -q

if [ "$fail" -ne 0 ]; then
  printf '\n\033[31mFAILED\033[0m — `uv run ruff format .` fixes formatting.\n'
  exit 1
fi
printf '\n\033[32mAll checks passed.\033[0m\n'
