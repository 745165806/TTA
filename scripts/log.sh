#!/usr/bin/env bash
# Reusable evidence logging helpers.  Source this file; it never runs a command by itself.

eptta_log_environment() {
  local output=$1
  if [ -e "${output}" ]; then
    echo "log exists; overwrite is forbidden: ${output}" >&2
    return 2
  fi
  {
    date --iso-8601=seconds
    uname -a
    command -v python || true
    python --version 2>&1 || true
    git rev-parse HEAD 2>&1 || true
    git status --short 2>&1 || true
  } >"${output}"
}

eptta_run_logged() {
  if [ "$#" -lt 2 ]; then
    echo "usage: eptta_run_logged LOG COMMAND [ARG ...]" >&2
    return 2
  fi
  local output=$1
  shift
  if [ -e "${output}" ]; then
    echo "log exists; overwrite is forbidden: ${output}" >&2
    return 2
  fi
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\nstarted_at: %s\n' "$(date --iso-8601=seconds)"
    "$@"
    local code=$?
    printf 'exit_code: %d\nfinished_at: %s\n' "${code}" "$(date --iso-8601=seconds)"
    return "${code}"
  } >"${output}" 2>&1
}
