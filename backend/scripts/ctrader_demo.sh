#!/usr/bin/env bash
# Runs cTrader CLI commands on the Fusion DEMO account only.
#
#   bash backend/scripts/ctrader_demo.sh accounts
#   bash backend/scripts/ctrader_demo.sh run "symbol EURUSD" "price EURUSD" "positions"
#   bash backend/scripts/ctrader_demo.sh --dry-run run "price EURUSD"     (prints the plan, runs nothing)
#
# The account (10139135, demo) and the cTID are fixed in fx_demo_runner.py. The password comes from a
# private file and is never on a command line: `accounts` hands the file to the CLI (--pwd-file), `run`
# types it at the CLI's prompt (scripts/research/ctrader_demo.py, which also holds the command allow-list).
set -euo pipefail

CTID="igorrevolute"
IMAGE="ghcr.io/spotware/ctrader-console@sha256:285484fad431e0ffa4ca96662e82ea66cead93c97cf0e3c46006e80cab4734ba"
PASSWORD_FILE="${CTRADER_PASSWORD_FILE:-$HOME/.config/trademaster/ctid.pwd}"
SAFE_ARGUMENT='^[A-Za-z0-9 ._=-]+$'
TIMEOUT_SECONDS=120

original=("$@")
dry_run=0
if [ "${1:-}" = "--dry-run" ]; then dry_run=1; shift; fi
mode="${1:-}"; [ $# -gt 0 ] && shift

fail() { echo "ctrader_demo: $*" >&2; exit 2; }

[ -r "$PASSWORD_FILE" ] || fail "password file not found or unreadable: $PASSWORD_FILE"

# A login session that predates the `docker` group re-runs this script under it (newgrp uses the
# group the user already belongs to; nothing is escalated).
if [ $dry_run -eq 0 ] && [ -z "${CTRADER_DEMO_INNER:-}" ] && ! docker info >/dev/null 2>&1; then
  for argument in "${original[@]}"; do
    [[ "$argument" =~ $SAFE_ARGUMENT ]] || fail "unsupported characters in argument: $argument"
  done
  exec newgrp docker <<<"env CTRADER_DEMO_INNER=1 bash $(printf '%q ' "$0" "${original[@]}")"
fi

if [ "$mode" = "accounts" ]; then
  [ $# -eq 0 ] || fail "accounts takes no arguments"
  docker_args=(run --rm -v "$PASSWORD_FILE:/run/ctid.pwd:ro" "$IMAGE" accounts "--ctid=$CTID" --pwd-file=/run/ctid.pwd)
  if [ $dry_run -eq 1 ]; then echo "docker ${docker_args[*]}"; exit 0; fi
  exec timeout "$TIMEOUT_SECONDS" docker "${docker_args[@]}"
elif [ "$mode" != "run" ]; then
  fail "usage: ctrader_demo.sh [--dry-run] accounts | run \"<cli command>\" ..."
fi

[ $# -gt 0 ] || fail "run needs at least one command"
cd "$(dirname "$0")/.."
if [ $dry_run -eq 1 ]; then exec .venv/bin/python -W ignore -m scripts.research.ctrader_demo --dry-run "$@"; fi
exec .venv/bin/python -W ignore -m scripts.research.ctrader_demo "$@"
