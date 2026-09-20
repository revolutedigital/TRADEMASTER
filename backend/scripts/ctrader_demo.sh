#!/usr/bin/env bash
# Runs cTrader CLI commands on the Fusion DEMO account only, in a throwaway container.
#
#   bash backend/scripts/ctrader_demo.sh accounts
#   bash backend/scripts/ctrader_demo.sh run "symbol EURUSD" "price EURUSD" "positions"
#   bash backend/scripts/ctrader_demo.sh --dry-run run "price EURUSD"     (prints the plan, runs nothing)
#
# The account (10139135, demo) and the cTID are fixed here. The password is read from a private
# file inside the container (never typed on this command line) and is masked in the output. Only the commands in ALLOWED_VERBS are accepted (no `account` switching, no cBots).
set -euo pipefail

CTID="igorrevolute"
ACCOUNT="10139135"
IMAGE="ghcr.io/spotware/ctrader-console@sha256:285484fad431e0ffa4ca96662e82ea66cead93c97cf0e3c46006e80cab4734ba"
PASSWORD_FILE="${CTRADER_PASSWORD_FILE:-$HOME/.config/trademaster/ctid.pwd}"
ALLOWED_VERBS="symbol symbols price prices positions position orders order deals orders-history account-stats exposure"
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
for line in "$@"; do
  [[ "$line" =~ $SAFE_ARGUMENT ]] || fail "unsupported characters in command: $line"
  verb="${line%% *}"
  [[ " $ALLOWED_VERBS " == *" $verb "* ]] || fail "command not allowed: $verb"
  # a command must not name another account: the session is pinned to the demo one
  for number in $(grep -oE '[0-9]{6,}' <<<"$line" || true); do
    [ "$number" = "$ACCOUNT" ] || fail "commands may not name another account: $number"
  done
done

if [ $dry_run -eq 1 ]; then
  echo "one container per command, session pinned to $CTID / account $ACCOUNT, password read from $PASSWORD_FILE inside the container:"
  printf '  %s\n' "$@"
  exit 0
fi

password="$(cat "$PASSWORD_FILE")"
mask() {
  awk -v p="$password" '{ if (length(p)) while ((i = index($0, p)) > 0) $0 = substr($0, 1, i - 1) "***" substr($0, i + length(p)); print }'
}
# The CLI refuses piped credentials, so each command is its own login (`-q` exits after it). The
# password is read from the mounted file inside the container, so it is not in this command line.
for line in "$@"; do
  read -r -a words <<<"$line"
  echo "> $line"
  timeout "$TIMEOUT_SECONDS" docker run --rm -v "$PASSWORD_FILE:/run/ctid.pwd:ro" -e "CTID=$CTID" -e "ACCOUNT=$ACCOUNT" \
    --entrypoint sh "$IMAGE" \
    -c 'exec /usr/local/bin/ctrader-cli-entrypoint "$@" --ctid="$CTID" --password="$(cat /run/ctid.pwd)" --account="$ACCOUNT" -q' \
    sh "${words[@]}" 2>&1 | mask
done
