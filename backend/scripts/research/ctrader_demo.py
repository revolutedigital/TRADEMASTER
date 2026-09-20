"""Runs allow-listed cTrader CLI commands on the Fusion DEMO account over one login.

    python -m scripts.research.ctrader_demo "positions" "deals EURUSD 3"
    python -m scripts.research.ctrader_demo --dry-run "price EURUSD"     (prints the plan, logs in to nothing)

The account is the demo runner's, fixed there. The password reaches the CLI through its own prompt, never
as an argument (an argument shows in every process listing), and is masked in the output.
Use `backend/scripts/ctrader_demo.sh run ...`, which also gets docker access for this shell.
"""

from __future__ import annotations

import re
import sys

from app.fx.runner.ctrader_cli import PtySession
from scripts.research.fx_demo_runner import ACCOUNT, CTID, PASSWORD_FILE, cli_argv, password, remove_container

TOOL = "tool"  # container name suffix: never collides with a runner's
ALLOWED_VERBS = frozenset(
    "symbol symbols price prices positions position orders order deals orders-history account-stats exposure".split()
)
SAFE_COMMAND = re.compile(r"^[A-Za-z0-9 ._=-]+$")
ACCOUNT_LIKE = re.compile(r"[0-9]{6,}")
COMMAND_TIMEOUT_SECONDS = 60.0


def check(command: str) -> None:
    """Raises ValueError unless the command is allowed: known verb, plain characters, no other account."""
    if not SAFE_COMMAND.match(command):
        raise ValueError(f"unsupported characters in command: {command}")
    verb = command.split(" ", 1)[0]
    if verb not in ALLOWED_VERBS:
        raise ValueError(f"command not allowed: {verb}")
    for number in ACCOUNT_LIKE.findall(command):
        if number != str(ACCOUNT):
            raise ValueError(f"commands may not name another account: {number}")


def main(arguments: list[str]) -> int:
    dry_run = arguments[:1] == ["--dry-run"]
    commands = arguments[1:] if dry_run else arguments
    if not commands:
        sys.stderr.write("usage: ctrader_demo [--dry-run] \"<cli command>\" ...\n")
        return 2
    try:
        for command in commands:
            check(command)
    except ValueError as error:
        sys.stderr.write(f"ctrader_demo: {error}\n")
        return 2
    if dry_run:
        sys.stdout.write(f"one login as {CTID}, pinned to account {ACCOUNT}, password from {PASSWORD_FILE} typed at the prompt:\n")
        sys.stdout.writelines(f"  {command}\n" for command in commands)
        return 0
    secret = password()
    remove_container(TOOL)
    session = PtySession(cli_argv(TOOL), password=secret)
    try:
        session.start()
        for command in commands:
            sys.stdout.write(f"> {command}\n{session.send(command, timeout=COMMAND_TIMEOUT_SECONDS).replace(secret, '***').rstrip()}\n")
    finally:
        session.close()
        remove_container(TOOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
