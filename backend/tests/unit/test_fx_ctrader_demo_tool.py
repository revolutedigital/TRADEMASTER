"""The demo tool's allow-list: what may be sent to the CLI on the demo account."""

import pytest

from scripts.research import ctrader_demo as tool


@pytest.mark.parametrize("command", ["positions", "price EURUSD", "deals EURUSD 3", "position close all yes",
                                     f"order place-market --symbol=EURUSD --side=buy --volume=1000 {tool.ACCOUNT}"])
def test_read_and_demo_trading_commands_pass(command: str) -> None:
    tool.check(command)


@pytest.mark.parametrize("command,reason", [
    ("account switch 9081881", "not allowed"),
    ("quit", "not allowed"),
    ("positions 9081881", "another account"),
    ("price EURUSD; ls", "unsupported characters"),
    ("price $(whoami)", "unsupported characters"),
])
def test_anything_else_is_refused(command: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        tool.check(command)


def test_the_main_entry_refuses_before_logging_in(capsys) -> None:
    assert tool.main(["account switch 9081881"]) == 2
    assert tool.main([]) == 2
    assert "not allowed" in capsys.readouterr().err


def test_dry_run_prints_the_plan_without_touching_the_password_file(capsys) -> None:
    assert tool.main(["--dry-run", "price EURUSD"]) == 0
    assert "price EURUSD" in capsys.readouterr().out
