"""Append-only record of what the runner sent and what the broker did, used to restart safely."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from app.fx.analytics import ClosedTrade


class Journal:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock

    def append(self, event: str, **fields: object) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, "at": self._clock(), **fields}, sort_keys=True) + "\n")

    def events(self) -> list[dict]:
        if not self._path.exists():
            return []
        return [json.loads(line) for line in self._path.read_text(encoding="utf-8").splitlines() if line]

    def has_order(self, client_order_id: str) -> bool:
        return any(e["event"] == "order_intent" and e["client_order_id"] == client_order_id for e in self.events())

    def open_position_ids(self) -> set[str]:
        """Positions the journal saw open and never saw closed."""
        opened: set[str] = set()
        for event in self.events():
            if event["event"] in ("order_filled", "adopted"):
                opened.add(event["position_id"])
            elif event["event"] in ("closed", "closed_while_down", "orphan_closed"):
                opened.discard(event["position_id"])
        return opened


def closed_trades(events: list[dict]) -> list[ClosedTrade]:
    """Trades that the journal saw open and then close with a known result, ready for the analytics.

    R is the result divided by the money the position risked at its stop when it was sized.
    """
    opened: dict[str, dict] = {}
    trades = []
    for event in events:
        if event["event"] in ("order_filled", "adopted"):
            opened[event["position_id"]] = event
        elif event["event"] == "closed" and event.get("result") is not None and event["position_id"] in opened:
            entry = opened.pop(event["position_id"])
            risk = entry.get("risk_at_stop")
            trades.append(ClosedTrade(
                symbol=entry["symbol"], side=entry["side"], entry_time=entry["at"], exit_time=event["at"],
                pnl=event["result"], r_multiple=event["result"] / risk if risk else None,
                exit_reason=event.get("reason", ""),
            ))
    return trades
