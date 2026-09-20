"""Append-only record of what the runner sent and what the broker did, used to restart safely."""

from __future__ import annotations

import json
from pathlib import Path


class Journal:
    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, event: str, **fields: object) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")

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
