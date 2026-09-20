"""Hard limits of the live runner: daily loss cap, position cap, spread cap and a latched kill switch.

The kill switch is written to disk, so a restart does not forget it, and only an explicit `reset`
by an operator clears it. The day (17:00 to 17:00 New York) and its starting equity are persisted
as well, so restarting the bot mid-day does not hand it a fresh loss allowance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.fx.sessions import fx_day


@dataclass(frozen=True)
class RiskLimits:
    risk_fraction: float = 0.0025
    max_daily_loss_fraction: float = 0.01
    max_open_positions: int = 1
    max_spread_pips: float = 1.5
    heartbeat_seconds: float = 120.0


@dataclass(frozen=True)
class EntryDecision:
    allowed: bool
    reason: str


class RiskGuard:
    def __init__(self, limits: RiskLimits, state_path: Path) -> None:
        self.limits = limits
        self._path = state_path
        self._state = {"killed": False, "reason": "", "day": None, "day_start_equity": None, "heartbeat": None}
        if state_path.exists():
            self._state.update(json.loads(state_path.read_text(encoding="utf-8")))

    def _save(self) -> None:
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._state), encoding="utf-8")
        os.replace(temporary, self._path)

    @property
    def killed(self) -> bool:
        return bool(self._state["killed"])

    @property
    def kill_reason(self) -> str:
        return str(self._state["reason"])

    def kill(self, reason: str) -> None:
        if not self.killed:
            self._state.update(killed=True, reason=reason)
            self._save()

    def reset(self) -> None:
        """Operator action: clear the kill switch."""
        self._state.update(killed=False, reason="")
        self._save()

    def loss_cap(self) -> float:
        start = self._state["day_start_equity"]
        return 0.0 if start is None else start * self.limits.max_daily_loss_fraction

    def day_loss(self, equity: float) -> float:
        start = self._state["day_start_equity"]
        return 0.0 if start is None else max(0.0, start - equity)

    def observe(self, now: float, equity: float) -> None:
        """Feed the guard the clock and the equity; starts a new day and trips the loss cap."""
        today = int(fx_day(now))
        if self._state["day"] != today:
            self._state.update(day=today, day_start_equity=equity)
            self._save()
        if self.day_loss(equity) >= self.loss_cap() > 0:
            self.kill("daily loss cap reached")

    def remaining_loss_budget(self, equity: float) -> float:
        return max(0.0, self.loss_cap() - self.day_loss(equity))

    def check_entry(self, *, open_positions: int, spread_pips: float) -> EntryDecision:
        if self.killed:
            return EntryDecision(False, f"kill switch: {self.kill_reason}")
        if open_positions >= self.limits.max_open_positions:
            return EntryDecision(False, "position cap reached")
        if spread_pips > self.limits.max_spread_pips:
            return EntryDecision(False, f"spread {spread_pips:.2f} pips above the cap")
        return EntryDecision(True, "ok")

    def heartbeat(self, now: float) -> None:
        self._state["heartbeat"] = now
        self._save()

    def is_stale(self, now: float) -> bool:
        beat = self._state["heartbeat"]
        return beat is not None and now - beat > self.limits.heartbeat_seconds
