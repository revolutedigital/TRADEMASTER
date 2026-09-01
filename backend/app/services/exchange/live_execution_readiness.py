"""In-memory freshness state for Binance Spot live-protection reconciliation."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass
class LiveProtectionReadiness:
    """Only a recent, issue-free exchange reconciliation permits live execution."""

    state: str = "UNVERIFIED"
    checked_at: datetime | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)

    def reset(self) -> None:
        self.state = "UNVERIFIED"
        self.checked_at = None
        self.issues = ()

    def mark_ready(self, checked_at: datetime | None = None) -> None:
        self.state = "READY"
        self.checked_at = checked_at or datetime.now(UTC)
        self.issues = ()

    def mark_unresolved(self, issues: list[str], checked_at: datetime | None = None) -> None:
        self.state = "UNRESOLVED"
        self.checked_at = checked_at or datetime.now(UTC)
        self.issues = tuple(issues)

    def mark_error(self, issue: str, checked_at: datetime | None = None) -> None:
        self.state = "ERROR"
        self.checked_at = checked_at or datetime.now(UTC)
        self.issues = (issue,)

    def is_ready(self, max_age_seconds: int) -> bool:
        if self.state != "READY" or self.checked_at is None:
            return False
        return datetime.now(UTC) - self.checked_at <= timedelta(
            seconds=max_age_seconds
        )

    def status(self, max_age_seconds: int) -> dict[str, object]:
        state = self.state
        if state == "READY" and not self.is_ready(max_age_seconds):
            state = "STALE"
        return {
            "ready": state == "READY",
            "state": state,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "max_age_seconds": max_age_seconds,
            "issues": list(self.issues),
        }


live_protection_readiness = LiveProtectionReadiness()
testnet_protection_readiness = LiveProtectionReadiness()
