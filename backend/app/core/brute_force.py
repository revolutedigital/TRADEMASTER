"""Brute-force protection with progressive delays and account lockout."""

from datetime import datetime, timezone
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LoginAttempt:
    count: int = 0
    last_attempt: datetime | None = None
    locked_until: datetime | None = None


class BruteForceProtection:
    """Progressive delay brute-force protection.

    - After 5 failed attempts: 1s delay
    - After 8 failed attempts: 5s delay
    - After 10 failed attempts: 15min lockout
    """

    def __init__(self):
        self._attempts: dict[str, LoginAttempt] = {}

    def check_and_record(self, identifier: str, success: bool = False) -> tuple[bool, int]:
        """Check if login is allowed and record the attempt.

        Returns:
            tuple of (is_allowed, wait_seconds)
        """
        now = datetime.now(timezone.utc)

        if identifier not in self._attempts:
            self._attempts[identifier] = LoginAttempt()

        attempt = self._attempts[identifier]

        # Check if currently locked out
        if attempt.locked_until and now < attempt.locked_until:
            remaining = int((attempt.locked_until - now).total_seconds())
            return False, remaining

        # Reset lock if expired
        if attempt.locked_until and now >= attempt.locked_until:
            attempt.locked_until = None
            attempt.count = 0

        if success:
            # Reset on successful login
            attempt.count = 0
            attempt.locked_until = None
            return True, 0

        # Record failed attempt
        attempt.count += 1
        attempt.last_attempt = now

        # Determine action based on failure count
        if attempt.count >= 10:
            from datetime import timedelta
            attempt.locked_until = now + timedelta(minutes=15)
            logger.warning("account_locked", identifier=identifier, duration_min=15)
            return False, 900
        elif attempt.count >= 8:
            return True, 5
        elif attempt.count >= 5:
            return True, 1

        return True, 0

    def is_locked(self, identifier: str) -> bool:
        attempt = self._attempts.get(identifier)
        if not attempt or not attempt.locked_until:
            return False
        return datetime.now(timezone.utc) < attempt.locked_until

    def reset(self, identifier: str) -> None:
        self._attempts.pop(identifier, None)


brute_force_protection = BruteForceProtection()
