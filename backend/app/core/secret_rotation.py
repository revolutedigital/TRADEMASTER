"""Secret rotation manager for JWT keys and other credentials.

Supports dual-key validation during rotation periods to prevent
invalidating active sessions during key transitions.
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timezone, timedelta

from app.core.logging import get_logger

logger = get_logger(__name__)


class SecretRotationManager:
    """Manages automatic rotation of secrets with dual-key validation."""

    def __init__(self, rotation_interval_days: int = 30):
        self._rotation_interval = timedelta(days=rotation_interval_days)
        self._current_key: str | None = None
        self._previous_key: str | None = None
        self._current_key_created_at: datetime | None = None
        self._rotation_count: int = 0

    @property
    def current_key(self) -> str:
        if self._current_key is None:
            self._current_key = secrets.token_hex(32)
            self._current_key_created_at = datetime.now(timezone.utc)
            logger.info("secret_initial_key_generated")
        return self._current_key

    @property
    def needs_rotation(self) -> bool:
        if self._current_key_created_at is None:
            return False
        return datetime.now(timezone.utc) - self._current_key_created_at > self._rotation_interval

    def rotate(self) -> str:
        """Rotate the current key. Previous key remains valid for grace period."""
        self._previous_key = self._current_key
        self._current_key = secrets.token_hex(32)
        self._current_key_created_at = datetime.now(timezone.utc)
        self._rotation_count += 1
        logger.info("secret_rotated", rotation_count=self._rotation_count)
        return self._current_key

    def validate_signature(self, data: str, signature: str) -> bool:
        """Validate a signature against current AND previous keys (dual-key validation)."""
        # Try current key first
        expected = hmac.new(self.current_key.encode(), data.encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True

        # Try previous key (grace period)
        if self._previous_key:
            expected_prev = hmac.new(self._previous_key.encode(), data.encode(), hashlib.sha256).hexdigest()
            if hmac.compare_digest(expected_prev, signature):
                logger.info("validated_with_previous_key")
                return True

        return False

    def get_status(self) -> dict:
        return {
            "rotation_count": self._rotation_count,
            "key_age_hours": (
                (datetime.now(timezone.utc) - self._current_key_created_at).total_seconds() / 3600
                if self._current_key_created_at
                else 0
            ),
            "needs_rotation": self.needs_rotation,
            "has_previous_key": self._previous_key is not None,
            "rotation_interval_days": self._rotation_interval.days,
        }


# Module-level singleton
secret_rotation = SecretRotationManager()
