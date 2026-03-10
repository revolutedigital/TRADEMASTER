"""TOTP (Time-based One-Time Password) two-factor authentication.

RFC 6238 compliant implementation using only the standard library.
Provides secret generation, provisioning URI creation, token verification,
and one-time backup code management.
"""

import base64
import hashlib
import hmac
import os
import secrets
import struct
import time
from urllib.parse import quote

from app.core.logging import get_logger

logger = get_logger(__name__)

# TOTP defaults per RFC 6238
_TOTP_PERIOD = 30  # seconds
_TOTP_DIGITS = 6
_SECRET_BYTES = 20  # 160-bit secret


class TOTPManager:
    """Manages TOTP 2FA operations: secret generation, token verification, backup codes."""

    # ------------------------------------------------------------------
    # Secret generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_secret() -> str:
        """Generate a cryptographically random base32-encoded shared secret.

        Returns:
            A base32-encoded string suitable for use with TOTP authenticator apps.
        """
        raw = os.urandom(_SECRET_BYTES)
        secret = base64.b32encode(raw).decode("ascii").rstrip("=")
        logger.debug("totp_secret_generated")
        return secret

    # ------------------------------------------------------------------
    # Provisioning URI (for QR codes)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_provisioning_uri(
        secret: str,
        username: str,
        issuer: str = "TradeMaster",
    ) -> str:
        """Build an ``otpauth://`` URI for QR-code enrolment.

        Args:
            secret: Base32-encoded shared secret.
            username: Account identifier shown in the authenticator app.
            issuer: Service name shown in the authenticator app.

        Returns:
            An ``otpauth://totp/…`` URI string.
        """
        label = f"{quote(issuer)}:{quote(username)}"
        params = (
            f"secret={secret}"
            f"&issuer={quote(issuer)}"
            f"&algorithm=SHA1"
            f"&digits={_TOTP_DIGITS}"
            f"&period={_TOTP_PERIOD}"
        )
        return f"otpauth://totp/{label}?{params}"

    # ------------------------------------------------------------------
    # TOTP computation (RFC 6238 / RFC 4226)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_secret(secret: str) -> bytes:
        """Decode a base32 secret, adding padding as needed."""
        # Re-add padding stripped during generation
        padded = secret + "=" * (-len(secret) % 8)
        return base64.b32decode(padded, casefold=True)

    @classmethod
    def _generate_hotp(cls, secret: str, counter: int) -> str:
        """Compute an HOTP value for the given counter (RFC 4226).

        Args:
            secret: Base32-encoded shared secret.
            counter: 8-byte counter value.

        Returns:
            Zero-padded OTP string of ``_TOTP_DIGITS`` digits.
        """
        key = cls._decode_secret(secret)
        msg = struct.pack(">Q", counter)
        digest = hmac.new(key, msg, hashlib.sha1).digest()

        # Dynamic truncation (RFC 4226 s5.3)
        offset = digest[-1] & 0x0F
        code_int = struct.unpack(">I", digest[offset : offset + 4])[0]
        code_int &= 0x7FFFFFFF
        code_int %= 10 ** _TOTP_DIGITS

        return str(code_int).zfill(_TOTP_DIGITS)

    @classmethod
    def _current_counter(cls) -> int:
        """Return the current TOTP time-step counter."""
        return int(time.time()) // _TOTP_PERIOD

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @classmethod
    def verify_totp(cls, secret: str, token: str, window: int = 1) -> bool:
        """Verify a TOTP token against the shared secret.

        Uses constant-time comparison to prevent timing side-channels.

        Args:
            secret: Base32-encoded shared secret.
            token: The 6-digit token string provided by the user.
            window: Number of time-steps to check before and after the
                    current step (default ``1`` = +/- 30 s).

        Returns:
            ``True`` if the token is valid within the time window.
        """
        if not token or not token.isdigit() or len(token) != _TOTP_DIGITS:
            return False

        counter = cls._current_counter()
        for offset in range(-window, window + 1):
            expected = cls._generate_hotp(secret, counter + offset)
            if hmac.compare_digest(expected, token):
                logger.info("totp_verified", offset=offset)
                return True

        logger.warning("totp_verification_failed")
        return False

    # ------------------------------------------------------------------
    # Backup codes
    # ------------------------------------------------------------------

    @staticmethod
    def generate_backup_codes(count: int = 8) -> list[str]:
        """Generate a set of single-use backup codes.

        Each code is an 8-character hex string derived from
        ``secrets.token_hex``.

        Args:
            count: Number of backup codes to generate.

        Returns:
            A list of unique backup-code strings.
        """
        codes = [secrets.token_hex(4).upper() for _ in range(count)]
        logger.info("backup_codes_generated", count=count)
        return codes

    @staticmethod
    def verify_backup_code(
        stored_codes: list[str],
        provided_code: str,
    ) -> tuple[bool, list[str]]:
        """Verify a backup code and remove it from the remaining set.

        Uses constant-time comparison for each candidate to avoid
        timing leaks.

        Args:
            stored_codes: List of unused backup codes.
            provided_code: The code supplied by the user.

        Returns:
            A tuple of ``(is_valid, remaining_codes)``.  If the code
            matched, ``remaining_codes`` will have it removed.
        """
        normalised = provided_code.strip().upper()
        matched_index: int | None = None

        # Compare against every stored code to keep timing constant
        for idx, code in enumerate(stored_codes):
            if hmac.compare_digest(code.upper(), normalised):
                matched_index = idx

        if matched_index is not None:
            remaining = [c for i, c in enumerate(stored_codes) if i != matched_index]
            logger.info("backup_code_used", remaining=len(remaining))
            return True, remaining

        logger.warning("backup_code_verification_failed")
        return False, list(stored_codes)
