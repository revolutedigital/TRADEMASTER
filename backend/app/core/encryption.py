"""Field-level encryption for sensitive data at rest."""

from cryptography.fernet import Fernet

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FieldEncryptor:
    """Encrypt/decrypt individual field values using Fernet symmetric encryption.

    Used for encrypting sensitive data stored in the database such as
    API keys, credentials, and other PII.
    """

    def __init__(self, key: bytes | None = None):
        if key:
            self._fernet = Fernet(key)
        else:
            # Use app secret as basis for encryption key
            import hashlib
            import base64
            key_material = hashlib.sha256(settings.jwt_secret_key.encode()).digest()
            self._fernet = Fernet(base64.urlsafe_b64encode(key_material))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string value."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string value."""
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be Fernet-encrypted."""
        try:
            self._fernet.decrypt(value.encode())
            return True
        except Exception:
            return False


field_encryptor = FieldEncryptor()
