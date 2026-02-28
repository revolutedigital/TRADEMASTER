"""Tests for security utilities."""

from app.core.security import (
    APIKeyEncryptor,
    create_access_token,
    hash_password,
    verify_password,
    verify_token,
)


def test_password_hash_and_verify():
    password = "super_secret_123"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False


def test_jwt_create_and_verify():
    token = create_access_token({"sub": "user123", "role": "admin"})
    payload = verify_token(token)
    assert payload is not None
    assert payload["sub"] == "user123"
    assert payload["role"] == "admin"


def test_jwt_invalid_token():
    payload = verify_token("invalid.token.here")
    assert payload is None


def test_api_key_encryption():
    encryptor = APIKeyEncryptor()
    original = "my_binance_api_key_12345"
    encrypted = encryptor.encrypt(original)
    assert encrypted != original
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == original
