"""API guard tests for the isolated Testnet native-OCO release proof."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.trading import (
    TestnetProtectionVerificationRequest,
    verify_testnet_native_spot_protection,
)


@pytest.mark.asyncio
async def test_testnet_protection_proof_requires_the_trading_engine_to_be_stopped() -> None:
    body = TestnetProtectionVerificationRequest(
        confirmation_phrase="VERIFY TESTNET OCO",
        symbol="BTCUSDT",
    )
    database = AsyncMock()
    engine = SimpleNamespace(_running=True)

    with (
        patch("app.api.v1.trading.testnet_protection_verifier.verify", new=AsyncMock()) as verify,
        pytest.raises(HTTPException, match="stop the trading engine") as error,
    ):
        await verify_testnet_native_spot_protection(
            body=body,
            db=database,
            user={"sub": "operator"},
            engine=engine,
        )

    assert error.value.status_code == 409
    verify.assert_not_awaited()
