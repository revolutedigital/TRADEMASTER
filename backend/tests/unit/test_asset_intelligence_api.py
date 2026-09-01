"""API-level safety and response tests for the one-asset research workflow."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.asset_intelligence import create_asset_study, list_eligible_assets
from app.schemas.asset_intelligence import AssetStudyCreateRequest
from app.services.market.spot_asset_catalog import SpotAsset


@pytest.mark.asyncio
async def test_universe_returns_only_catalog_eligible_assets() -> None:
    asset = SpotAsset(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        quote_volume_24h=2_000_000.0,
        price_change_pct_24h=3.4,
    )
    generated_at = datetime(2026, 9, 1, tzinfo=UTC)

    with patch(
        "app.api.v1.asset_intelligence.spot_asset_catalog.list",
        new=AsyncMock(return_value=([asset], generated_at)),
    ) as list_catalog:
        response = await list_eligible_assets(search="sol", limit=25, _user={"sub": "operator"})

    assert response["assets"] == [asset.as_dict()]
    assert response["generated_at"] == generated_at.isoformat()
    list_catalog.assert_awaited_once_with(search="sol", limit=25)


@pytest.mark.asyncio
async def test_study_commits_evidence_but_does_not_activate_or_order() -> None:
    database = AsyncMock()
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers.get.return_value = "pytest"
    study = {
        "symbol": "SOLUSDT",
        "execution_mode": "TESTNET",
        "market_study": {
            "trend": "UPTREND",
            "volatility_pct": 3.1,
            "liquidity_quote_volume_24h": 2_000_000.0,
            "candles": 4_000,
        },
        "predictive_model": {
            "model_type": "xgboost",
            "trained": True,
            "validation_accuracy": 0.54,
            "samples": 3_900,
            "latest_signal": "BUY",
        },
        "recommendation": {
            "strategy_name": "Tendência SMA + RSI",
            "backtest_id": 11,
            "deployment_id": 12,
            "deployment_status": "APPROVED",
            "reasons": [],
        },
    }

    with (
        patch("app.api.v1.asset_intelligence.study_asset", new=AsyncMock(return_value=study)) as study_asset,
        patch("app.core.audit.audit_logger.log_event", new=AsyncMock()) as audit_event,
    ):
        response = await create_asset_study(
            body=AssetStudyCreateRequest(symbol="SOLUSDT"),
            request=request,
            db=database,
            user={"sub": "operator"},
        )

    assert response == study
    study_asset.assert_awaited_once_with(database, symbol="SOLUSDT")
    database.commit.assert_awaited_once()
    database.rollback.assert_not_awaited()
    audit_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_asset_study_rolls_back_without_an_activation_path() -> None:
    database = AsyncMock()

    with (
        patch(
            "app.api.v1.asset_intelligence.study_asset",
            new=AsyncMock(side_effect=ValueError("pair is not eligible")),
        ),
        pytest.raises(HTTPException, match="pair is not eligible") as error,
    ):
        await create_asset_study(
            body=AssetStudyCreateRequest(symbol="BADUSDT"),
            request=MagicMock(),
            db=database,
            user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    database.rollback.assert_awaited_once()
    database.commit.assert_not_awaited()
