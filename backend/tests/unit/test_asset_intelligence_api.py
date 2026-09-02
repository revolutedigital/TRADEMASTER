"""API-level safety and response tests for the one-asset research workflow."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.asset_intelligence import (
    create_asset_study,
    get_market_opportunity_scan,
    list_eligible_assets,
    start_market_opportunity_scan,
)
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


@pytest.mark.asyncio
async def test_market_scan_is_queued_without_activation_or_order() -> None:
    database = AsyncMock()
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers.get.return_value = "pytest"
    scan = MagicMock(id=37)
    response = {
        "id": 37,
        "status": "QUEUED",
        "total_assets": 0,
        "screened_assets": 0,
        "shortlisted_assets": 0,
        "studied_assets": 0,
        "failed_assets": 0,
        "message": "Aguardando início da varredura do mercado.",
        "candidates": [],
        "started_at": None,
        "completed_at": None,
    }

    with (
        patch(
            "app.api.v1.asset_intelligence.market_opportunity_scan_service.start_or_reuse",
            new=AsyncMock(return_value=(scan, True)),
        ) as start_scan,
        patch(
            "app.api.v1.asset_intelligence.market_opportunity_scan_service.get",
            new=AsyncMock(return_value=scan),
        ),
        patch("app.api.v1.asset_intelligence.serialize_scan", return_value=response),
        patch(
            "app.api.v1.asset_intelligence.market_opportunity_scan_service.launch"
        ) as launch_scan,
        patch("app.core.audit.audit_logger.log_event", new=AsyncMock()) as audit_event,
    ):
        result = await start_market_opportunity_scan(
            request=request,
            db=database,
            user={"sub": "operator"},
        )

    assert result == response
    start_scan.assert_awaited_once_with(database, requested_by="operator")
    database.commit.assert_awaited_once()
    launch_scan.assert_called_once_with(37)
    audit_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_market_scan_is_reused_without_a_409_or_duplicate_launch() -> None:
    database = AsyncMock()
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers.get.return_value = "pytest"
    scan = MagicMock(id=37)
    response = {
        "id": 37,
        "status": "RUNNING",
        "total_assets": 250,
        "screened_assets": 10,
        "shortlisted_assets": 0,
        "studied_assets": 0,
        "failed_assets": 0,
        "message": "Triagem: 10 de 250 ativos analisados.",
        "candidates": [],
        "started_at": "2026-09-01T00:00:00+00:00",
        "completed_at": None,
    }

    with (
        patch(
            "app.api.v1.asset_intelligence.market_opportunity_scan_service.start_or_reuse",
            new=AsyncMock(return_value=(scan, False)),
        ),
        patch(
            "app.api.v1.asset_intelligence.market_opportunity_scan_service.get",
            new=AsyncMock(return_value=scan),
        ),
        patch("app.api.v1.asset_intelligence.serialize_scan", return_value=response),
        patch(
            "app.api.v1.asset_intelligence.market_opportunity_scan_service.launch"
        ) as launch_scan,
    ):
        result = await start_market_opportunity_scan(
            request=request,
            db=database,
            user={"sub": "operator"},
        )

    assert result == response
    database.commit.assert_not_awaited()
    launch_scan.assert_not_called()


@pytest.mark.asyncio
async def test_market_scan_read_returns_404_when_the_id_does_not_exist() -> None:
    with (
        patch(
            "app.api.v1.asset_intelligence.market_opportunity_scan_service.get",
            new=AsyncMock(return_value=None),
        ),
        pytest.raises(HTTPException, match="was not found") as error,
    ):
        await get_market_opportunity_scan(scan_id=99, db=AsyncMock(), _user={"sub": "operator"})

    assert error.value.status_code == 404
