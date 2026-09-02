"""Safe, persisted market-wide research scans with no execution side effects."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.base import async_session_factory
from app.models.market_opportunity_scan import (
    MarketOpportunityCandidate as PersistedOpportunityCandidate,
)
from app.models.market_opportunity_scan import MarketOpportunityScan
from app.services.asset_intelligence import AssetIntelligenceError, study_asset
from app.services.market.public_binance_data import (
    PublicMarketDataUnavailable,
    public_binance_market_data,
)
from app.services.market.spot_asset_catalog import MIN_QUOTE_VOLUME_24H, SpotAsset, spot_asset_catalog

logger = get_logger(__name__)

MAX_SCANNED_ASSETS = 500
MAX_FULL_STUDIES = 6
SCREENING_CANDLE_LIMIT = 240
SCREENING_CONCURRENCY = 4
PROGRESS_COMMIT_INTERVAL = 10
ACTIVE_SCAN_STATUSES = ("QUEUED", "RUNNING")


@dataclass(frozen=True)
class ScreeningResult:
    """A deterministic short-listing record, never a trading instruction."""

    asset: SpotAsset
    score: float
    market_trend: str


class MarketOpportunityScanService:
    """Coordinates the one allowed non-executing market-wide research job."""

    def __init__(self) -> None:
        self._tasks: dict[int, asyncio.Task[None]] = {}

    async def start_or_reuse(
        self,
        db: AsyncSession,
        *,
        requested_by: str,
    ) -> tuple[MarketOpportunityScan, bool]:
        """Create one scan or return the existing active scan without a 409 loop."""
        existing = await self._active_scan(db)
        if existing is not None:
            return existing, False

        scan = MarketOpportunityScan(
            requested_by=requested_by[:128],
            status="QUEUED",
            message="Aguardando início da varredura do mercado.",
        )
        db.add(scan)
        try:
            await db.flush()
        except IntegrityError:
            # The database singleton index is the race-safe source of truth.
            # A parallel click should resume its scan, not give the operator a
            # meaningless conflict error.
            await db.rollback()
            existing = await self._active_scan(db)
            if existing is not None:
                return existing, False
            raise
        return scan, True

    def launch(self, scan_id: int) -> None:
        """Launch the persisted scan once; repeated POSTs keep polling the same one."""
        existing = self._tasks.get(scan_id)
        if existing and not existing.done():
            return

        task = asyncio.create_task(self._run(scan_id), name=f"market_opportunity_scan_{scan_id}")
        self._tasks[scan_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(scan_id, None))

    async def get(self, db: AsyncSession, scan_id: int) -> MarketOpportunityScan | None:
        result = await db.execute(
            select(MarketOpportunityScan)
            .options(selectinload(MarketOpportunityScan.candidates))
            .where(MarketOpportunityScan.id == scan_id)
        )
        return result.scalar_one_or_none()

    async def recover_interrupted_scans(self) -> int:
        """Make restarts honest: a vanished asyncio task is never reported as running."""
        async with async_session_factory() as db:
            result = await db.execute(
                update(MarketOpportunityScan)
                .where(MarketOpportunityScan.status.in_(ACTIVE_SCAN_STATUSES))
                .values(
                    status="INTERRUPTED",
                    completed_at=datetime.now(UTC),
                    message="A varredura foi interrompida por reinício do serviço. Inicie uma nova busca.",
                )
            )
            await db.commit()
        return int(result.rowcount or 0)

    async def stop_all(self) -> None:
        """Cancel in-process scans and persist their interruption before shutdown."""
        active = list(self._tasks.items())
        for _scan_id, task in active:
            task.cancel()
        if active:
            await asyncio.gather(*(task for _scan_id, task in active), return_exceptions=True)
        for scan_id, _task in active:
            await self._interrupt_scan(scan_id, "A varredura foi interrompida pelo desligamento do serviço.")
        self._tasks.clear()

    async def _run(self, scan_id: int) -> None:
        try:
            assets, _generated_at = await spot_asset_catalog.list(limit=MAX_SCANNED_ASSETS)
            if not assets:
                await self._complete_without_assets(scan_id)
                return

            async with async_session_factory() as db:
                scan = await self._require_scan(db, scan_id)
                if scan.status != "QUEUED":
                    return
                scan.status = "RUNNING"
                scan.started_at = datetime.now(UTC)
                scan.total_assets = len(assets)
                scan.message = f"Triagem iniciada para {len(assets)} ativos líquidos."
                await db.commit()

                finalists = await self._screen_assets(db, scan, assets)
                persisted_finalists = await self._persist_finalists(db, scan, finalists)
                await self._study_finalists(db, scan, persisted_finalists)

                scan.status = "COMPLETED"
                scan.completed_at = datetime.now(UTC)
                scan.message = (
                    "Varredura concluída: "
                    f"{scan.studied_assets} ativos passaram pelo estudo completo. "
                    "Nenhuma estratégia foi ativada."
                )
                await db.commit()
        except asyncio.CancelledError:
            await self._interrupt_scan(scan_id, "A varredura foi interrompida pelo desligamento do serviço.")
            raise
        except Exception as exc:
            logger.exception("market_opportunity_scan_failed", scan_id=scan_id, error=str(exc))
            await self._fail_scan(scan_id)

    async def _screen_assets(
        self,
        db: AsyncSession,
        scan: MarketOpportunityScan,
        assets: list[SpotAsset],
    ) -> list[ScreeningResult]:
        semaphore = asyncio.Semaphore(SCREENING_CONCURRENCY)

        async def screen(asset: SpotAsset) -> ScreeningResult | None:
            async with semaphore:
                try:
                    candles = await public_binance_market_data.klines(
                        symbol=asset.symbol,
                        interval="1h",
                        limit=SCREENING_CANDLE_LIMIT,
                    )
                    return screen_asset(asset, candles)
                except (AssetIntelligenceError, PublicMarketDataUnavailable, ValueError) as exc:
                    logger.info(
                        "market_opportunity_asset_skipped",
                        scan_id=scan.id,
                        symbol=asset.symbol,
                        error=str(exc),
                    )
                    return None
                except Exception as exc:
                    logger.warning(
                        "market_opportunity_asset_screen_failed",
                        scan_id=scan.id,
                        symbol=asset.symbol,
                        error=str(exc),
                    )
                    return None

        tasks = [asyncio.create_task(screen(asset)) for asset in assets]
        results: list[ScreeningResult] = []
        for completed in asyncio.as_completed(tasks):
            result = await completed
            scan.screened_assets += 1
            if result is None:
                scan.failed_assets += 1
            else:
                results.append(result)

            if scan.screened_assets % PROGRESS_COMMIT_INTERVAL == 0 or scan.screened_assets == scan.total_assets:
                scan.message = f"Triagem: {scan.screened_assets} de {scan.total_assets} ativos analisados."
                await db.commit()

        return sorted(
            results,
            key=lambda result: (
                -result.score,
                -result.asset.quote_volume_24h,
                result.asset.symbol,
            ),
        )[:MAX_FULL_STUDIES]

    async def _persist_finalists(
        self,
        db: AsyncSession,
        scan: MarketOpportunityScan,
        finalists: list[ScreeningResult],
    ) -> list[PersistedOpportunityCandidate]:
        rows = [
            PersistedOpportunityCandidate(
                scan_id=scan.id,
                rank=rank,
                symbol=result.asset.symbol,
                screening_score=result.score,
                market_trend=result.market_trend,
                price_change_pct_24h=result.asset.price_change_pct_24h,
                quote_volume_24h=result.asset.quote_volume_24h,
                status="SHORTLISTED",
            )
            for rank, result in enumerate(finalists, start=1)
        ]
        db.add_all(rows)
        scan.shortlisted_assets = len(rows)
        scan.message = (
            "Triagem concluída. Estudo completo do modelo e da estratégia "
            f"iniciado para os {len(rows)} melhores candidatos."
        )
        await db.commit()
        return rows

    async def _study_finalists(
        self,
        db: AsyncSession,
        scan: MarketOpportunityScan,
        candidates: list[PersistedOpportunityCandidate],
    ) -> None:
        candidate_ids = [candidate.id for candidate in candidates]
        for index, candidate_id in enumerate(candidate_ids, start=1):
            candidate = await db.get(PersistedOpportunityCandidate, candidate_id)
            if candidate is None:
                raise LookupError(f"Market opportunity candidate {candidate_id} was not found")
            symbol = candidate.symbol
            candidate.status = "STUDYING"
            scan.message = (
                "Triagem concluída. Estudo completo em andamento para "
                f"{symbol} ({index} de {len(candidate_ids)})."
            )
            await db.commit()

            try:
                study = await study_asset(db, symbol=symbol)
                candidate.study_json = json.dumps(study, ensure_ascii=False)
                candidate.status = str(study["recommendation"]["deployment_status"])
                scan.studied_assets += 1
                await db.commit()
            except (AssetIntelligenceError, PublicMarketDataUnavailable, ValueError) as exc:
                await db.rollback()
                await self._mark_candidate_failed(
                    db,
                    scan_id=scan.id,
                    candidate_id=candidate.id,
                    error=str(exc),
                )
                scan = await self._require_scan(db, scan.id)
            except Exception:
                await db.rollback()
                logger.exception(
                    "market_opportunity_candidate_study_failed",
                    scan_id=scan.id,
                    symbol=symbol,
                )
                await self._mark_candidate_failed(
                    db,
                    scan_id=scan.id,
                    candidate_id=candidate.id,
                    error="O estudo completo não pôde ser concluído para este ativo.",
                )
                scan = await self._require_scan(db, scan.id)

    async def _mark_candidate_failed(
        self,
        db: AsyncSession,
        *,
        scan_id: int,
        candidate_id: int,
        error: str,
    ) -> None:
        candidate = await db.get(PersistedOpportunityCandidate, candidate_id)
        scan = await self._require_scan(db, scan_id)
        if candidate is None:
            raise LookupError(f"Market opportunity candidate {candidate_id} was not found")
        candidate.status = "FAILED"
        candidate.error_message = _safe_error_message(error)
        scan.failed_assets += 1
        await db.commit()

    async def _complete_without_assets(self, scan_id: int) -> None:
        async with async_session_factory() as db:
            scan = await self._require_scan(db, scan_id)
            scan.status = "COMPLETED"
            scan.started_at = datetime.now(UTC)
            scan.completed_at = datetime.now(UTC)
            scan.message = "Nenhum ativo líquido estava disponível para a varredura."
            await db.commit()

    async def _interrupt_scan(self, scan_id: int, message: str) -> None:
        async with async_session_factory() as db:
            result = await db.execute(
                update(MarketOpportunityScan)
                .where(
                    MarketOpportunityScan.id == scan_id,
                    MarketOpportunityScan.status.in_(ACTIVE_SCAN_STATUSES),
                )
                .values(status="INTERRUPTED", completed_at=datetime.now(UTC), message=message)
            )
            if result.rowcount:
                await db.commit()

    async def _fail_scan(self, scan_id: int) -> None:
        async with async_session_factory() as db:
            result = await db.execute(
                update(MarketOpportunityScan)
                .where(
                    MarketOpportunityScan.id == scan_id,
                    MarketOpportunityScan.status.in_(ACTIVE_SCAN_STATUSES),
                )
                .values(
                    status="FAILED",
                    completed_at=datetime.now(UTC),
                    message="A varredura não pôde ser concluída. Aguarde e tente novamente.",
                )
            )
            if result.rowcount:
                await db.commit()

    async def _active_scan(self, db: AsyncSession) -> MarketOpportunityScan | None:
        result = await db.execute(
            select(MarketOpportunityScan)
            .where(MarketOpportunityScan.status.in_(ACTIVE_SCAN_STATUSES))
            .order_by(MarketOpportunityScan.created_at.desc(), MarketOpportunityScan.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _require_scan(self, db: AsyncSession, scan_id: int) -> MarketOpportunityScan:
        scan = await db.get(MarketOpportunityScan, scan_id)
        if scan is None:
            raise LookupError(f"Market opportunity scan {scan_id} was not found")
        return scan


def screen_asset(asset: SpotAsset, candles: pd.DataFrame) -> ScreeningResult:
    """Score an asset's public data for research priority, never expected return."""
    close = pd.to_numeric(candles.get("close"), errors="coerce").dropna()
    if len(close) < 200:
        raise AssetIntelligenceError("Asset history is insufficient for market-wide screening")

    last_price = float(close.iloc[-1])
    sma_50 = float(close.iloc[-50:].mean())
    sma_200 = float(close.iloc[-200:].mean())
    if last_price > sma_50 > sma_200:
        trend = "UPTREND"
        trend_score = 34.0
    elif last_price < sma_50 < sma_200:
        trend = "DOWNTREND"
        trend_score = 0.0
    else:
        trend = "RANGE"
        trend_score = 15.0

    hourly_returns = close.pct_change().dropna().tail(168)
    if hourly_returns.empty:
        raise AssetIntelligenceError("Asset history is insufficient for volatility screening")
    daily_volatility_pct = float(hourly_returns.std(ddof=0) * math.sqrt(24) * 100)
    volatility_score = max(0.0, 20.0 - abs(daily_volatility_pct - 4.0) * 3.0)

    momentum_pct = float((last_price / float(close.iloc[-25]) - 1.0) * 100)
    # Favour healthy upward movement without presenting an already parabolic
    # candle as an opportunity. Negative momentum gets no priority score.
    momentum_score = max(0.0, 20.0 - abs(momentum_pct - 3.0) * 2.0) if momentum_pct > 0 else 0.0
    liquidity_multiple = max(asset.quote_volume_24h / MIN_QUOTE_VOLUME_24H, 1.0)
    liquidity_score = min(26.0, math.log10(liquidity_multiple) * 8.0)
    score = round(min(100.0, trend_score + volatility_score + momentum_score + liquidity_score), 3)

    return ScreeningResult(asset=asset, score=score, market_trend=trend)


def serialize_scan(scan: MarketOpportunityScan) -> dict[str, object]:
    """Return only research state; stored studies remain explicit and inspectable."""
    return {
        "id": scan.id,
        "status": scan.status,
        "total_assets": scan.total_assets,
        "screened_assets": scan.screened_assets,
        "shortlisted_assets": scan.shortlisted_assets,
        "studied_assets": scan.studied_assets,
        "failed_assets": scan.failed_assets,
        "message": scan.message,
        "candidates": [
            {
                "rank": candidate.rank,
                "symbol": candidate.symbol,
                "screening_score": candidate.screening_score,
                "market_trend": candidate.market_trend,
                "price_change_pct_24h": candidate.price_change_pct_24h,
                "quote_volume_24h": candidate.quote_volume_24h,
                "status": candidate.status,
                "study": _decode_study(candidate.study_json),
                "error_message": candidate.error_message,
            }
            for candidate in scan.candidates
        ],
        "started_at": scan.started_at.isoformat() if scan.started_at else None,
        "completed_at": scan.completed_at.isoformat() if scan.completed_at else None,
    }


def _decode_study(study_json: str | None) -> dict[str, object] | None:
    if not study_json:
        return None
    try:
        decoded = json.loads(study_json)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _safe_error_message(error: str) -> str:
    if isinstance(error, str) and error:
        return error[:300]
    return "O estudo completo não pôde ser concluído para este ativo."


market_opportunity_scan_service = MarketOpportunityScanService()
