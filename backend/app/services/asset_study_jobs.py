"""Asynchronous full-asset research jobs that protect the HTTP request budget."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.asset_study_job import AssetStudyJob
from app.models.base import async_session_factory
from app.services.asset_intelligence import AssetIntelligenceError, study_asset
from app.services.market.public_binance_data import PublicMarketDataUnavailable

logger = get_logger(__name__)

ACTIVE_STUDY_STATUSES = ("QUEUED", "RUNNING")


class AssetStudyJobService:
    """Owns the single full study that is allowed to run at a time."""

    def __init__(self) -> None:
        self._tasks: dict[int, asyncio.Task[None]] = {}

    async def start_or_reuse(
        self,
        db: AsyncSession,
        *,
        symbol: str,
        requested_by: str,
    ) -> tuple[AssetStudyJob, bool]:
        """Create a durable job or return the active one without a 500/409 loop."""
        existing = await self._active_job(db)
        if existing is not None:
            return existing, False

        job = AssetStudyJob(
            symbol=symbol.upper(),
            requested_by=requested_by[:128],
            status="QUEUED",
            message="Aguardando início do estudo completo.",
        )
        db.add(job)
        try:
            await db.flush()
        except IntegrityError:
            await db.rollback()
            existing = await self._active_job(db)
            if existing is not None:
                return existing, False
            raise
        return job, True

    def launch(self, job_id: int) -> None:
        """Run a new job once; repeated requests continue observing the same job."""
        existing = self._tasks.get(job_id)
        if existing and not existing.done():
            return
        task = asyncio.create_task(self._run(job_id), name=f"asset_study_job_{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))

    async def get(self, db: AsyncSession, job_id: int) -> AssetStudyJob | None:
        return await db.get(AssetStudyJob, job_id)

    async def recover_interrupted_jobs(self) -> int:
        async with async_session_factory() as db:
            result = await db.execute(
                update(AssetStudyJob)
                .where(AssetStudyJob.status.in_(ACTIVE_STUDY_STATUSES))
                .values(
                    status="INTERRUPTED",
                    completed_at=datetime.now(UTC),
                    message="O estudo foi interrompido por reinício do serviço. Inicie novamente.",
                )
            )
            await db.commit()
        return int(result.rowcount or 0)

    async def stop_all(self) -> None:
        active = list(self._tasks.items())
        for _job_id, task in active:
            task.cancel()
        if active:
            await asyncio.gather(*(task for _job_id, task in active), return_exceptions=True)
        for job_id, _task in active:
            await self._interrupt_job(job_id, "O estudo foi interrompido pelo desligamento do serviço.")
        self._tasks.clear()

    async def _run(self, job_id: int) -> None:
        try:
            async with async_session_factory() as db:
                job = await self._require_job(db, job_id)
                if job.status != "QUEUED":
                    return
                job.status = "RUNNING"
                job.started_at = datetime.now(UTC)
                job.message = f"Estudando {job.symbol}: dados, modelo, backtest e validação."
                await db.commit()

                symbol = job.symbol
                try:
                    study = await study_asset(db, symbol=symbol)
                    job.study_json = json.dumps(study, ensure_ascii=False)
                    job.status = "COMPLETED"
                    job.completed_at = datetime.now(UTC)
                    job.message = "Estudo concluído. Nenhuma estratégia foi ativada."
                    await db.commit()
                except (AssetIntelligenceError, PublicMarketDataUnavailable, ValueError) as exc:
                    await db.rollback()
                    await self._mark_failed(job_id, _safe_error_message(str(exc)))
                except Exception:
                    await db.rollback()
                    logger.exception("asset_study_job_failed", job_id=job_id, symbol=symbol)
                    await self._mark_failed(
                        job_id,
                        "O estudo não pôde ser concluído agora. Tente novamente em alguns minutos.",
                    )
        except asyncio.CancelledError:
            await self._interrupt_job(job_id, "O estudo foi interrompido pelo desligamento do serviço.")
            raise
        except Exception:
            logger.exception("asset_study_job_lifecycle_failed", job_id=job_id)
            await self._mark_failed(
                job_id,
                "O estudo não pôde ser preparado agora. Tente novamente em alguns minutos.",
            )

    async def _mark_failed(self, job_id: int, message: str) -> None:
        async with async_session_factory() as db:
            result = await db.execute(
                update(AssetStudyJob)
                .where(AssetStudyJob.id == job_id, AssetStudyJob.status.in_(ACTIVE_STUDY_STATUSES))
                .values(status="FAILED", completed_at=datetime.now(UTC), message=message, error_message=message)
            )
            if result.rowcount:
                await db.commit()

    async def _interrupt_job(self, job_id: int, message: str) -> None:
        async with async_session_factory() as db:
            result = await db.execute(
                update(AssetStudyJob)
                .where(AssetStudyJob.id == job_id, AssetStudyJob.status.in_(ACTIVE_STUDY_STATUSES))
                .values(status="INTERRUPTED", completed_at=datetime.now(UTC), message=message)
            )
            if result.rowcount:
                await db.commit()

    async def _active_job(self, db: AsyncSession) -> AssetStudyJob | None:
        result = await db.execute(
            select(AssetStudyJob)
            .where(AssetStudyJob.status.in_(ACTIVE_STUDY_STATUSES))
            .order_by(AssetStudyJob.created_at.desc(), AssetStudyJob.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _require_job(self, db: AsyncSession, job_id: int) -> AssetStudyJob:
        job = await db.get(AssetStudyJob, job_id)
        if job is None:
            raise LookupError(f"Asset study job {job_id} was not found")
        return job


def serialize_study_job(job: AssetStudyJob) -> dict[str, object]:
    return {
        "id": job.id,
        "symbol": job.symbol,
        "status": job.status,
        "message": job.message,
        "study": _decode_study(job.study_json),
        "error_message": job.error_message,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def _decode_study(study_json: str | None) -> dict[str, object] | None:
    if not study_json:
        return None
    try:
        decoded = json.loads(study_json)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _safe_error_message(message: str) -> str:
    return message[:300] if message else "O estudo não pôde ser concluído agora."


asset_study_job_service = AssetStudyJobService()
