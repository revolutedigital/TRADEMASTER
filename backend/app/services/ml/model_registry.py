"""Model registry: versioning, promotion, and lifecycle management for ML models."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.signal import ModelMetadata

logger = get_logger(__name__)


class ModelRegistry:
    """Manages ML model lifecycle: registration, promotion, and retrieval."""

    async def register_model(
        self,
        db: AsyncSession,
        model_type: str,
        symbol: str,
        version: str,
        artifact_path: str,
        accuracy: float | None = None,
        sharpe_ratio: float | None = None,
        profit_factor: float | None = None,
        training_samples: int | None = None,
    ) -> ModelMetadata:
        """Register a newly trained model."""
        metadata = ModelMetadata(
            model_type=model_type,
            version=version,
            symbol=symbol,
            accuracy=accuracy,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            artifact_path=artifact_path,
            is_active=False,
            trained_at=datetime.now(timezone.utc),
            training_samples=training_samples,
        )
        db.add(metadata)
        await db.flush()

        logger.info(
            "model_registered",
            model_type=model_type,
            symbol=symbol,
            version=version,
            accuracy=accuracy,
        )
        return metadata

    async def get_active_model(
        self, db: AsyncSession, model_type: str, symbol: str
    ) -> ModelMetadata | None:
        """Get the currently active model for a given type and symbol."""
        result = await db.execute(
            select(ModelMetadata).where(
                ModelMetadata.model_type == model_type,
                ModelMetadata.symbol == symbol,
                ModelMetadata.is_active == True,
            )
        )
        return result.scalar_one_or_none()

    async def promote_model(self, db: AsyncSession, model_id: int) -> ModelMetadata:
        """Promote a model to active, deactivating any current active model."""
        # Get the model to promote
        result = await db.execute(
            select(ModelMetadata).where(ModelMetadata.id == model_id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Deactivate current active model for same type+symbol
        active_result = await db.execute(
            select(ModelMetadata).where(
                ModelMetadata.model_type == model.model_type,
                ModelMetadata.symbol == model.symbol,
                ModelMetadata.is_active == True,
            )
        )
        current_active = active_result.scalars().all()
        for m in current_active:
            m.is_active = False

        # Activate the new model
        model.is_active = True
        await db.flush()

        logger.info(
            "model_promoted",
            model_id=model_id,
            model_type=model.model_type,
            symbol=model.symbol,
            version=model.version,
        )
        return model

    async def list_models(
        self,
        db: AsyncSession,
        model_type: str | None = None,
        symbol: str | None = None,
        limit: int = 20,
    ) -> list[ModelMetadata]:
        """List registered models with optional filters."""
        query = select(ModelMetadata).order_by(ModelMetadata.trained_at.desc()).limit(limit)
        if model_type:
            query = query.where(ModelMetadata.model_type == model_type)
        if symbol:
            query = query.where(ModelMetadata.symbol == symbol)
        result = await db.execute(query)
        return list(result.scalars().all())


model_registry = ModelRegistry()
