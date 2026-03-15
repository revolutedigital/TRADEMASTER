"""Price alert monitoring service."""
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.alert import PriceAlert

logger = get_logger(__name__)


class AlertChecker:
    """Check price alerts against current market prices."""

    async def check_alerts(self, db: AsyncSession, current_prices: dict[str, Decimal]) -> list[PriceAlert]:
        """Check all active alerts against current prices. Returns triggered alerts."""
        result = await db.execute(
            select(PriceAlert).where(
                PriceAlert.is_active == True,
                PriceAlert.is_triggered == False,
            )
        )
        alerts = result.scalars().all()
        triggered = []

        for alert in alerts:
            price = current_prices.get(alert.symbol)
            if price is None:
                continue

            price_float = float(price)
            should_trigger = False

            if alert.condition == "above" and price_float >= alert.target_price:
                should_trigger = True
            elif alert.condition == "below" and price_float <= alert.target_price:
                should_trigger = True

            if should_trigger:
                alert.is_triggered = True
                alert.triggered_at = datetime.now(timezone.utc)
                triggered.append(alert)
                logger.info("alert_triggered", symbol=alert.symbol,
                           condition=alert.condition, target=alert.target_price,
                           current=price_float)

        if triggered:
            await db.commit()

        return triggered

    async def create_alert(
        self, db: AsyncSession, symbol: str, condition: str,
        target_price: float, notes: str | None = None,
    ) -> PriceAlert:
        """Create a new price alert."""
        if condition not in ("above", "below"):
            raise ValueError("condition must be 'above' or 'below'")

        alert = PriceAlert(
            symbol=symbol.upper(),
            condition=condition,
            target_price=target_price,
            notes=notes,
        )
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        logger.info("alert_created", symbol=symbol, condition=condition, target=target_price)
        return alert

    async def get_alerts(self, db: AsyncSession, active_only: bool = True) -> list[PriceAlert]:
        """Get all alerts, optionally filtered to active only."""
        query = select(PriceAlert).order_by(PriceAlert.created_at.desc())
        if active_only:
            query = query.where(PriceAlert.is_active == True)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def delete_alert(self, db: AsyncSession, alert_id: int) -> bool:
        """Deactivate an alert."""
        result = await db.execute(
            update(PriceAlert).where(PriceAlert.id == alert_id).values(is_active=False)
        )
        await db.commit()
        return result.rowcount > 0


alert_checker = AlertChecker()
