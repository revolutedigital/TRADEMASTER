"""Price alert checker - evaluates alerts against current prices."""

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.alert import PriceAlert

logger = get_logger(__name__)


class AlertChecker:
    """Checks price alerts against current market prices and triggers notifications."""

    async def check_alerts(self, db: AsyncSession, current_prices: dict[str, float]) -> list[dict]:
        """
        Check all active alerts against current prices.

        Args:
            db: Database session
            current_prices: Dict of symbol -> current price

        Returns:
            List of triggered alert details
        """
        triggered = []

        result = await db.execute(
            select(PriceAlert).where(
                PriceAlert.is_active == True,  # noqa: E712
                PriceAlert.is_triggered == False,  # noqa: E712
            )
        )
        alerts = result.scalars().all()

        for alert in alerts:
            price = current_prices.get(alert.symbol)
            if price is None:
                continue

            should_trigger = False
            if alert.condition == "above" and price >= float(alert.target_price):
                should_trigger = True
            elif alert.condition == "below" and price <= float(alert.target_price):
                should_trigger = True

            if should_trigger:
                await db.execute(
                    update(PriceAlert)
                    .where(PriceAlert.id == alert.id)
                    .values(
                        is_triggered=True,
                        triggered_at=datetime.now(timezone.utc),
                    )
                )
                triggered.append({
                    "alert_id": alert.id,
                    "symbol": alert.symbol,
                    "condition": alert.condition,
                    "target_price": float(alert.target_price),
                    "current_price": price,
                    "triggered_at": datetime.now(timezone.utc).isoformat(),
                })
                logger.info("alert_triggered", alert_id=alert.id, symbol=alert.symbol,
                           condition=alert.condition, target=float(alert.target_price), price=price)

        if triggered:
            await db.commit()

        return triggered

    async def get_active_count(self, db: AsyncSession) -> int:
        """Get count of active, untriggered alerts."""
        result = await db.execute(
            select(PriceAlert).where(
                PriceAlert.is_active == True,  # noqa: E712
                PriceAlert.is_triggered == False,  # noqa: E712
            )
        )
        return len(result.scalars().all())


alert_checker = AlertChecker()
