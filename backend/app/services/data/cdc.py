"""Change Data Capture: publish events when data changes."""

from app.core.events import event_bus
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChangeDataCapture:
    """Publish events on data mutations for downstream consumers."""

    async def on_trade_created(self, trade_id: str, symbol: str, side: str, quantity: float, price: float) -> None:
        await event_bus.publish("cdc.trade.created", {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        })
        logger.debug("cdc_trade_created", trade_id=trade_id)

    async def on_position_opened(self, position_id: str, symbol: str, side: str) -> None:
        await event_bus.publish("cdc.position.opened", {
            "position_id": position_id,
            "symbol": symbol,
            "side": side,
        })
        logger.debug("cdc_position_opened", position_id=position_id)

    async def on_position_closed(self, position_id: str, symbol: str, pnl: float) -> None:
        await event_bus.publish("cdc.position.closed", {
            "position_id": position_id,
            "symbol": symbol,
            "realized_pnl": pnl,
        })
        logger.debug("cdc_position_closed", position_id=position_id, pnl=pnl)

    async def on_signal_generated(self, signal_id: str, symbol: str, action: str, confidence: float) -> None:
        await event_bus.publish("cdc.signal.generated", {
            "signal_id": signal_id,
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
        })
        logger.debug("cdc_signal_generated", signal_id=signal_id)

    async def on_config_changed(self, key: str, old_value: str, new_value: str) -> None:
        await event_bus.publish("cdc.config.changed", {
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
        })
        logger.info("cdc_config_changed", key=key)


cdc = ChangeDataCapture()
