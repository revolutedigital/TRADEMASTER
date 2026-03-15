"""Risk alert dispatcher: sends notifications when circuit breaker state changes.

Supports webhook (Slack/Discord/custom), and logs. Easily extensible.
"""

import asyncio

import httpx

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RiskAlertDispatcher:
    """Dispatches risk alerts via configured channels."""

    async def send_circuit_breaker_alert(
        self,
        prev_state: str,
        new_state: str,
        daily_dd: float,
        weekly_dd: float,
        total_dd: float,
        equity: float,
    ) -> None:
        """Send alert when circuit breaker state changes."""
        severity = self._severity(new_state)
        message = (
            f"[TradeMaster] Circuit Breaker: {prev_state} -> {new_state}\n"
            f"Severity: {severity}\n"
            f"Daily DD: {daily_dd:.2%} | Weekly DD: {weekly_dd:.2%} | Total DD: {total_dd:.2%}\n"
            f"Current Equity: ${equity:,.2f}"
        )

        logger.warning(
            "risk_alert_dispatched",
            prev_state=prev_state,
            new_state=new_state,
            severity=severity,
            equity=equity,
        )

        # Send to webhook if configured
        webhook_url = getattr(settings, "risk_alert_webhook_url", None)
        if webhook_url:
            await self._send_webhook(webhook_url, message, severity)

    async def send_trade_alert(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        signal_strength: float,
    ) -> None:
        """Send alert for executed trades (optional, for monitoring)."""
        webhook_url = getattr(settings, "trade_alert_webhook_url", None)
        if not webhook_url:
            return

        message = (
            f"[TradeMaster] Trade Executed\n"
            f"Symbol: {symbol} | Side: {side}\n"
            f"Qty: {quantity:.6f} | Price: ${price:,.2f}\n"
            f"Signal: {signal_strength:+.4f}"
        )
        await self._send_webhook(webhook_url, message, "info")

    async def _send_webhook(self, url: str, message: str, severity: str) -> None:
        """Send alert to a webhook URL (Slack/Discord compatible)."""
        try:
            # Detect format by URL pattern
            if "discord" in url:
                payload = {"content": message}
            elif "slack" in url:
                emoji = {"critical": ":rotating_light:", "warning": ":warning:", "info": ":information_source:"}.get(severity, "")
                payload = {"text": f"{emoji} {message}"}
            else:
                # Generic webhook
                payload = {
                    "text": message,
                    "severity": severity,
                    "source": "trademaster",
                }

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    logger.warning("webhook_send_failed", status=resp.status_code)
        except Exception as e:
            logger.warning("webhook_send_error", error=str(e))

    @staticmethod
    def _severity(state: str) -> str:
        if state == "HALTED":
            return "critical"
        if state in ("PAUSED", "REDUCED"):
            return "warning"
        return "info"


risk_alert_dispatcher = RiskAlertDispatcher()
