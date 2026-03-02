"""Webhook dispatcher for external integrations.

Supports: Discord, Telegram, Slack, custom HTTP webhooks.
"""

import asyncio
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)


class WebhookType(str, Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    CUSTOM = "custom"
    TRADINGVIEW = "tradingview"


@dataclass
class WebhookConfig:
    id: str
    name: str
    type: WebhookType
    url: str
    secret: str | None = None
    events: list[str] | None = None  # ["trade.executed", "alert.triggered", "circuit_breaker.activated"]
    is_active: bool = True


class WebhookDispatcher:
    """Dispatch events to configured webhook endpoints."""

    def __init__(self):
        self._webhooks: dict[str, WebhookConfig] = {}
        self._client: httpx.AsyncClient | None = None

    async def start(self):
        self._client = httpx.AsyncClient(timeout=10.0)

    async def stop(self):
        if self._client:
            await self._client.aclose()

    def register_webhook(self, config: WebhookConfig) -> None:
        self._webhooks[config.id] = config
        logger.info("webhook_registered", id=config.id, type=config.type, events=config.events)

    def unregister_webhook(self, webhook_id: str) -> None:
        self._webhooks.pop(webhook_id, None)
        logger.info("webhook_unregistered", id=webhook_id)

    async def dispatch(self, event: str, payload: dict) -> None:
        """Dispatch an event to all matching webhooks."""
        for webhook in self._webhooks.values():
            if not webhook.is_active:
                continue
            if webhook.events and event not in webhook.events:
                continue
            asyncio.create_task(self._send(webhook, event, payload))

    async def _send(self, webhook: WebhookConfig, event: str, payload: dict) -> None:
        """Send payload to a specific webhook."""
        if not self._client:
            return

        try:
            body = self._format_payload(webhook.type, event, payload)
            headers = {"Content-Type": "application/json"}

            # Add HMAC signature for custom webhooks
            if webhook.secret and webhook.type == WebhookType.CUSTOM:
                signature = hmac.new(
                    webhook.secret.encode(),
                    json.dumps(body, sort_keys=True).encode(),
                    hashlib.sha256,
                ).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={signature}"

            response = await self._client.post(webhook.url, json=body, headers=headers)
            logger.debug("webhook_sent", id=webhook.id, event=event, status=response.status_code)
        except Exception as e:
            logger.warning("webhook_send_failed", id=webhook.id, event=event, error=str(e))

    def _format_payload(self, webhook_type: WebhookType, event: str, payload: dict) -> dict:
        """Format payload based on webhook type."""
        timestamp = datetime.now(timezone.utc).isoformat()

        if webhook_type == WebhookType.DISCORD:
            return {
                "embeds": [{
                    "title": f"TradeMaster: {event}",
                    "description": json.dumps(payload, indent=2),
                    "color": 3447003,  # Blue
                    "timestamp": timestamp,
                    "footer": {"text": "TradeMaster Trading Bot"},
                }]
            }

        if webhook_type == WebhookType.SLACK:
            return {
                "text": f"*TradeMaster: {event}*",
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": f"TradeMaster: {event}"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"```{json.dumps(payload, indent=2)}```"}},
                ],
            }

        if webhook_type == WebhookType.TELEGRAM:
            text = f"*TradeMaster: {event}*\n"
            for key, value in payload.items():
                text += f"_{key}_: `{value}`\n"
            return {"text": text, "parse_mode": "Markdown"}

        # Custom / TradingView
        return {
            "event": event,
            "timestamp": timestamp,
            "data": payload,
            "source": "trademaster",
        }


webhook_dispatcher = WebhookDispatcher()
