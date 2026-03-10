"""Notification webhook dispatcher with retry logic and multi-platform support.

Delivers event notifications to external services (Discord, Telegram, Slack)
and arbitrary HTTP endpoints with exponential-backoff retry, HMAC signing,
and per-webhook event filtering.
"""

import asyncio
import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# Enums & constants
# ======================================================================


class WebhookPlatform(str, Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    CUSTOM = "custom"


class EventType(str, Enum):
    TRADE_EXECUTED = "trade_executed"
    ALERT_TRIGGERED = "alert_triggered"
    CIRCUIT_BREAKER = "circuit_breaker"
    MODEL_RETRAINED = "model_retrained"
    DCA_BUY = "dca_buy"
    STOP_LOSS_HIT = "stop_loss_hit"
    DAILY_SUMMARY = "daily_summary"
    SYSTEM_ERROR = "system_error"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


# Retry defaults
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 15.0


# ======================================================================
# Dataclasses
# ======================================================================


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook endpoint."""

    id: str
    user_id: str
    name: str
    platform: WebhookPlatform
    url: str
    secret: str | None = None
    events: list[str] = field(default_factory=list)  # Empty = all events
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    headers: dict[str, str] = field(default_factory=dict)

    # Telegram-specific
    telegram_chat_id: str | None = None


@dataclass
class DeliveryRecord:
    """Tracks an individual delivery attempt."""

    id: str
    webhook_id: str
    event_type: str
    status: DeliveryStatus
    attempts: int = 0
    last_attempt_at: datetime | None = None
    last_error: str | None = None
    response_code: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ======================================================================
# Payload formatters
# ======================================================================


def _format_discord(event: str, payload: dict, timestamp: str) -> dict:
    """Format a Discord embed payload."""
    color_map = {
        EventType.TRADE_EXECUTED.value: 3066993,   # Green
        EventType.ALERT_TRIGGERED.value: 15105570,  # Orange
        EventType.CIRCUIT_BREAKER.value: 15158332,  # Red
        EventType.MODEL_RETRAINED.value: 3447003,   # Blue
        EventType.STOP_LOSS_HIT.value: 15158332,    # Red
        EventType.DAILY_SUMMARY.value: 10181046,     # Purple
    }

    fields = [
        {"name": k, "value": str(v), "inline": True}
        for k, v in list(payload.items())[:25]  # Discord max 25 fields
    ]

    return {
        "embeds": [{
            "title": f"TradeMaster | {event.replace('_', ' ').title()}",
            "color": color_map.get(event, 3447003),
            "fields": fields,
            "timestamp": timestamp,
            "footer": {"text": "TradeMaster Notifications"},
        }]
    }


def _format_slack(event: str, payload: dict, _timestamp: str) -> dict:
    """Format a Slack Block Kit payload."""
    detail_lines = "\n".join(f"*{k}*: {v}" for k, v in payload.items())
    return {
        "text": f"TradeMaster | {event.replace('_', ' ').title()}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"TradeMaster | {event.replace('_', ' ').title()}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": detail_lines or "_No additional data._"},
            },
        ],
    }


def _format_telegram(event: str, payload: dict, _timestamp: str, chat_id: str | None = None) -> dict:
    """Format a Telegram ``sendMessage`` payload."""
    lines = [f"*TradeMaster | {event.replace('_', ' ').title()}*", ""]
    for k, v in payload.items():
        lines.append(f"_{k}_: `{v}`")

    body: dict = {"text": "\n".join(lines), "parse_mode": "Markdown"}
    if chat_id:
        body["chat_id"] = chat_id
    return body


def _format_custom(event: str, payload: dict, timestamp: str) -> dict:
    """Generic JSON envelope for custom HTTP webhooks."""
    return {
        "event": event,
        "timestamp": timestamp,
        "data": payload,
        "source": "trademaster",
    }


# ======================================================================
# Dispatcher
# ======================================================================


class WebhookDispatcher:
    """Dispatch event notifications to registered webhook endpoints.

    Features:
    - Multi-platform formatting (Discord, Telegram, Slack, custom)
    - Per-webhook event filtering
    - HMAC-SHA256 request signing for custom webhooks
    - Exponential backoff retry with jitter
    - Delivery tracking

    Usage::

        dispatcher = WebhookDispatcher()
        await dispatcher.start()

        dispatcher.register(WebhookConfig(
            id="wh_1",
            user_id="user_123",
            name="My Discord",
            platform=WebhookPlatform.DISCORD,
            url="https://discord.com/api/webhooks/...",
            events=["trade_executed", "alert_triggered"],
        ))

        await dispatcher.dispatch("trade_executed", {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.01})
        await dispatcher.stop()
    """

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        base_backoff: float = BASE_BACKOFF_SECONDS,
        max_backoff: float = MAX_BACKOFF_SECONDS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._webhooks: dict[str, WebhookConfig] = {}
        self._delivery_log: list[DeliveryRecord] = []
        self._client: httpx.AsyncClient | None = None
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._timeout = timeout
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the HTTP client."""
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._running = True
        logger.info("webhook_dispatcher_started")

    async def stop(self) -> None:
        """Gracefully shut down the HTTP client."""
        self._running = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("webhook_dispatcher_stopped")

    # ------------------------------------------------------------------
    # Webhook registration
    # ------------------------------------------------------------------

    def register(self, config: WebhookConfig) -> None:
        """Register a webhook endpoint."""
        self._webhooks[config.id] = config
        logger.info(
            "webhook_registered",
            id=config.id,
            name=config.name,
            platform=config.platform.value,
            events=config.events or ["*"],
        )

    def unregister(self, webhook_id: str) -> bool:
        """Remove a webhook. Returns ``True`` if it existed."""
        removed = self._webhooks.pop(webhook_id, None)
        if removed:
            logger.info("webhook_unregistered", id=webhook_id)
        return removed is not None

    def get_webhook(self, webhook_id: str) -> WebhookConfig | None:
        """Retrieve a webhook by id."""
        return self._webhooks.get(webhook_id)

    def list_webhooks(self, user_id: str | None = None) -> list[WebhookConfig]:
        """Return all webhooks, optionally filtered by user."""
        hooks = list(self._webhooks.values())
        if user_id is not None:
            hooks = [h for h in hooks if h.user_id == user_id]
        return hooks

    def update_webhook(self, webhook_id: str, **kwargs) -> WebhookConfig:
        """Update mutable fields on a webhook.

        Accepted keyword arguments: ``name``, ``url``, ``secret``,
        ``events``, ``is_active``, ``headers``, ``telegram_chat_id``.
        """
        webhook = self._webhooks.get(webhook_id)
        if webhook is None:
            raise KeyError(f"Webhook not found: {webhook_id}")

        allowed = {"name", "url", "secret", "events", "is_active", "headers", "telegram_chat_id"}
        for key, value in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Cannot update field: {key}")
            setattr(webhook, key, value)

        logger.info("webhook_updated", id=webhook_id, fields=list(kwargs.keys()))
        return webhook

    # ------------------------------------------------------------------
    # Dispatching
    # ------------------------------------------------------------------

    async def dispatch(self, event: str, payload: dict) -> list[DeliveryRecord]:
        """Dispatch an event to all matching, active webhooks.

        Each delivery runs as a background task so the caller is not
        blocked.  Returns a list of :class:`DeliveryRecord` objects (one
        per webhook).
        """
        records: list[DeliveryRecord] = []
        for webhook in self._webhooks.values():
            if not webhook.is_active:
                continue
            if webhook.events and event not in webhook.events:
                continue

            record = DeliveryRecord(
                id=f"dlv_{uuid.uuid4().hex[:12]}",
                webhook_id=webhook.id,
                event_type=event,
                status=DeliveryStatus.PENDING,
            )
            records.append(record)
            self._delivery_log.append(record)

            asyncio.create_task(self._send_with_retry(webhook, event, payload, record))

        logger.info("webhook_dispatch", event=event, targets=len(records))
        return records

    async def dispatch_to(self, webhook_id: str, event: str, payload: dict) -> DeliveryRecord:
        """Dispatch an event to a single specific webhook."""
        webhook = self._webhooks.get(webhook_id)
        if webhook is None:
            raise KeyError(f"Webhook not found: {webhook_id}")

        record = DeliveryRecord(
            id=f"dlv_{uuid.uuid4().hex[:12]}",
            webhook_id=webhook.id,
            event_type=event,
            status=DeliveryStatus.PENDING,
        )
        self._delivery_log.append(record)
        await self._send_with_retry(webhook, event, payload, record)
        return record

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _send_with_retry(
        self,
        webhook: WebhookConfig,
        event: str,
        payload: dict,
        record: DeliveryRecord,
    ) -> None:
        """Attempt delivery with exponential backoff and jitter."""
        import random

        for attempt in range(1, self._max_retries + 1):
            record.attempts = attempt
            record.last_attempt_at = datetime.now(timezone.utc)

            try:
                status_code = await self._send(webhook, event, payload)
                record.response_code = status_code

                if 200 <= status_code < 300:
                    record.status = DeliveryStatus.DELIVERED
                    logger.debug(
                        "webhook_delivered",
                        webhook_id=webhook.id,
                        event=event,
                        attempt=attempt,
                        status=status_code,
                    )
                    return

                # 4xx errors (except 429) are not retryable
                if 400 <= status_code < 500 and status_code != 429:
                    record.status = DeliveryStatus.FAILED
                    record.last_error = f"HTTP {status_code}"
                    logger.warning(
                        "webhook_delivery_rejected",
                        webhook_id=webhook.id,
                        event=event,
                        status=status_code,
                    )
                    return

                record.last_error = f"HTTP {status_code}"

            except Exception as exc:
                record.last_error = str(exc)
                logger.warning(
                    "webhook_delivery_error",
                    webhook_id=webhook.id,
                    event=event,
                    attempt=attempt,
                    error=str(exc),
                )

            if attempt < self._max_retries:
                record.status = DeliveryStatus.RETRYING
                backoff = min(
                    self._base_backoff * (2 ** (attempt - 1)),
                    self._max_backoff,
                )
                jitter = random.uniform(0, backoff * 0.25)
                await asyncio.sleep(backoff + jitter)

        record.status = DeliveryStatus.FAILED
        logger.error(
            "webhook_delivery_exhausted",
            webhook_id=webhook.id,
            event=event,
            attempts=self._max_retries,
            last_error=record.last_error,
        )

    async def _send(self, webhook: WebhookConfig, event: str, payload: dict) -> int:
        """Send a single HTTP request and return the status code."""
        if not self._client:
            raise RuntimeError("WebhookDispatcher has not been started")

        body = self._build_body(webhook, event, payload)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(webhook.headers)

        # HMAC signature for custom webhooks
        if webhook.secret and webhook.platform == WebhookPlatform.CUSTOM:
            raw = json.dumps(body, sort_keys=True).encode()
            sig = hmac.new(webhook.secret.encode(), raw, hashlib.sha256).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={sig}"
            headers["X-Webhook-Event"] = event

        response = await self._client.post(webhook.url, json=body, headers=headers)
        return response.status_code

    # ------------------------------------------------------------------
    # Body formatting
    # ------------------------------------------------------------------

    def _build_body(self, webhook: WebhookConfig, event: str, payload: dict) -> dict:
        """Build the HTTP body based on the target platform."""
        timestamp = datetime.now(timezone.utc).isoformat()

        if webhook.platform == WebhookPlatform.DISCORD:
            return _format_discord(event, payload, timestamp)

        if webhook.platform == WebhookPlatform.SLACK:
            return _format_slack(event, payload, timestamp)

        if webhook.platform == WebhookPlatform.TELEGRAM:
            return _format_telegram(event, payload, timestamp, webhook.telegram_chat_id)

        return _format_custom(event, payload, timestamp)

    # ------------------------------------------------------------------
    # Delivery log / observability
    # ------------------------------------------------------------------

    def get_delivery_log(
        self,
        webhook_id: str | None = None,
        status: DeliveryStatus | None = None,
        limit: int = 100,
    ) -> list[DeliveryRecord]:
        """Query the in-memory delivery log with optional filters."""
        records = self._delivery_log
        if webhook_id is not None:
            records = [r for r in records if r.webhook_id == webhook_id]
        if status is not None:
            records = [r for r in records if r.status == status]
        return records[-limit:]

    def get_delivery_stats(self, webhook_id: str | None = None) -> dict:
        """Return aggregate delivery statistics."""
        records = self._delivery_log
        if webhook_id is not None:
            records = [r for r in records if r.webhook_id == webhook_id]

        total = len(records)
        delivered = sum(1 for r in records if r.status == DeliveryStatus.DELIVERED)
        failed = sum(1 for r in records if r.status == DeliveryStatus.FAILED)
        retrying = sum(1 for r in records if r.status == DeliveryStatus.RETRYING)
        pending = sum(1 for r in records if r.status == DeliveryStatus.PENDING)

        return {
            "total": total,
            "delivered": delivered,
            "failed": failed,
            "retrying": retrying,
            "pending": pending,
            "success_rate": round(delivered / total * 100, 1) if total > 0 else 0.0,
        }

    async def test_webhook(self, webhook_id: str) -> DeliveryRecord:
        """Send a test event to verify webhook connectivity."""
        return await self.dispatch_to(
            webhook_id,
            "test",
            {"message": "This is a test notification from TradeMaster.", "status": "ok"},
        )


webhook_dispatcher = WebhookDispatcher()
