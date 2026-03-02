"""Real-time streaming pipeline for data ingestion.

Provides a Kafka-like interface using Redis Streams for:
- High-throughput data ingestion from exchanges
- Stream processing with consumer groups
- At-least-once delivery semantics
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.events import event_bus
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamMessage:
    stream: str
    id: str
    data: dict
    timestamp: datetime


class StreamProducer:
    """Produces messages to Redis Streams."""

    async def produce(self, stream: str, data: dict) -> str | None:
        """Publish a message to a stream. Returns the message ID."""
        try:
            if event_bus._redis:
                msg_id = await event_bus._redis.xadd(
                    stream,
                    {"data": json.dumps(data), "ts": datetime.now(timezone.utc).isoformat()},
                    maxlen=100000,  # Keep last 100k messages
                )
                return msg_id
        except Exception as e:
            logger.warning("stream_produce_failed", stream=stream, error=str(e))
        return None


class StreamConsumer:
    """Consumes messages from Redis Streams with consumer groups."""

    def __init__(self, group: str, consumer: str):
        self._group = group
        self._consumer = consumer
        self._running = False

    async def start(self, streams: list[str], handler, batch_size: int = 10):
        """Start consuming from streams."""
        self._running = True

        # Create consumer groups if needed
        for stream in streams:
            try:
                if event_bus._redis:
                    await event_bus._redis.xgroup_create(stream, self._group, id="0", mkstream=True)
            except Exception:
                pass  # Group may already exist

        logger.info("stream_consumer_started", group=self._group, consumer=self._consumer, streams=streams)

        while self._running:
            try:
                if not event_bus._redis:
                    await asyncio.sleep(1)
                    continue

                # Read from all streams
                stream_keys = {s: ">" for s in streams}
                messages = await event_bus._redis.xreadgroup(
                    self._group, self._consumer, stream_keys, count=batch_size, block=1000,
                )

                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        try:
                            data = json.loads(msg_data.get(b"data", b"{}"))
                            await handler(StreamMessage(
                                stream=stream_name.decode() if isinstance(stream_name, bytes) else stream_name,
                                id=msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                                data=data,
                                timestamp=datetime.now(timezone.utc),
                            ))
                            # Acknowledge message
                            await event_bus._redis.xack(stream_name, self._group, msg_id)
                        except Exception as e:
                            logger.warning("stream_message_failed", stream=stream_name, error=str(e))
            except Exception as e:
                if self._running:
                    logger.warning("stream_consumer_error", error=str(e))
                    await asyncio.sleep(1)

    async def stop(self):
        self._running = False
        logger.info("stream_consumer_stopped", group=self._group, consumer=self._consumer)


class StreamPipeline:
    """Composable stream processing pipeline."""

    def __init__(self):
        self._producer = StreamProducer()
        self._transformations: list = []

    def add_transformation(self, name: str, fn):
        self._transformations.append({"name": name, "fn": fn})
        return self

    async def process(self, stream: str, data: dict) -> dict:
        """Process data through the pipeline."""
        result = data
        for transform in self._transformations:
            try:
                result = await transform["fn"](result) if asyncio.iscoroutinefunction(transform["fn"]) else transform["fn"](result)
            except Exception as e:
                logger.warning("pipeline_transform_failed", transform=transform["name"], error=str(e))
                break
        return result


stream_producer = StreamProducer()
