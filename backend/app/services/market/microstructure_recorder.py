"""Prospective USD-M event recorder with backpressure and fail-closed gaps."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
import websockets

from app.core.logging import get_logger
from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.market.order_book_rebuilder import OrderBookGapError, OrderBookRebuilder


logger = get_logger(__name__)
FUTURES_STREAM_URL = "wss://fstream.binance.com/stream"
FUTURES_REST_URL = "https://fapi.binance.com"


class EventSink(Protocol):
    async def append_batch(self, events: list[MicrostructureEvent]) -> None: ...


class JsonlEventSink:
    """Append-only daily WAL; fsync makes acknowledged batches crash durable."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = asyncio.Lock()

    async def append_batch(self, events: list[MicrostructureEvent]) -> None:
        if not events:
            return
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for event in events:
            day = event.event_time.astimezone(UTC).date().isoformat()
            grouped[(event.event_type.value, day)].append(event.model_dump_json(exclude_none=True))
        async with self._lock:
            await asyncio.to_thread(self._append_grouped, grouped)

    def _append_grouped(self, grouped: dict[tuple[str, str], list[str]]) -> None:
        for (event_type, day), serialized_events in grouped.items():
            path = self._root / event_type.lower() / f"date={day}" / "events.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = ("\n".join(serialized_events) + "\n").encode()
            with path.open("ab") as raw_output:
                with gzip.GzipFile(
                    fileobj=raw_output, mode="wb", compresslevel=6, mtime=0
                ) as output:
                    output.write(payload)
                raw_output.flush()
                os.fsync(raw_output.fileno())


class StreamSequenceGap(RuntimeError):
    """Raised when the public trade stream skipped a sequence identifier."""


class SequenceTracker:
    def __init__(self) -> None:
        self._last_by_stream: dict[str, int] = {}

    def observe(self, stream: str, sequence_id: int, *, require_contiguous: bool = True) -> bool:
        previous = self._last_by_stream.get(stream)
        if previous is not None:
            if sequence_id <= previous:
                return False
            if require_contiguous and sequence_id != previous + 1:
                raise StreamSequenceGap(
                    f"{stream} sequence gap: received {sequence_id}, expected {previous + 1}"
                )
        self._last_by_stream[stream] = sequence_id
        return True

    def reset(self) -> None:
        self._last_by_stream.clear()


class MicrostructureRecorder:
    """Capture trades, quotes, depth, mark price, and liquidations without orders."""

    def __init__(
        self,
        *,
        sink: EventSink,
        symbol: str = "BTCUSDT",
        rest_base_url: str = FUTURES_REST_URL,
        stream_base_url: str = FUTURES_STREAM_URL,
        batch_size: int = 500,
        queue_size: int = 50_000,
        snapshot_fetcher: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        mark_price_fetcher: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self._sink = sink
        self._symbol = symbol.upper()
        self._rest_base_url = rest_base_url.rstrip("/")
        self._stream_base_url = stream_base_url.rstrip("/")
        self._batch_size = batch_size
        self._queue: asyncio.Queue[MicrostructureEvent] = asyncio.Queue(maxsize=queue_size)
        self._snapshot_fetcher = snapshot_fetcher or self._fetch_depth_snapshot
        self._mark_price_fetcher = mark_price_fetcher or self._fetch_mark_price
        self._sequence_tracker = SequenceTracker()
        self._book = OrderBookRebuilder()
        self._running = False
        self.reconnect_count = 0
        self.gap_count = 0

    @property
    def stream_url(self) -> str:
        symbol = self._symbol.lower()
        streams = "/".join(
            (
                f"{symbol}@trade",
                f"{symbol}@depth@100ms",
                f"{symbol}@forceOrder",
            )
        )
        return f"{self._stream_base_url}?streams={streams}"

    async def run(self, stop_event: asyncio.Event) -> None:
        self._running = True
        writer_stop_event = asyncio.Event()
        writer_task = asyncio.create_task(
            self._writer(writer_stop_event), name="microstructure_writer"
        )
        mark_price_task = asyncio.create_task(
            self._mark_price_poller(stop_event), name="mark_price_poller"
        )

        def stop_on_writer_failure(task: asyncio.Task[None]) -> None:
            if not task.cancelled() and task.exception() is not None:
                stop_event.set()

        writer_task.add_done_callback(stop_on_writer_failure)
        try:
            while not stop_event.is_set():
                try:
                    await self._run_connection(stop_event)
                except asyncio.CancelledError:
                    raise
                except (OrderBookGapError, StreamSequenceGap) as error:
                    self.gap_count += 1
                    self.reconnect_count += 1
                    self._sequence_tracker.reset()
                    logger.warning("microstructure_stream_gap", error=str(error))
                except Exception as error:
                    self.reconnect_count += 1
                    self._sequence_tracker.reset()
                    logger.warning("microstructure_stream_reconnect", error=str(error))
                if not stop_event.is_set():
                    await asyncio.sleep(min(2 ** min(self.reconnect_count, 5), 30))
        finally:
            self._running = False
            stop_event.set()
            await mark_price_task
            writer_stop_event.set()
            await writer_task

    async def _run_connection(self, stop_event: asyncio.Event) -> None:
        buffered_messages: list[dict[str, Any]] = []
        async with websockets.connect(
            self.stream_url,
            open_timeout=20,
            ping_interval=20,
            ping_timeout=20,
            max_queue=10_000,
        ) as websocket:
            snapshot_task = asyncio.create_task(self._snapshot_fetcher())
            while not snapshot_task.done() and not stop_event.is_set():
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    buffered_messages.append(_decode_message(raw_message))
                except asyncio.TimeoutError:
                    continue
            snapshot = await snapshot_task
            self._book.initialize(snapshot)
            for message in buffered_messages:
                await self._handle_message(message)
            self.reconnect_count = 0

            while not stop_event.is_set():
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=60)
                await self._handle_message(_decode_message(raw_message))

    async def _handle_message(self, wrapper: dict[str, Any]) -> None:
        stream = str(wrapper.get("stream", ""))
        payload = wrapper.get("data")
        if not stream or not isinstance(payload, dict):
            return
        try:
            event = parse_market_message(
                stream=stream,
                payload=payload,
                receive_time=datetime.now(UTC),
                symbol=self._symbol,
            )
        except Exception as error:
            raise ValueError(
                f"Invalid market event stream={stream} event={payload.get('e')} "
                f"price={payload.get('p')} order={payload.get('o')}"
            ) from error
        if event is None:
            return
        if (
            event.event_type
            in {
                MarketEventType.AGG_TRADE,
                MarketEventType.TRADE,
            }
            and event.sequence_id is not None
        ):
            if not self._sequence_tracker.observe(
                stream,
                event.sequence_id,
                require_contiguous=event.event_type == MarketEventType.AGG_TRADE,
            ):
                return
        if event.event_type == MarketEventType.DEPTH:
            top = self._book.apply(payload)
            if top is None:
                return
            event.bid_price = float(top.bid_price)
            event.bid_quantity = float(top.bid_quantity)
            event.ask_price = float(top.ask_price)
            event.ask_quantity = float(top.ask_quantity)
        await self._queue.put(event)

    async def _writer(self, writer_stop_event: asyncio.Event) -> None:
        batch: list[MicrostructureEvent] = []
        loop = asyncio.get_running_loop()
        flush_deadline = loop.time() + 1
        while not writer_stop_event.is_set() or not self._queue.empty():
            try:
                timeout = max(0.01, flush_deadline - loop.time())
                event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                batch.append(event)
            except asyncio.TimeoutError:
                pass
            if len(batch) >= self._batch_size or (
                batch
                and (
                    loop.time() >= flush_deadline
                    or (writer_stop_event.is_set() and self._queue.empty())
                )
            ):
                await self._sink.append_batch(batch)
                for _ in batch:
                    self._queue.task_done()
                batch = []
                flush_deadline = loop.time() + 1
        if batch:
            await self._sink.append_batch(batch)
            for _ in batch:
                self._queue.task_done()

    async def _fetch_depth_snapshot(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{self._rest_base_url}/fapi/v1/depth",
                params={"symbol": self._symbol, "limit": 1000},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise OrderBookGapError("Depth snapshot response is not an object")
        return payload

    async def _fetch_mark_price(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{self._rest_base_url}/fapi/v1/premiumIndex",
                params={"symbol": self._symbol},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Mark price response is not an object")
        return payload

    async def _mark_price_poller(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                payload = await self._mark_price_fetcher()
                receive_time = datetime.now(UTC)
                event_time = datetime.fromtimestamp(
                    int(payload.get("time", int(receive_time.timestamp() * 1000))) / 1000,
                    tz=UTC,
                )
                await self._queue.put(
                    MicrostructureEvent(
                        product="usdm_perpetual",
                        symbol=self._symbol,
                        event_type=MarketEventType.MARK_PRICE,
                        event_time=event_time,
                        receive_time=receive_time,
                        price=float(payload["markPrice"]),
                        payload={
                            "index_price": float(payload["indexPrice"]),
                            "estimated_settle_price": float(payload["estimatedSettlePrice"]),
                            "funding_rate": float(payload["lastFundingRate"]),
                            "next_funding_time": int(payload["nextFundingTime"]),
                            "source": "rest_premium_index",
                        },
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("mark_price_poll_failed", error=str(error))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass


def parse_market_message(
    *,
    stream: str,
    payload: dict[str, Any],
    receive_time: datetime,
    symbol: str,
) -> MicrostructureEvent | None:
    event_name = payload.get("e")
    event_time = _event_time(payload)
    base = {
        "product": "usdm_perpetual",
        "symbol": symbol.upper(),
        "event_time": event_time,
        "receive_time": receive_time,
    }
    if event_name == "aggTrade":
        is_buyer_maker = bool(payload["m"])
        return MicrostructureEvent(
            **base,
            event_type=MarketEventType.AGG_TRADE,
            sequence_id=int(payload["a"]),
            first_sequence_id=int(payload["f"]),
            last_sequence_id=int(payload["l"]),
            price=float(payload["p"]),
            quantity=float(payload["q"]),
            quote_quantity=float(payload["p"]) * float(payload["q"]),
            is_buyer_maker=is_buyer_maker,
            side="SELL" if is_buyer_maker else "BUY",
        )
    if event_name == "trade":
        if float(payload["p"]) <= 0 or float(payload["q"]) <= 0:
            return None
        is_buyer_maker = bool(payload["m"])
        return MicrostructureEvent(
            **base,
            event_type=MarketEventType.TRADE,
            sequence_id=int(payload["t"]),
            price=float(payload["p"]),
            quantity=float(payload["q"]),
            quote_quantity=float(payload["p"]) * float(payload["q"]),
            is_buyer_maker=is_buyer_maker,
            side="SELL" if is_buyer_maker else "BUY",
            payload={"order_type": payload.get("X")},
        )
    if event_name == "bookTicker" or stream.endswith("@bookTicker"):
        return MicrostructureEvent(
            **base,
            event_type=MarketEventType.BOOK_TICKER,
            sequence_id=int(payload["u"]),
            bid_price=float(payload["b"]),
            bid_quantity=float(payload["B"]),
            ask_price=float(payload["a"]),
            ask_quantity=float(payload["A"]),
        )
    if event_name == "depthUpdate":
        return MicrostructureEvent(
            **base,
            event_type=MarketEventType.DEPTH,
            sequence_id=int(payload["u"]),
            first_sequence_id=int(payload["U"]),
            last_sequence_id=int(payload["u"]),
            payload={
                "previous_final_update_id": int(payload["pu"]),
                "bids": payload.get("b", []),
                "asks": payload.get("a", []),
            },
        )
    if event_name == "markPriceUpdate":
        return MicrostructureEvent(
            **base,
            event_type=MarketEventType.MARK_PRICE,
            price=float(payload["p"]),
            payload={
                "index_price": float(payload["i"]),
                "estimated_settle_price": float(payload["P"]),
                "funding_rate": float(payload["r"]),
                "next_funding_time": int(payload["T"]),
            },
        )
    if event_name == "forceOrder":
        order = payload["o"]
        liquidation_base = {
            **base,
            "event_time": datetime.fromtimestamp(int(order["T"]) / 1000, tz=UTC),
        }
        return MicrostructureEvent(
            **liquidation_base,
            event_type=MarketEventType.LIQUIDATION,
            price=_first_positive(order.get("ap"), order.get("p"), order.get("sp")),
            quantity=max(float(order.get("z", 0)), float(order["q"])),
            side=str(order["S"]),
            payload={"status": order.get("X"), "order_type": order.get("o")},
        )
    return None


def _event_time(payload: dict[str, Any]) -> datetime:
    timestamp = payload.get("T", payload.get("E"))
    if timestamp is None:
        raise ValueError("Market event has no exchange timestamp")
    return datetime.fromtimestamp(int(timestamp) / 1000, tz=UTC)


def _first_positive(*values: Any) -> float | None:
    for value in values:
        parsed = float(value or 0)
        if parsed > 0:
            return parsed
    return None


def _decode_message(raw_message: str | bytes) -> dict[str, Any]:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    decoded = json.loads(raw_message)
    if not isinstance(decoded, dict):
        raise ValueError("WebSocket message is not an object")
    return decoded
