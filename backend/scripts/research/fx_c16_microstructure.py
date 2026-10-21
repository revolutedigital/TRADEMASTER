"""Prospective, read-only cTrader Level II capture and C13 signal alignment.

Examples (run from backend/):
    python -m scripts.research.fx_c16_microstructure record --environment demo
    python -m scripts.research.fx_c16_microstructure materialize --run-id RUN_ID
    python -m scripts.research.fx_c16_microstructure join-c13 --run-id RUN_ID

This script has no order-submission path. cTrader depth is broker-provided OTC FX depth, not a
consolidated global order book, and the API does not provide historical depth for backtesting.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.fx.instruments import Instrument
from app.fx.microstructure import CTraderBookReplay, PRICE_SCALE, normalize_symbol_name
from app.fx.runner.ctrader.client import (
    DEMO_HOST,
    LIVE_HOST,
    PORT,
    Credentials,
    CTraderClient,
)
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from scripts.research.ctrader_accounts import load_environment
from scripts.research.fx_demo_runner import ACCOUNT, CTID, IMAGE, PASSWORD_FILE

DEFAULT_ROOT = BACKEND_ROOT / "data" / "c16_ctrader_depth"
DEFAULT_RESEARCH_SYMBOLS = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    "EURJPY", "GBPJPY", "EURGBP", "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
    "EURAUD", "EURCAD", "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "CADCHF",
    "SPXUSD", "NSXUSD", "UDXUSD", "GRXEUR", "UKXGBP", "JPXJPY", "FRXEUR", "AUXAUD", "HKXHKD",
)
RESEARCH_TO_CTRADER_SYMBOL = {
    "SPXUSD": "US500",
    "NSXUSD": "NAS100",
    "UDXUSD": "USDX",
    "GRXEUR": "GER40",
    "UKXGBP": "UK100",
    "JPXJPY": "JPN225",
    "FRXEUR": "FRA40",
    "AUXAUD": "AUS200",
    "HKXHKD": "HK50",
}
CTRADER_TO_RESEARCH_SYMBOL = {value: key for key, value in RESEARCH_TO_CTRADER_SYMBOL.items()}
DEFAULT_CAPTURE_SYMBOLS = tuple(RESEARCH_TO_CTRADER_SYMBOL.get(symbol, symbol) for symbol in DEFAULT_RESEARCH_SYMBOLS)
DEFAULT_SYMBOLS = DEFAULT_RESEARCH_SYMBOLS
DEFAULT_C13_JOURNAL = BACKEND_ROOT / "data" / "c13" / "demo_journal.jsonl"
QUEUE_CAPACITY = 20_000
BATCH_SIZE = 500
SNAPSHOT_INTERVAL_MS = 1_000
MAX_JOIN_STALENESS_MS = 1_000
CLI_CBOT_PROJECT = BACKEND_ROOT / "scripts" / "research" / "ctrader_cbots" / "C16DepthProbe"
CLI_CBOT_ROBOT_ROOT = CLI_CBOT_PROJECT.parent
CLI_CBOT_ALGO = CLI_CBOT_ROBOT_ROOT / "C16DepthProbe.algo"
CLI_CBOT_CONTAINER_NAME = "ctrader-demo-C16DEPTH"
CLI_CBOT_MOUNT_PATH = "/mnt/C16DepthProbe.algo"
CLI_PASSWORD_MOUNT_PATH = "/run/ctid.pwd"
C16_MARKER = "C16DOM|"
C16_SPOT_MARKER = "C16SPOT|"


class RecorderError(RuntimeError):
    """A market-data capture or replay could not continue without risking corrupt evidence."""


class GzipEventSink:
    """Append-only daily event WAL, partitioned by run and normalized symbol."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.run_id = run_id

    async def append_batch(self, records: list[dict[str, Any]]) -> None:
        if records:
            await asyncio.to_thread(self._append_batch, records)

    def _append_batch(self, records: list[dict[str, Any]]) -> None:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for record in records:
            day = datetime.fromtimestamp(record["received_at_ms"] / 1000, UTC).date().isoformat()
            grouped[(record["symbol"], day)].append(json.dumps(record, separators=(",", ":")))
        for (symbol, day), rows in grouped.items():
            path = (
                self.root / f"run={self.run_id}" / f"symbol={symbol}"
                / f"date={day}" / "events.jsonl.gz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as raw_output:
                with gzip.GzipFile(fileobj=raw_output, mode="wb", compresslevel=6, mtime=0) as output:
                    output.write(("\n".join(rows) + "\n").encode("utf-8"))
                raw_output.flush()
                os.fsync(raw_output.fileno())


async def _write_queue(
    queue: asyncio.Queue[dict[str, Any] | None],
    sink: GzipEventSink,
    stop_event: asyncio.Event,
    failure: list[BaseException],
) -> None:
    try:
        finishing = False
        while not finishing:
            first = await queue.get()
            if first is None:
                break
            batch = [first]
            while len(batch) < BATCH_SIZE:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    finishing = True
                    break
                batch.append(item)
            await sink.append_batch(batch)
    except BaseException as error:
        failure.append(error)
        stop_event.set()
        if not isinstance(error, asyncio.CancelledError):
            raise


def _record_from_message(
    message: Any,
    *,
    symbol: str,
    run_id: str,
    environment: str,
    connection_generation: int,
    event_sequence: int,
) -> dict[str, Any]:
    received_at_ms = time.time_ns() // 1_000_000
    received_monotonic_ns = time.monotonic_ns()
    common = {
        "schema_version": 1,
        "run_id": run_id,
        "environment": environment,
        "connection_generation": connection_generation,
        "event_sequence": event_sequence,
        "symbol": symbol,
        "symbol_id": int(message.symbolId),
        "received_at_ms": received_at_ms,
        "received_monotonic_ns": received_monotonic_ns,
    }
    if isinstance(message, messages.ProtoOASpotEvent):
        return {
            **common,
            "event_type": "spot",
            "event_timestamp_ms": int(message.timestamp) if message.HasField("timestamp") else None,
            "bid_raw": int(message.bid) if message.HasField("bid") else None,
            "ask_raw": int(message.ask) if message.HasField("ask") else None,
        }
    if isinstance(message, messages.ProtoOADepthEvent):
        quotes = []
        for quote in message.newQuotes:
            if quote.HasField("bid") == quote.HasField("ask"):
                side, price_raw = "invalid", 0
            elif quote.HasField("bid"):
                side, price_raw = "bid", int(quote.bid)
            else:
                side, price_raw = "ask", int(quote.ask)
            quotes.append({
                "id": int(quote.id),
                "side": side,
                "price_raw": price_raw,
                "size_raw": int(quote.size),
            })
        return {
            **common,
            "event_type": "depth_delta",
            "new_quotes": quotes,
            "deleted_quote_ids": [int(quote_id) for quote_id in message.deletedQuotes],
        }
    raise TypeError(f"unsupported cTrader market-data message: {type(message).__name__}")


async def record(args: argparse.Namespace) -> None:
    environment = load_environment()
    required = (
        "CTRADER_CLIENT_ID",
        "CTRADER_CLIENT_SECRET",
        "CTRADER_ACCESS_TOKEN",
        "CTRADER_ACCOUNT_ID",
    )
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise RecorderError(f"missing cTrader environment values: {', '.join(missing)}")
    try:
        account_id = int(environment["CTRADER_ACCOUNT_ID"])
    except ValueError as error:
        raise RecorderError("CTRADER_ACCOUNT_ID must be numeric") from error

    credentials = Credentials(
        environment["CTRADER_CLIENT_ID"],
        environment["CTRADER_CLIENT_SECRET"],
        environment["CTRADER_ACCESS_TOKEN"],
        account_id,
    )
    host = LIVE_HOST if args.environment == "live" else DEMO_HOST
    requested_symbols = tuple(normalize_symbol_name(item) for item in args.symbols.split(",") if item.strip())
    if not requested_symbols or any(len(symbol) != 6 or not symbol.isalpha() for symbol in requested_symbols):
        raise RecorderError("--symbols must be a comma-separated list of six-letter FX pairs")
    if len(set(requested_symbols)) != len(requested_symbols):
        raise RecorderError("--symbols contains duplicates")

    run_id = uuid.uuid4().hex
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=QUEUE_CAPACITY)
    stop_event = asyncio.Event()
    failure: list[BaseException] = []
    names_by_id: dict[int, str] = {}
    event_sequence = 0
    sink = GzipEventSink(args.root, run_id)
    client = CTraderClient(credentials, host=host, port=PORT)

    def on_message(message: Any, _client_msg_id: str | None) -> None:
        nonlocal event_sequence
        if not isinstance(message, (messages.ProtoOASpotEvent, messages.ProtoOADepthEvent)):
            return
        symbol = names_by_id.get(int(message.symbolId))
        if symbol is None:
            return
        event_sequence += 1
        try:
            queue.put_nowait(_record_from_message(
                message,
                symbol=symbol,
                run_id=run_id,
                environment=args.environment,
                connection_generation=client.connection_generation,
                event_sequence=event_sequence,
            ))
        except asyncio.QueueFull:
            failure.append(RecorderError("WAL queue overflow; capture stopped to avoid silent data loss"))
            stop_event.set()

    client.add_listener(on_message)
    writer_task = asyncio.create_task(_write_queue(queue, sink, stop_event, failure))
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await client.start(timeout=30)
        listing = await client.request(
            messages.ProtoOASymbolsListReq(ctidTraderAccountId=client.account_id),
            messages.ProtoOASymbolsListRes,
        )
        ids_by_name = {
            normalize_symbol_name(item.symbolName): int(item.symbolId)
            for item in listing.symbol
        }
        missing_symbols = [symbol for symbol in requested_symbols if symbol not in ids_by_name]
        if missing_symbols:
            raise RecorderError(f"account does not provide symbols: {', '.join(missing_symbols)}")
        names_by_id = {ids_by_name[symbol]: symbol for symbol in requested_symbols}
        symbol_ids = [ids_by_name[symbol] for symbol in requested_symbols]
        await client.subscribe_spots(symbol_ids)
        await client.subscribe_depth(symbol_ids)
        sys.stdout.write(
            f"C16 capture started: run_id={run_id} environment={args.environment} "
            f"symbols={','.join(requested_symbols)} orders_sent=0 root={args.root}\n"
        )
        if args.duration_seconds is None:
            await stop_event.wait()
        else:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.duration_seconds)
            except TimeoutError:
                stop_event.set()
    finally:
        await client.stop()
        if not writer_task.done():
            await queue.put(None)
        await asyncio.gather(writer_task, return_exceptions=True)
    if failure:
        raise RecorderError(str(failure[0])) from failure[0]
    sys.stdout.write(f"C16 capture stopped: run_id={run_id} events={event_sequence}\n")


def _build_cli_cbot() -> None:
    command = [
        "docker", "run", "--rm", "-v", f"{CLI_CBOT_ROBOT_ROOT}:/root/cAlgo/Sources/Robots",
        IMAGE, "build", "/root/cAlgo/Sources/Robots/C16DepthProbe",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RecorderError(f"could not build the C16 cBot with cTrader CLI: {error}") from error
    output = result.stdout + result.stderr
    try:
        payload_start = output.index("{")
        build_result, _ = json.JSONDecoder().raw_decode(output[payload_start:])
    except (ValueError, json.JSONDecodeError) as error:
        raise RecorderError("cTrader CLI build returned no parseable result") from error
    if result.returncode != 0 or build_result.get("success") is not True or not CLI_CBOT_ALGO.is_file():
        errors = build_result.get("errors", [])
        details = "; ".join(
            f"{item.get('code')}: {item.get('text')}" for item in errors[:8]
        ) or "no diagnostic returned"
        raise RecorderError(f"C16 cBot build failed: {details}")


def _assert_c16_cli_ready() -> None:
    if not PASSWORD_FILE.is_file() or not os.access(PASSWORD_FILE, os.R_OK):
        raise RecorderError("private cTrader demo password file is unavailable to the existing CLI workflow")
    try:
        container_result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],  # noqa: S607
            capture_output=True, text=True, timeout=15, check=False,
        )  # noqa: S603,S607
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RecorderError("Docker is unavailable for the cTrader CLI recorder") from error
    if container_result.returncode != 0:
        raise RecorderError("Docker could not list containers; refusing to start a duplicate CLI session")
    if CLI_CBOT_CONTAINER_NAME in container_result.stdout.splitlines():
        raise RecorderError(
            f"container {CLI_CBOT_CONTAINER_NAME} already exists; refusing to disturb an existing session"
        )


async def record_cli(args: argparse.Namespace) -> None:
    requested_symbols = _parse_symbols(args.symbols)
    symbols_to_store = {symbol: CTRADER_TO_RESEARCH_SYMBOL.get(symbol, symbol) for symbol in requested_symbols}
    expected_stored_symbols = set(symbols_to_store.values())
    _assert_c16_cli_ready()
    _build_cli_cbot()

    run_id = uuid.uuid4().hex
    sink = GzipEventSink(args.root, run_id)
    stop_event = asyncio.Event()
    cbot_started = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop_event.set)
        except NotImplementedError:
            pass

    command = [
        "docker", "run", "--rm", "-i", "--name", CLI_CBOT_CONTAINER_NAME,
        "--log-opt", "max-size=8m", "--log-opt", "max-file=2",
        "-v", f"{PASSWORD_FILE}:{CLI_PASSWORD_MOUNT_PATH}:ro",
        "-v", f"{CLI_CBOT_ALGO}:{CLI_CBOT_MOUNT_PATH}:ro",
        IMAGE, "run", f"--ctid={CTID}", f"--pwd-file={CLI_PASSWORD_MOUNT_PATH}",
        f"--account={ACCOUNT}", CLI_CBOT_MOUNT_PATH,
        f"--Pairs={','.join(requested_symbols)}", "--MaxLevels=20",
        "--symbol=EURUSD", "--period=m1", "--exit-on-stop",
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        limit=1_048_576,
    )
    event_count = 0
    spot_tick_count = 0
    depth_snapshot_count = 0
    symbols_seen: set[str] = set()
    symbols_with_populated_depth: set[str] = set()
    pending: list[dict[str, Any]] = []
    failures: list[BaseException] = []
    last_flush = time.monotonic()

    async def flush_pending() -> None:
        nonlocal last_flush
        if pending:
            await sink.append_batch(pending.copy())
            pending.clear()
        last_flush = time.monotonic()

    async def consume_cli_output() -> None:
        nonlocal event_count, spot_tick_count, depth_snapshot_count, last_flush
        assert process.stdout is not None
        try:
            while line := await process.stdout.readline():
                text = line.decode("utf-8", errors="replace")
                if "C16DOM_STATUS|disconnected|" in text:
                    failures.append(RecorderError("cTrader server disconnected; this capture was stopped fail-closed"))
                    stop_event.set()
                    break
                marker = next((value for value in (C16_MARKER, C16_SPOT_MARKER) if value in text), None)
                if marker is None:
                    if "C16DOM_STATUS|started|" in text:
                        cbot_started.set()
                        sys.stdout.write("C16 CLI cBot connected; capturing broker DOM snapshots (read-only).\n")
                        sys.stdout.flush()
                    continue
                try:
                    marker_index = text.find(marker)
                    sample = json.loads(text[marker_index + len(marker):])
                    broker_symbol = normalize_symbol_name(str(sample["symbol"]))
                    if broker_symbol not in symbols_to_store:
                        raise RecorderError(f"cBot returned unexpected symbol {broker_symbol}")
                    symbol = symbols_to_store[broker_symbol]
                    is_spot = marker == C16_SPOT_MARKER
                    spot_bid = float(sample["bid"] if is_spot else sample["spot_bid"])
                    spot_ask = float(sample["ask"] if is_spot else sample["spot_ask"])
                    if not all(map(math.isfinite, (spot_bid, spot_ask))) or spot_bid <= 0 or spot_ask < spot_bid:
                        raise RecorderError(f"cBot returned invalid bid/ask for {symbol}")
                    received_at_ms = time.time_ns() // 1_000_000
                    event_count += 1
                    if is_spot:
                        spot_tick_count += 1
                    else:
                        depth_snapshot_count += 1
                    event: dict[str, Any] = {
                        "schema_version": 1,
                        "run_id": run_id,
                        "environment": "demo",
                        "market_data_source": "ctrader_cli_cbot_spot" if is_spot else "ctrader_cli_cbot_marketdepth",
                        "connection_generation": 1,
                        "event_sequence": event_count,
                        "source_event_sequence": int(sample["sequence"]),
                        "source_timestamp_ms": int(sample["source_timestamp_ms"]),
                        "symbol": symbol,
                        "received_at_ms": received_at_ms,
                        "received_monotonic_ns": time.monotonic_ns(),
                        "spot_bid_raw": round(spot_bid * PRICE_SCALE),
                        "spot_ask_raw": round(spot_ask * PRICE_SCALE),
                        "event_type": "spot" if is_spot else "market_snapshot",
                    }
                    if not is_spot:
                        levels: dict[str, list[dict[str, int]]] = {"bid_levels": [], "ask_levels": []}
                        for side, key in (("bid", "bid_levels"), ("ask", "ask_levels")):
                            for level in sample["bids" if side == "bid" else "asks"]:
                                price = float(level["price"])
                                volume = float(level["volume_in_units"])
                                if not math.isfinite(price) or not math.isfinite(volume):
                                    raise RecorderError(f"cBot returned non-finite {side} depth for {symbol}")
                                levels[key].append({
                                    "price_raw": round(price * PRICE_SCALE),
                                    "size_raw": round(volume * 100),
                                })
                        if levels["bid_levels"] and levels["ask_levels"]:
                            symbols_with_populated_depth.add(symbol)
                        event.update(levels)
                    pending.append(event)
                    symbols_seen.add(symbol)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError, RecorderError) as error:
                    failures.append(error)
                    stop_event.set()
                    break
                if len(pending) >= BATCH_SIZE or time.monotonic() - last_flush >= 1.0:
                    await flush_pending()
        except (OSError, asyncio.IncompleteReadError) as error:
            failures.append(RecorderError(f"failed while reading the cTrader CLI stream: {error}"))
            stop_event.set()
        finally:
            try:
                await flush_pending()
            except BaseException as error:
                failures.append(error)
                stop_event.set()
            stop_event.set()

    reader_task = asyncio.create_task(consume_cli_output())
    sys.stdout.write(
        f"C16 CLI capture starting: run_id={run_id} environment=demo "
        f"symbols={','.join(requested_symbols)} orders_sent=0\n"
    )
    sys.stdout.flush()
    try:
        if args.duration_seconds is None:
            await stop_event.wait()
        else:
            start_waiter = asyncio.create_task(cbot_started.wait())
            stop_waiter = asyncio.create_task(stop_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {start_waiter, stop_waiter}, timeout=120, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    raise RecorderError("cTrader CLI cBot did not start within 120 seconds")
                if cbot_started.is_set() and not stop_event.is_set():
                    await asyncio.wait_for(stop_event.wait(), timeout=args.duration_seconds)
            except TimeoutError:
                pass
            finally:
                start_waiter.cancel()
                stop_waiter.cancel()
                await asyncio.gather(start_waiter, stop_waiter, return_exceptions=True)
    finally:
        stop_event.set()
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "stop", "--time", "10", CLI_CBOT_CONTAINER_NAME],
            capture_output=True, text=True, timeout=20, check=False,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=15)
        except TimeoutError:
            process.kill()
            await process.wait()
        await reader_task

    if failures:
        raise RecorderError(str(failures[0])) from failures[0]
    if depth_snapshot_count == 0 or symbols_with_populated_depth != expected_stored_symbols:
        missing = sorted(expected_stored_symbols - symbols_with_populated_depth)
        raise RecorderError(
            "cTrader CLI did not deliver populated DOM for every requested symbol; "
            f"missing={','.join(missing)} depth_snapshots={depth_snapshot_count}"
        )
    sys.stdout.write(
        f"C16 CLI capture stopped: run_id={run_id} events={event_count} "
        f"spot_ticks={spot_tick_count} dom_snapshots={depth_snapshot_count} "
        f"symbols_seen={','.join(sorted(symbols_seen))} orders_sent=0\n"
    )


def _event_files(root: Path, run_id: str, symbol: str) -> list[Path]:
    directory = root / f"run={run_id}" / f"symbol={normalize_symbol_name(symbol)}"
    return sorted(directory.glob("date=*/events.jsonl.gz"))


def _read_event_file(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RecorderError(f"corrupt WAL at {path}:{line_number}") from error


def _write_feature_snapshots(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for symbol in args.symbols:
        paths = _event_files(args.root, args.run_id, symbol)
        if not paths:
            counts[symbol] = {"snapshots": 0, "generations": []}
            continue
        replay = CTraderBookReplay(
            symbol=symbol,
            pip_size=Instrument.from_symbol(symbol).pip_size,
            verified_generations=args.verified_generations,
            max_staleness_ms=args.max_staleness_ms,
        )
        output = args.output_dir / f"run={args.run_id}" / f"symbol={symbol}" / "features.jsonl.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise RecorderError(f"refusing to overwrite existing feature file: {output}")
        temporary = output.with_name(f"{output.name}.tmp")
        written = 0
        generations: set[int] = set()
        last_sample_second: int | None = None
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as target:
                for path in paths:
                    for event in _read_event_file(path):
                        snapshot = replay.observe(event)
                        generations.add(int(snapshot["connection_generation"]))
                        sample_second = int(snapshot["received_at_ms"]) // SNAPSHOT_INTERVAL_MS
                        if sample_second == last_sample_second:
                            continue
                        last_sample_second = sample_second
                        target.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
                        written += 1
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        counts[symbol] = {"snapshots": written, "generations": sorted(generations)}
    return counts


def _load_c13_signals(path: Path, symbols: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise RecorderError(f"C13 journal not found: {path}")
    signals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RecorderError(f"corrupt C13 journal at {path}:{line_number}") from error
            if event.get("event") != "signal":
                continue
            symbol = normalize_symbol_name(str(event.get("symbol", "")))
            if symbol not in symbols or not isinstance(event.get("at"), (int, float)):
                continue
            signal_row = dict(event)
            signal_row["symbol"] = symbol
            signal_row["signal_time_ms"] = round(float(event["at"]) * 1000)
            signals[symbol].append(signal_row)
    for rows in signals.values():
        rows.sort(key=lambda item: (item["signal_time_ms"], str(item.get("key", ""))))
    return signals


def _join_symbol_signals(
    symbol: str,
    signals: list[dict[str, Any]],
    feature_path: Path,
    *,
    max_staleness_ms: int,
):
    if not feature_path.is_file():
        for signal_row in signals:
            yield _joined_row(signal_row, None, max_staleness_ms)
        return
    with gzip.open(feature_path, "rt", encoding="utf-8") as source:
        feature_iterator = (json.loads(line) for line in source)
        next_feature = next(feature_iterator, None)
        latest: dict[str, Any] | None = None
        for signal_row in signals:
            signal_time_ms = signal_row["signal_time_ms"]
            while next_feature is not None and int(next_feature["received_at_ms"]) <= signal_time_ms:
                latest = next_feature
                next_feature = next(feature_iterator, None)
            yield _joined_row(signal_row, latest, max_staleness_ms)


def _joined_row(
    signal_row: dict[str, Any],
    feature_row: dict[str, Any] | None,
    max_staleness_ms: int,
) -> dict[str, Any]:
    signal_time_ms = signal_row["signal_time_ms"]
    age_ms = (
        signal_time_ms - int(feature_row["received_at_ms"])
        if feature_row is not None else None
    )
    verified = bool(
        feature_row is not None
        and feature_row.get("book_snapshot_verified")
    )
    row = {
        "schema_version": 1,
        "symbol": signal_row["symbol"],
        "signal_time_ms": signal_time_ms,
        "signal_key": signal_row.get("key"),
        "set_id": signal_row.get("set_id"),
        "side": signal_row.get("side"),
        "c13_prediction": signal_row.get("prediction"),
        "c13_threshold": signal_row.get("threshold"),
        "c13_take": signal_row.get("take"),
        "c13_stale": signal_row.get("stale"),
        "c13_rollover_long": signal_row.get("rollover_long"),
        "c13_delay_seconds": signal_row.get("delay"),
        "c13_model_sha": signal_row.get("model_sha"),
        "feed_environment": feature_row.get("environment") if feature_row else None,
        "feature_snapshot_at_ms": feature_row.get("received_at_ms") if feature_row else None,
        "feature_age_ms": age_ms,
        "book_snapshot_verified": verified,
        "book_eligible": bool(
            feature_row is not None
            and verified
            and feature_row.get("book_eligible")
            and age_ms is not None
            and 0 <= age_ms <= max_staleness_ms
        ),
    }
    if feature_row is not None:
        for key, value in feature_row.items():
            if key not in row and key not in {"schema_version", "event_sequence", "source_event_type"}:
                row[f"book_{key}"] = value
        side = signal_row.get("side")
        direction = float(side) if side in (-1, 1) else None
        if direction is not None:
            for feature_name in (
                "depth_imbalance_1",
                "depth_imbalance_5",
                "depth_imbalance_10",
                "microprice_displacement_pips",
                "quote_add_cancel_imbalance_5s",
            ):
                value = feature_row.get(feature_name)
                row[f"directed_{feature_name}"] = (
                    float(value) * direction if value is not None else None
                )
    return row


def _write_c13_join(args: argparse.Namespace) -> tuple[int, int]:
    signals = _load_c13_signals(args.journal, {normalize_symbol_name(s) for s in args.symbols})
    if args.output.exists():
        raise RecorderError(f"refusing to overwrite existing joined dataset: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    counts = {"signals": 0, "eligible": 0}
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as target:
            for symbol in args.symbols:
                feature_path = (
                    args.features_dir / f"run={args.run_id}" / f"symbol={symbol}" / "features.jsonl.gz"
                )
                for row in _join_symbol_signals(
                    symbol,
                    signals.get(symbol, []),
                    feature_path,
                    max_staleness_ms=args.max_staleness_ms,
                ):
                    target.write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts["signals"] += 1
                    counts["eligible"] += int(row["book_eligible"])
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return counts["signals"], counts["eligible"]


def _spot_quote_path(
    root: Path,
    run_id: str,
    symbol: str,
    signal_date: date,
    *,
    max_staleness_ms: int,
) -> list[tuple[int, float, float]]:
    """Return complete, fresh bid/ask states spanning a UTC day and its label horizon."""
    bid: int | None = None
    ask: int | None = None
    bid_at_ms: int | None = None
    ask_at_ms: int | None = None
    generation: int | None = None
    snapshots: list[tuple[int, float, float]] = []
    for day in (signal_date - timedelta(days=1), signal_date, signal_date + timedelta(days=1)):
        path = (
            root / f"run={run_id}" / f"symbol={symbol}"
            / f"date={day.isoformat()}" / "events.jsonl.gz"
        )
        if not path.is_file():
            continue
        for event in _read_event_file(path):
            event_generation = int(event["connection_generation"])
            if generation != event_generation:
                generation = event_generation
                bid = ask = bid_at_ms = ask_at_ms = None
            if event["event_type"] == "market_snapshot":
                received_at_ms = int(event["received_at_ms"])
                snapshot_bid = event.get("spot_bid_raw")
                snapshot_ask = event.get("spot_ask_raw")
                if snapshot_bid is None or snapshot_ask is None:
                    continue
                snapshot_bid = int(snapshot_bid)
                snapshot_ask = int(snapshot_ask)
                if snapshot_bid <= 0 or snapshot_ask < snapshot_bid:
                    continue
                snapshots.append((received_at_ms, snapshot_bid / PRICE_SCALE, snapshot_ask / PRICE_SCALE))
                continue
            if event["event_type"] != "spot":
                continue
            received_at_ms = int(event["received_at_ms"])
            if event.get("bid_raw") is not None:
                bid = int(event["bid_raw"])
                bid_at_ms = received_at_ms
            if event.get("ask_raw") is not None:
                ask = int(event["ask_raw"])
                ask_at_ms = received_at_ms
            if bid is None or ask is None or bid_at_ms is None or ask_at_ms is None:
                continue
            if max(received_at_ms - bid_at_ms, received_at_ms - ask_at_ms) > max_staleness_ms:
                continue
            if bid <= 0 or ask <= 0 or ask < bid:
                continue
            snapshots.append((received_at_ms, bid / PRICE_SCALE, ask / PRICE_SCALE))
    return snapshots


def _label_signal(
    signal_row: dict[str, Any],
    quotes: list[tuple[int, float, float]],
    *,
    pip_size: float,
    latency_ms: int,
    max_entry_delay_ms: int,
    max_exit_quote_age_ms: int,
) -> dict[str, Any]:
    import bisect

    labeled = dict(signal_row)
    labeled.update({
        "quote_label_latency_ms": latency_ms,
        "quote_label_max_entry_delay_ms": max_entry_delay_ms,
        "quote_label_max_quote_age_ms": max_exit_quote_age_ms,
    })
    times = [quote[0] for quote in quotes]
    desired_entry_ms = int(signal_row["signal_time_ms"]) + latency_ms
    entry_index = bisect.bisect_left(times, desired_entry_ms)
    if entry_index >= len(quotes) or times[entry_index] - desired_entry_ms > max_entry_delay_ms:
        labeled["quote_label_status"] = "no_fresh_entry_quote"
        return labeled
    side = signal_row.get("side")
    if side not in (-1, 1):
        labeled["quote_label_status"] = "invalid_side"
        return labeled
    entry_time_ms, entry_bid, entry_ask = quotes[entry_index]
    entry_price = entry_ask if side == 1 else entry_bid
    labeled.update({
        "quote_label_status": "partial_labels",
        "quote_entry_time_ms": entry_time_ms,
        "quote_entry_delay_ms": entry_time_ms - int(signal_row["signal_time_ms"]),
        "quote_entry_bid": entry_bid,
        "quote_entry_ask": entry_ask,
        "quote_entry_spread_pips": (entry_ask - entry_bid) / pip_size,
        "quote_entry_price": entry_price,
        "quote_label_cost_scope": "spread included through executable bid_ask; commission_and_slippage_excluded",
    })
    for horizon_seconds in (30, 120, 300):
        target_time_ms = entry_time_ms + horizon_seconds * 1000
        exit_index = bisect.bisect_right(times, target_time_ms) - 1
        if exit_index < entry_index or target_time_ms - times[exit_index] > max_exit_quote_age_ms:
            labeled[f"return_{horizon_seconds}s_pips"] = None
            labeled[f"stress_return_{horizon_seconds}s_pips"] = None
            labeled[f"mfe_{horizon_seconds}s_pips"] = None
            labeled[f"mae_{horizon_seconds}s_pips"] = None
            continue
        path = quotes[entry_index : exit_index + 1]
        pips = [
            ((bid - entry_price) if side == 1 else (entry_price - ask)) / pip_size
            for _, bid, ask in path
        ]
        exit_bid, exit_ask = quotes[exit_index][1:]
        exit_spread_pips = (exit_ask - exit_bid) / pip_size
        labeled[f"return_{horizon_seconds}s_pips"] = pips[-1]
        labeled[f"exit_spread_{horizon_seconds}s_pips"] = exit_spread_pips
        labeled[f"stress_return_{horizon_seconds}s_pips"] = (
            pips[-1] - labeled["quote_entry_spread_pips"] - exit_spread_pips
        )
        labeled[f"mfe_{horizon_seconds}s_pips"] = max(pips)
        labeled[f"mae_{horizon_seconds}s_pips"] = min(pips)
    if all(labeled[f"return_{horizon}s_pips"] is not None for horizon in (30, 120, 300)):
        labeled["quote_label_status"] = "complete"
    return labeled


def _write_c13_labels(args: argparse.Namespace) -> tuple[int, int]:
    if not args.input.is_file():
        raise RecorderError(f"joined C13 dataset not found: {args.input}")
    if args.output.exists():
        raise RecorderError(f"refusing to overwrite existing labeled dataset: {args.output}")
    grouped: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(args.input, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RecorderError(f"corrupt joined dataset at {args.input}:{line_number}") from error
            signal_date = datetime.fromtimestamp(row["signal_time_ms"] / 1000, UTC).date()
            grouped[(row["symbol"], signal_date)].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    counts = {"signals": 0, "complete": 0}
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as target:
            for (symbol, signal_date), signals in sorted(grouped.items()):
                signals.sort(key=lambda row: row["signal_time_ms"])
                quotes = _spot_quote_path(
                    args.root,
                    args.run_id,
                    symbol,
                    signal_date,
                    max_staleness_ms=args.max_quote_age_ms,
                )
                pip_size = Instrument.from_symbol(symbol).pip_size
                for signal_row in signals:
                    labeled = _label_signal(
                        signal_row,
                        quotes,
                        pip_size=pip_size,
                        latency_ms=args.latency_ms,
                        max_entry_delay_ms=args.max_entry_delay_ms,
                        max_exit_quote_age_ms=args.max_quote_age_ms,
                    )
                    target.write(json.dumps(labeled, separators=(",", ":")) + "\n")
                    counts["signals"] += 1
                    counts["complete"] += int(labeled["quote_label_status"] == "complete")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return counts["signals"], counts["complete"]


def _evaluate_c16(args: argparse.Namespace) -> dict[str, Any]:
    from scipy.stats import t as student_t

    if not args.input.is_file():
        raise RecorderError(f"labeled C13 dataset not found: {args.input}")
    eligible_signals: list[dict[str, Any]] = []
    complete_eligible_count = 0
    with gzip.open(args.input, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RecorderError(f"corrupt labeled dataset at {args.input}:{line_number}") from error
            if row.get("c13_take") is not True or row.get("book_eligible") is not True:
                continue
            if row.get("feed_environment") != "demo":
                raise RecorderError("C16/C13 evaluation is preregistered for the demo feed")
            if (
                row.get("book_snapshot_verified") is not True
                or row.get("feature_age_ms") is None
                or int(row["feature_age_ms"]) > MAX_JOIN_STALENESS_MS
                or row.get("book_depth_age_ms") is None
                or int(row["book_depth_age_ms"]) > MAX_JOIN_STALENESS_MS
            ):
                continue
            if (
                row.get("quote_label_latency_ms") != 100
                or row.get("quote_label_max_entry_delay_ms") != 2_000
                or row.get("quote_label_max_quote_age_ms") != 1_000
            ):
                raise RecorderError("label parameters differ from the C16 pre-registration")
            eligible_signals.append(row)
            if row.get("quote_label_status") == "complete":
                complete_eligible_count += 1

    complete = [row for row in eligible_signals if row.get("quote_label_status") == "complete"]
    confirmation = [
        row for row in complete
        if row.get("directed_depth_imbalance_5") is not None
        and row.get("directed_microprice_displacement_pips") is not None
        and row["directed_depth_imbalance_5"] > 0
        and row["directed_microprice_displacement_pips"] > 0
    ]
    days = {
        datetime.fromtimestamp(row["signal_time_ms"] / 1000, UTC).date()
        for row in confirmation
    }
    capture_days = {
        path.parent.name.removeprefix("date=")
        for path in (args.root / f"run={args.run_id}").glob("symbol=*/date=*/events.jsonl.gz")
    }
    dated_capture_days = sorted(date.fromisoformat(day) for day in capture_days)
    capture_span_days = (
        (dated_capture_days[-1] - dated_capture_days[0]).days + 1 if dated_capture_days else 0
    )
    label_coverage = (
        complete_eligible_count / len(eligible_signals) if eligible_signals else 0.0
    )
    if confirmation:
        filtered_mean = sum(row["return_120s_pips"] for row in confirmation) / len(confirmation)
        filtered_stress_mean = sum(row["stress_return_120s_pips"] for row in confirmation) / len(confirmation)
    else:
        filtered_mean = filtered_stress_mean = None

    baseline_by_day: dict[date, list[float]] = defaultdict(list)
    filter_by_day: dict[date, list[float]] = defaultdict(list)
    for row in complete:
        day = datetime.fromtimestamp(row["signal_time_ms"] / 1000, UTC).date()
        baseline_by_day[day].append(float(row["return_120s_pips"]))
    for row in confirmation:
        day = datetime.fromtimestamp(row["signal_time_ms"] / 1000, UTC).date()
        filter_by_day[day].append(float(row["return_120s_pips"]))
    paired_differences = [
        sum(filter_by_day[day]) / len(filter_by_day[day])
        - sum(baseline_by_day[day]) / len(baseline_by_day[day])
        for day in sorted(filter_by_day.keys() & baseline_by_day.keys())
    ]
    if len(paired_differences) >= 2:
        mean_difference = sum(paired_differences) / len(paired_differences)
        variance = sum((value - mean_difference) ** 2 for value in paired_differences) / (
            len(paired_differences) - 1
        )
        standard_error = (variance / len(paired_differences)) ** 0.5
        if standard_error == 0:
            p_one_sided = 0.0 if mean_difference > 0 else 1.0
        else:
            t_value = mean_difference / standard_error
            p_one_sided = float(student_t.sf(t_value, len(paired_differences) - 1))
    else:
        mean_difference, p_one_sided = None, 1.0

    sample_ready = (
        len(confirmation) >= 300
        and len(days) >= 30
        and capture_span_days >= 90
        and len(capture_days) >= 90
        and label_coverage >= 0.90
    )
    survives = bool(
        sample_ready
        and filtered_mean is not None
        and filtered_mean > 0
        and filtered_stress_mean is not None
        and filtered_stress_mean > 0
        and mean_difference is not None
        and mean_difference > 0
        and p_one_sided <= 0.05
    )
    preregistration_path = REPO_ROOT / "docs" / "forex" / "c16-preregistration.md"
    return {
        "experiment": "C16",
        "run_id": args.run_id,
        "labeled_dataset_sha256": _file_sha256(args.input),
        "preregistration_sha256": _file_sha256(preregistration_path),
        "status": "provisional_survivor" if survives else (
            "inconclusive_insufficient_sample" if not sample_ready else "not_survived"
        ),
        "research_only": True,
        "orders_sent": False,
        "unit": "spread-adjusted executable quote return; commission and slippage excluded",
        "horizon_seconds_primary": 120,
        "confirmation_rule": "directed_depth_imbalance_5 > 0 and directed_microprice_displacement_pips > 0",
        "c13_take_signals_with_eligible_book": len(eligible_signals),
        "complete_labels": complete_eligible_count,
        "label_coverage": label_coverage,
        "confirmed_signals": len(confirmation),
        "confirmed_days": len(days),
        "capture_days": len(capture_days),
        "capture_span_days": capture_span_days,
        "confirmed_mean_120s_pips": filtered_mean,
        "confirmed_stress_mean_120s_pips": filtered_stress_mean,
        "paired_daily_mean_difference_vs_c13_120s_pips": mean_difference,
        "paired_days": len(paired_differences),
        "p_one_sided_daily_cluster_t": p_one_sided,
        "minimum_sample_met": sample_ready,
        "commission_slippage_verified": False,
        "approval": False,
    }


def _write_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise RecorderError(f"refusing to overwrite existing evaluation: {args.output}")
    result = _evaluate_c16(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_symbols(raw: str) -> tuple[str, ...]:
    symbols = tuple(normalize_symbol_name(item.strip()) for item in raw.split(",") if item.strip())
    if not symbols or any(not symbol.isalnum() or len(symbol) < 3 or len(symbol) > 12 for symbol in symbols):
        raise argparse.ArgumentTypeError("expected comma-separated cTrader symbols")
    if len(set(symbols)) != len(symbols):
        raise argparse.ArgumentTypeError("symbols must be unique")
    return symbols


def _parse_generations(raw: str) -> frozenset[int]:
    try:
        generations = frozenset(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("generations must be comma-separated positive integers") from error
    if any(generation <= 0 for generation in generations):
        raise argparse.ArgumentTypeError("generations must be positive")
    return generations


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="fx_c16_microstructure")
    commands = root.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("record", help="capture cTrader Open API spot and depth deltas; never sends orders")
    capture.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    capture.add_argument("--environment", choices=("demo", "live"), default="demo")
    capture.add_argument("--symbols", default=",".join(DEFAULT_CAPTURE_SYMBOLS))
    capture.add_argument("--duration-seconds", type=int)

    capture_cli = commands.add_parser(
        "record-cli", help="capture broker DOM snapshots through the existing cTrader demo CLI; read-only"
    )
    capture_cli.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    capture_cli.add_argument("--symbols", default=",".join(DEFAULT_CAPTURE_SYMBOLS))
    capture_cli.add_argument("--duration-seconds", type=int)

    materialize = commands.add_parser("materialize", help="replay raw WAL into causal book feature snapshots")
    materialize.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    materialize.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "features")
    materialize.add_argument("--run-id", required=True)
    materialize.add_argument("--symbols", type=_parse_symbols, default=DEFAULT_SYMBOLS)
    materialize.add_argument("--initial-snapshot-verified", action="store_true", help="verify generation 1 only")
    materialize.add_argument("--verified-generations", type=_parse_generations, default=frozenset())
    materialize.add_argument("--max-staleness-ms", type=int, default=MAX_JOIN_STALENESS_MS)

    join = commands.add_parser("join-c13", help="as-of join C13 signal decisions to prior book snapshots")
    join.add_argument("--features-dir", type=Path, default=DEFAULT_ROOT / "features")
    join.add_argument("--journal", type=Path, default=DEFAULT_C13_JOURNAL)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--run-id", required=True)
    join.add_argument("--symbols", type=_parse_symbols, default=DEFAULT_SYMBOLS)
    join.add_argument("--max-staleness-ms", type=int, default=MAX_JOIN_STALENESS_MS)

    labels = commands.add_parser("label-c13", help="label C13 opportunities from captured executable spot quotes")
    labels.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    labels.add_argument("--input", type=Path, required=True, help="output from join-c13")
    labels.add_argument("--output", type=Path, required=True)
    labels.add_argument("--run-id", required=True)
    labels.add_argument("--latency-ms", type=int, default=100)
    labels.add_argument("--max-entry-delay-ms", type=int, default=2_000)
    labels.add_argument("--max-quote-age-ms", type=int, default=1_000)

    evaluate = commands.add_parser(
        "evaluate", help="evaluate the preregistered book confirmation once sample gates pass"
    )
    evaluate.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    evaluate.add_argument("--input", type=Path, required=True, help="output from label-c13")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--run-id", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "record":
        if args.duration_seconds is not None and args.duration_seconds <= 0:
            raise RecorderError("--duration-seconds must be positive")
        asyncio.run(record(args))
    elif args.command == "record-cli":
        if args.duration_seconds is not None and args.duration_seconds <= 0:
            raise RecorderError("--duration-seconds must be positive")
        asyncio.run(record_cli(args))
    elif args.command == "materialize":
        if args.initial_snapshot_verified:
            args.verified_generations = args.verified_generations | frozenset({1})
        counts = _write_feature_snapshots(args)
        sys.stdout.write("C16 feature snapshots: " + json.dumps(counts, sort_keys=True) + "\n")
    elif args.command == "join-c13":
        signal_count, eligible_count = _write_c13_join(args)
        sys.stdout.write(
            f"C16 C13 join: signals={signal_count} book_eligible={eligible_count} "
            "snapshots_unverified=excluded_from_eligibility\n"
        )
    elif args.command == "label-c13":
        signal_count, labeled_count = _write_c13_labels(args)
        sys.stdout.write(f"C16 quote labels: signals={signal_count} complete={labeled_count}\n")
    else:
        result = _write_evaluation(args)
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecorderError as error:
        sys.stderr.write(f"C16 stopped safely: {error}\n")
        raise SystemExit(2) from error
