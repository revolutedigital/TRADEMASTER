"""Run the prospective research-only USD-M market-data recorder."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging import get_logger
from app.schemas.research_experiment import EvidenceGateStatusResponse
from app.services.market.microstructure_recorder import JsonlEventSink, MicrostructureRecorder
from app.services.market.microstructure_wal_audit import (
    ProspectiveWalAuditor,
    build_evidence_gate_status,
)

DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1" / "prospective-wal"
DEFAULT_AUDIT_INTERVAL_SECONDS = 86_400
DEFAULT_AUDIT_ROLLING_DAYS = 60
logger = get_logger(__name__)


async def run(
    root: Path,
    symbol: str,
    duration_seconds: int | None,
    *,
    include_spot_trades: bool = False,
    audit_status_path: Path | None = None,
    audit_rolling_days: int = DEFAULT_AUDIT_ROLLING_DAYS,
    audit_interval_seconds: int = DEFAULT_AUDIT_INTERVAL_SECONDS,
    serve_status_host: str = "0.0.0.0",  # noqa: S104 - Railway needs external bind.
    serve_status_port: int | None = None,
) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_name, stop_event.set)
    if duration_seconds is not None:
        loop.call_later(duration_seconds, stop_event.set)
    recorder = MicrostructureRecorder(
        sink=JsonlEventSink(root),
        symbol=symbol,
        include_spot_trades=include_spot_trades,
    )
    audit_task = (
        asyncio.create_task(
            _audit_status_loop(
                root=root,
                status_path=audit_status_path,
                rolling_days=audit_rolling_days,
                interval_seconds=audit_interval_seconds,
                stop_event=stop_event,
            ),
            name="microstructure_wal_audit_status",
        )
        if audit_status_path is not None
        else None
    )
    status_server_task = (
        asyncio.create_task(
            _serve_status_http(
                status_path=audit_status_path,
                host=serve_status_host,
                port=serve_status_port,
                stop_event=stop_event,
            ),
            name="microstructure_status_http",
        )
        if audit_status_path is not None and serve_status_port is not None
        else None
    )
    try:
        await recorder.run(stop_event)
    finally:
        stop_event.set()
        for task in (audit_task, status_server_task):
            if task is not None:
                await task


async def _audit_status_loop(
    *,
    root: Path,
    status_path: Path,
    rolling_days: int,
    interval_seconds: int,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(
                _write_audit_status,
                root=root,
                status_path=status_path,
                rolling_days=rolling_days,
            )
        except Exception as error:
            logger.warning("microstructure_wal_audit_status_failed", error=str(error))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass


def _write_audit_status(*, root: Path, status_path: Path, rolling_days: int) -> None:
    if rolling_days <= 0:
        raise ValueError("audit rolling days must be positive")
    audited_end_date = datetime.now(UTC).date() - timedelta(days=1)
    audited_start_date = audited_end_date - timedelta(days=rolling_days - 1)
    audits = ProspectiveWalAuditor(root).audit_range(audited_start_date, audited_end_date)
    payload = build_evidence_gate_status(audits)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_name(f"{status_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(status_path)
    logger.info(
        "microstructure_wal_audit_status_written",
        path=str(status_path),
        audited_start_date=audited_start_date.isoformat(),
        audited_end_date=audited_end_date.isoformat(),
    )


async def _serve_status_http(
    *,
    status_path: Path,
    host: str,
    port: int,
    stop_event: asyncio.Event,
) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: _handle_status_http_request(
            reader,
            writer,
            status_path=status_path,
        ),
        host,
        port,
    )
    addresses = ", ".join(str(socket.getsockname()) for socket in server.sockets or ())
    logger.info("microstructure_status_http_started", addresses=addresses)
    serve_task = asyncio.create_task(server.serve_forever())
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _pending = await asyncio.wait(
            (serve_task, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if serve_task in done and serve_task.exception() is not None:
            stop_event.set()
            raise serve_task.exception()  # type: ignore[misc]
    finally:
        stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        server.close()
        await server.wait_closed()
        serve_task.cancel()
        with suppress(asyncio.CancelledError):
            await serve_task


async def _handle_status_http_request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    status_path: Path,
) -> None:
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=2)
        method, path, _version = _parse_http_request_line(request_line)
        while True:
            header_line = await asyncio.wait_for(reader.readline(), timeout=2)
            if header_line in {b"\r\n", b"\n", b""}:
                break
        if method not in {"GET", "HEAD"}:
            response_status = "405 Method Not Allowed"
            payload = {"error": "method_not_allowed"}
        elif path == "/health":
            response_status = "200 OK"
            payload = _health_payload(status_path)
        elif path == "/evidence-gate-status.json":
            response_status = "200 OK"
            payload = _read_status_artifact(status_path)
        else:
            response_status = "404 Not Found"
            payload = {"error": "not_found"}
        body = b"" if method == "HEAD" else _json_bytes(payload)
        headers = (
            f"HTTP/1.1 {response_status}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        writer.write(headers + body)
        await writer.drain()
    except Exception as error:
        logger.warning("microstructure_status_http_request_failed", error=str(error))
        fallback_body = _json_bytes({"error": "bad_request"})
        writer.write(
            b"HTTP/1.1 400 Bad Request\r\n"
            b"Content-Type: application/json; charset=utf-8\r\n"
            + f"Content-Length: {len(fallback_body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + fallback_body
        )
        await writer.drain()
    finally:
        writer.close()
        with suppress(ConnectionError):
            await writer.wait_closed()


def _parse_http_request_line(raw_line: bytes) -> tuple[str, str, str]:
    try:
        method, raw_path, version = raw_line.decode("ascii").strip().split(" ", 2)
    except ValueError as error:
        raise ValueError("invalid HTTP request line") from error
    path = raw_path.split("?", 1)[0]
    return method.upper(), path, version


def _health_payload(status_path: Path) -> dict[str, Any]:
    return {
        "status": "healthy",
        "artifact_exists": status_path.exists(),
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _read_status_artifact(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return _fallback_status_payload("evidence_gate_artifact_missing")
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        EvidenceGateStatusResponse.model_validate(payload)
        return payload
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return _fallback_status_payload(
            f"evidence_gate_artifact_unreadable:{type(error).__name__}"
        )


def _fallback_status_payload(reason: str) -> dict[str, Any]:
    return build_evidence_gate_status(
        (),
        artifact_available=False,
        status_reasons=(reason,),
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument(
        "--include-spot-trades",
        action="store_true",
        help="Also record the public Binance Spot trade stream into spot_trade WAL files.",
    )
    parser.add_argument(
        "--audit-status-path",
        type=Path,
        help="Write the dashboard evidence-gate artifact from complete UTC days.",
    )
    parser.add_argument(
        "--audit-rolling-days",
        type=int,
        default=DEFAULT_AUDIT_ROLLING_DAYS,
        help="Number of complete UTC days to include in the rolling evidence audit.",
    )
    parser.add_argument(
        "--audit-interval-seconds",
        type=int,
        default=DEFAULT_AUDIT_INTERVAL_SECONDS,
        help="Seconds between background evidence-gate artifact refreshes.",
    )
    parser.add_argument(
        "--serve-status-host",
        default="0.0.0.0",  # noqa: S104 - Railway health routing needs external bind.
        help="Host for the read-only recorder status HTTP server.",
    )
    parser.add_argument(
        "--serve-status-port",
        type=int,
        help="Port for the read-only recorder status HTTP server.",
    )
    arguments = parser.parse_args()
    if arguments.duration_seconds is not None and arguments.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if arguments.audit_rolling_days <= 0:
        parser.error("--audit-rolling-days must be positive")
    if arguments.audit_interval_seconds <= 0:
        parser.error("--audit-interval-seconds must be positive")
    if arguments.serve_status_port is not None:
        if arguments.serve_status_port <= 0:
            parser.error("--serve-status-port must be positive")
        if arguments.audit_status_path is None:
            parser.error("--serve-status-port requires --audit-status-path")
    asyncio.run(
        run(
            arguments.root,
            arguments.symbol,
            arguments.duration_seconds,
            include_spot_trades=arguments.include_spot_trades,
            audit_status_path=arguments.audit_status_path,
            audit_rolling_days=arguments.audit_rolling_days,
            audit_interval_seconds=arguments.audit_interval_seconds,
            serve_status_host=arguments.serve_status_host,
            serve_status_port=arguments.serve_status_port,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
