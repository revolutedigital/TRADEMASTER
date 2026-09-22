"""Run the prospective research-only USD-M market-data recorder."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging import get_logger
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
    try:
        await recorder.run(stop_event)
    finally:
        stop_event.set()
        if audit_task is not None:
            await audit_task


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
    arguments = parser.parse_args()
    if arguments.duration_seconds is not None and arguments.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if arguments.audit_rolling_days <= 0:
        parser.error("--audit-rolling-days must be positive")
    if arguments.audit_interval_seconds <= 0:
        parser.error("--audit-interval-seconds must be positive")
    asyncio.run(
        run(
            arguments.root,
            arguments.symbol,
            arguments.duration_seconds,
            include_spot_trades=arguments.include_spot_trades,
            audit_status_path=arguments.audit_status_path,
            audit_rolling_days=arguments.audit_rolling_days,
            audit_interval_seconds=arguments.audit_interval_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
