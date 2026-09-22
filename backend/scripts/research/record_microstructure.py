"""Run the prospective research-only USD-M market-data recorder."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.market.microstructure_recorder import JsonlEventSink, MicrostructureRecorder

DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1" / "prospective-wal"


async def run(
    root: Path,
    symbol: str,
    duration_seconds: int | None,
    *,
    include_spot_trades: bool = False,
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
    await recorder.run(stop_event)


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
    arguments = parser.parse_args()
    if arguments.duration_seconds is not None and arguments.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    asyncio.run(
        run(
            arguments.root,
            arguments.symbol,
            arguments.duration_seconds,
            include_spot_trades=arguments.include_spot_trades,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
