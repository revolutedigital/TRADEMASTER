"""Samples the demo account's live bid/ask for the 37 banca markets every two minutes, to measure the real spread by
market and hour against the cost the research rounds assume. Read-only: it only sends `price`.

    python -m scripts.research.fx_spread_sampler        (touch ~/.config/trademaster/STOP_SPREADS to stop it)

Writes data/c13/spreads.jsonl: one line per quote {at, symbol, bid, ask}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.fx.runner.ctrader_cli import PtySession, json_of
from scripts.research import fx_banca_demo as banca
from scripts.research import fx_c9_lab as c9
from scripts.research import fx_demo_runner as demo

OUT = Path("data/c13/spreads.jsonl")
STOP = demo.STOP_FILE.with_name("STOP_SPREADS")
CYCLE_SECONDS = 120
NAME = "SPREAD"  # container suffix: never collides with the banca's or the dashboard's session


def cycle(session: PtySession, symbols: list[str]) -> list[dict]:
    rows = []
    for symbol in symbols:
        quote = json_of(session.send(f"price {symbol}", timeout=60))
        if quote.get("bid") and quote.get("ask"):
            rows.append({"at": time.time(), "symbol": symbol, "bid": quote["bid"], "ask": quote["ask"]})
    return rows


def main() -> int:
    symbols = [banca.FUSION[s] for s in c9.SYMBOLS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    session = None
    while not STOP.exists():
        started = time.time()
        try:
            if session is None:
                demo.remove_container(NAME)
                session = demo.cli_session(NAME)
                session.start()
            rows = cycle(session, symbols)
            with OUT.open("a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row) + "\n" for row in rows)
        except Exception as error:  # noqa: BLE001 - any failure: log it, log in again on the next cycle
            sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {type(error).__name__}: {str(error)[:200]}\n")
            if session is not None:
                session.close()
                session = None
            demo.remove_container(NAME)
        time.sleep(max(5.0, CYCLE_SECONDS - (time.time() - started)))
    if session is not None:
        session.close()
    demo.remove_container(NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
