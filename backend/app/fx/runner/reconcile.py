"""On boot, make the journal and the broker agree before any new order is allowed.

Positions the broker has and the journal does not know are either adopted (they carry a server-side
stop) or closed at once (they do not). Positions the journal thinks are open but the broker no
longer has were closed by their stop or target, and are recorded as such: with their result when the
broker's own record of the deal can be asked for (`exit_of`), without one when it cannot.
A known position that lost its stop gets it back from the journal, or is closed if that is not
possible. Nothing here ever sends a new entry.

`record_exits` is the same bookkeeping for a bot that is running: a stop or target fills on the
server without the bot sending anything, so only the broker knows the result.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.fx.runner.journal import Journal
from app.fx.runner.venue import Exit, OrderRejected, Venue

# (position id, symbol) -> how the broker says the position ended, or None while it has no record.
# ponytail: a callable and not a `Venue` method, because only the CLI venue has one so far; it moves
# into the protocol when the Open API adapter gets its own (deal list request).
ExitLookup = Callable[[str, str], Awaitable[Exit | None]]


@dataclass
class ReconcileReport:
    adopted: list[str] = field(default_factory=list)
    closed_orphans: list[str] = field(default_factory=list)
    closed_while_down: list[str] = field(default_factory=list)
    restored_stops: list[str] = field(default_factory=list)


async def _journal_exits(journal: Journal, position_ids: set[str], exit_of: ExitLookup) -> list[str]:
    symbols = {e["position_id"]: e["symbol"] for e in journal.events() if e["event"] in ("order_filled", "adopted")}
    recorded = []
    for position_id in sorted(position_ids):
        exit_ = await exit_of(position_id, symbols[position_id])
        if exit_ is None:
            continue
        journal.append("closed", position_id=position_id, reason=f"broker {exit_.reason}", result=exit_.result,
                       price=exit_.price, exit_time=exit_.time)
        recorded.append(position_id)
    return recorded


async def record_exits(venue: Venue, journal: Journal, exit_of: ExitLookup) -> list[str]:
    """Journal, with their result, the positions the journal has open and the broker no longer does.

    One the broker has no record of yet stays open in the journal and is asked for again next time."""
    gone = journal.open_position_ids() - {p.id for p in await venue.positions()}
    return await _journal_exits(journal, gone, exit_of)


async def reconcile(venue: Venue, journal: Journal, exit_of: ExitLookup | None = None) -> ReconcileReport:
    report = ReconcileReport()
    known = journal.open_position_ids()
    recorded_stops = {e["position_id"]: (e["stop_price"], e["target_price"])
                      for e in journal.events() if e["event"] in ("order_filled", "adopted")}
    on_venue = {p.id: p for p in await venue.positions()}
    for position_id, position in on_venue.items():
        if position_id not in known:
            if position.stop_price is None:
                await venue.close(position_id)
                journal.append("orphan_closed", position_id=position_id, reason="no stop and unknown to the journal")
                report.closed_orphans.append(position_id)
            else:
                journal.append("adopted", position_id=position_id, symbol=position.symbol, side=position.side,
                               units=position.units, stop_price=position.stop_price,
                               target_price=position.target_price)
                report.adopted.append(position_id)
        elif position.stop_price is None:
            stop, target = recorded_stops.get(position_id, (None, None))
            try:
                restored = await venue.amend_protection(position_id, stop_price=stop, target_price=target) if stop else position
            except OrderRejected:
                restored = position
            if restored.stop_price is None:
                await venue.close(position_id)
                journal.append("orphan_closed", position_id=position_id, reason="stop lost and could not be restored")
                report.closed_orphans.append(position_id)
            else:
                report.restored_stops.append(position_id)
    gone = known - on_venue.keys()
    with_result = set(await _journal_exits(journal, gone, exit_of)) if exit_of else set()
    for position_id in sorted(gone - with_result):
        journal.append("closed_while_down", position_id=position_id)
    report.closed_while_down = sorted(gone)
    return report
