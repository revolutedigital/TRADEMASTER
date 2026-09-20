"""On boot, make the journal and the broker agree before any new order is allowed.

Positions the broker has and the journal does not know are either adopted (they carry a server-side
stop) or closed at once (they do not). Positions the journal thinks are open but the broker no
longer has were closed by their stop or target while the bot was down, and are recorded as such.
A known position that lost its stop gets it back from the journal, or is closed if that is not
possible. Nothing here ever sends a new entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.fx.runner.journal import Journal
from app.fx.runner.venue import OrderRejected, Venue


@dataclass
class ReconcileReport:
    adopted: list[str] = field(default_factory=list)
    closed_orphans: list[str] = field(default_factory=list)
    closed_while_down: list[str] = field(default_factory=list)
    restored_stops: list[str] = field(default_factory=list)


async def reconcile(venue: Venue, journal: Journal) -> ReconcileReport:
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
    for position_id in known - on_venue.keys():
        journal.append("closed_while_down", position_id=position_id)
        report.closed_while_down.append(position_id)
    return report
