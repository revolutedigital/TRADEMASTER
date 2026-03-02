"""Tax reporting service: FIFO cost-basis calculation."""
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaxableEvent:
    date: datetime
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    cost_basis: Decimal
    realized_pnl: Decimal
    holding_period_days: int
    is_short_term: bool


class TaxReporter:
    """Calculate realized gains/losses using FIFO method."""

    def calculate_fifo_gains(self, trades: list[dict]) -> list[TaxableEvent]:
        """Apply FIFO to calculate cost basis and realized gains."""
        lots: dict[str, list[dict]] = {}
        events: list[TaxableEvent] = []

        for trade in sorted(trades, key=lambda t: t["timestamp"]):
            symbol = trade["symbol"]
            side = trade["side"]
            qty = Decimal(str(trade["quantity"]))
            price = Decimal(str(trade["price"]))

            if side == "BUY":
                lots.setdefault(symbol, []).append({"qty": qty, "price": price, "date": trade["timestamp"]})
            elif side == "SELL" and symbol in lots:
                remaining = qty
                while remaining > 0 and lots.get(symbol):
                    lot = lots[symbol][0]
                    matched = min(remaining, lot["qty"])
                    cost_basis = lot["price"] * matched
                    proceeds = price * matched
                    pnl = proceeds - cost_basis
                    days = (trade["timestamp"] - lot["date"]).days

                    events.append(TaxableEvent(
                        date=trade["timestamp"],
                        symbol=symbol,
                        side="SELL",
                        quantity=matched,
                        price=price,
                        cost_basis=cost_basis,
                        realized_pnl=pnl,
                        holding_period_days=days,
                        is_short_term=days < 365,
                    ))

                    lot["qty"] -= matched
                    remaining -= matched
                    if lot["qty"] <= 0:
                        lots[symbol].pop(0)

        return events

    def generate_summary(self, events: list[TaxableEvent]) -> dict:
        total_short = sum(e.realized_pnl for e in events if e.is_short_term)
        total_long = sum(e.realized_pnl for e in events if not e.is_short_term)
        return {
            "total_events": len(events),
            "short_term_gains": float(total_short),
            "long_term_gains": float(total_long),
            "total_gains": float(total_short + total_long),
        }


tax_reporter = TaxReporter()
