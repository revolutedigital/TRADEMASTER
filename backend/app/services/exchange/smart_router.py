"""Smart Order Router: find best execution venue with fee, liquidity, and latency awareness."""
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger
from app.services.exchange.factory import get_exchange_adapter, list_exchanges

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fee schedule per exchange (maker / taker in bps).  Values represent the
# *default* tier; a production system would pull these from the exchange API
# or from the user's VIP level.
# ---------------------------------------------------------------------------
FEE_SCHEDULE_BPS: dict[str, dict[str, Decimal]] = {
    "binance": {"maker": Decimal("10"), "taker": Decimal("10")},   # 0.10%
    "bybit":   {"maker": Decimal("10"), "taker": Decimal("10")},   # 0.10%
    # Easy to extend:
    # "okx":  {"maker": Decimal("8"),  "taker": Decimal("10")},
}

# Default fee for unknown exchanges (conservative estimate)
_DEFAULT_FEE_BPS = Decimal("15")  # 0.15%


@dataclass
class VenueQuote:
    exchange: str
    price: Decimal
    available: bool
    # --- new fields ---
    fee_bps: Decimal = Decimal("0")       # commission in basis-points
    available_liquidity: Decimal = Decimal("1")  # normalised 0-1 (1 = deep book)
    latency_ms: float = 0.0               # time to fetch the quote


@dataclass
class EffectiveCost:
    """Breakdown of effective cost for a venue."""
    exchange: str
    raw_price: Decimal
    fee_cost: Decimal          # price * fee_bps / 10_000
    slippage_estimate: Decimal # inverse-liquidity penalty
    latency_penalty: Decimal   # extra cost from slow response
    effective_price: Decimal   # final comparable number

    def __repr__(self) -> str:
        return (
            f"EffectiveCost({self.exchange}: raw={self.raw_price}, "
            f"fee={self.fee_cost}, slip={self.slippage_estimate}, "
            f"lat_pen={self.latency_penalty}, eff={self.effective_price})"
        )


class SmartOrderRouter:
    """Compare prices across exchanges and route to best venue.

    Considers:
    - Raw quoted price
    - Exchange commission fees (from FEE_SCHEDULE_BPS)
    - Liquidity depth proxy (available_liquidity 0-1)
    - Quote latency (penalises slow venues)
    """

    # Slippage multiplier: the lower the liquidity score, the higher the
    # estimated slippage.  At liquidity=1.0 -> 0 bps extra; at 0.1 -> ~9 bps.
    SLIPPAGE_MAX_BPS = Decimal("10")

    # Latency penalty: each 100 ms of latency adds this many bps to cost.
    LATENCY_PENALTY_BPS_PER_100MS = Decimal("1")  # 0.01% per 100ms

    def __init__(self) -> None:
        # Rolling average latency per exchange (EMA, alpha=0.3)
        self._avg_latency_ms: dict[str, float] = defaultdict(lambda: 0.0)
        self._latency_alpha = 0.3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def find_best_venue(self, symbol: str, side: str) -> str:
        """Find the exchange with the best *effective* price for the order.

        Effective price accounts for fees, estimated slippage, and latency.
        """
        quotes = await self._collect_quotes(symbol)

        available = [q for q in quotes if q.available]
        if not available:
            return "binance"  # Default fallback

        costs = [self._compute_effective_cost(q, side) for q in available]

        if side == "BUY":
            # Buyer wants the LOWEST effective price
            best_cost = min(costs, key=lambda c: c.effective_price)
        else:
            # Seller wants the HIGHEST effective price (= lowest cost for seller)
            best_cost = max(costs, key=lambda c: c.effective_price)

        logger.info(
            "smart_route_selected",
            exchange=best_cost.exchange,
            raw_price=float(best_cost.raw_price),
            effective_price=float(best_cost.effective_price),
            fee_cost=float(best_cost.fee_cost),
            slippage_est=float(best_cost.slippage_estimate),
            latency_pen=float(best_cost.latency_penalty),
            side=side,
        )
        return best_cost.exchange

    async def find_best_venue_detailed(
        self, symbol: str, side: str
    ) -> tuple[str, list[EffectiveCost]]:
        """Like find_best_venue but also returns the full cost breakdown."""
        quotes = await self._collect_quotes(symbol)
        available = [q for q in quotes if q.available]
        if not available:
            return "binance", []

        costs = [self._compute_effective_cost(q, side) for q in available]

        if side == "BUY":
            best_cost = min(costs, key=lambda c: c.effective_price)
        else:
            best_cost = max(costs, key=lambda c: c.effective_price)

        return best_cost.exchange, costs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect_quotes(self, symbol: str) -> list[VenueQuote]:
        """Fetch price quotes from all exchanges, recording latency."""
        quotes: list[VenueQuote] = []

        for exchange_name in list_exchanges():
            t0 = time.monotonic()
            try:
                adapter = get_exchange_adapter(exchange_name)
                if not adapter.is_connected:
                    quotes.append(VenueQuote(
                        exchange=exchange_name, price=Decimal("0"), available=False,
                    ))
                    continue

                price = await adapter.get_ticker_price(symbol)
                latency_ms = (time.monotonic() - t0) * 1000.0

                # Update rolling average latency
                prev = self._avg_latency_ms[exchange_name]
                if prev == 0.0:
                    self._avg_latency_ms[exchange_name] = latency_ms
                else:
                    self._avg_latency_ms[exchange_name] = (
                        self._latency_alpha * latency_ms
                        + (1 - self._latency_alpha) * prev
                    )

                # Fee lookup
                fee_info = FEE_SCHEDULE_BPS.get(exchange_name, {})
                fee_bps = fee_info.get("taker", _DEFAULT_FEE_BPS)

                # Liquidity score placeholder: in production this would come
                # from order-book depth (adapter.get_order_book_depth).
                # For now, connected exchanges get a base score of 1.0.
                available_liquidity = Decimal("1.0")

                quotes.append(VenueQuote(
                    exchange=exchange_name,
                    price=price,
                    available=True,
                    fee_bps=fee_bps,
                    available_liquidity=available_liquidity,
                    latency_ms=latency_ms,
                ))
            except Exception as exc:
                latency_ms = (time.monotonic() - t0) * 1000.0
                logger.debug("smart_route_quote_failed", exchange=exchange_name, error=str(exc))
                quotes.append(VenueQuote(
                    exchange=exchange_name, price=Decimal("0"), available=False,
                    latency_ms=latency_ms,
                ))

        return quotes

    def _compute_effective_cost(self, quote: VenueQuote, side: str) -> EffectiveCost:
        """Compute the all-in effective price for a venue quote.

        effective_price (BUY)  = price + fee_cost + slippage_est + latency_penalty
        effective_price (SELL) = price - fee_cost - slippage_est - latency_penalty
        """
        price = quote.price

        # 1. Fee cost: price * fee_bps / 10_000
        fee_cost = price * quote.fee_bps / Decimal("10000")

        # 2. Slippage estimate based on liquidity score.
        #    Lower liquidity -> higher slippage.
        #    slippage_bps = SLIPPAGE_MAX_BPS * (1 - liquidity)
        liquidity = max(min(quote.available_liquidity, Decimal("1")), Decimal("0"))
        slippage_bps = self.SLIPPAGE_MAX_BPS * (Decimal("1") - liquidity)
        slippage_estimate = price * slippage_bps / Decimal("10000")

        # 3. Latency penalty: use the rolling average, not just this sample
        avg_lat = Decimal(str(self._avg_latency_ms.get(quote.exchange, quote.latency_ms)))
        latency_penalty_bps = self.LATENCY_PENALTY_BPS_PER_100MS * (avg_lat / Decimal("100"))
        latency_penalty = price * latency_penalty_bps / Decimal("10000")

        # 4. Combine into effective price
        if side == "BUY":
            effective_price = price + fee_cost + slippage_estimate + latency_penalty
        else:
            # Seller: costs reduce the net proceeds
            effective_price = price - fee_cost - slippage_estimate - latency_penalty

        return EffectiveCost(
            exchange=quote.exchange,
            raw_price=price,
            fee_cost=fee_cost,
            slippage_estimate=slippage_estimate,
            latency_penalty=latency_penalty,
            effective_price=effective_price,
        )


smart_order_router = SmartOrderRouter()
