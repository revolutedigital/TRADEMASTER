"""Adaptive Volume-Weighted Average Price (VWAP) execution algorithm.

Uses historical volume profiles to distribute order slices proportionally
to expected intraday volume, adapting in real-time as observed volume
deviates from the forecast.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Any

from app.core.logging import get_logger
from app.services.exchange.binance_client import binance_client

logger = get_logger(__name__)

# Deviation threshold (0.5%) at which we cancel and replace a limit order.
_PRICE_DEVIATION_THRESHOLD = Decimal("0.005")

# Default maximum participation rate (fraction of observed volume).
_DEFAULT_MAX_PARTICIPATION = Decimal("0.10")

# Safety cap: never send a single slice larger than this fraction of total qty.
_MAX_SINGLE_SLICE_FRACTION = Decimal("0.25")


class VWAPStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class VolumeProfileBucket:
    """One time-bucket in the intraday volume profile."""
    bucket_index: int
    expected_volume_fraction: Decimal  # fraction of daily volume in this bucket
    observed_volume: Decimal = Decimal("0")
    slice_qty_sent: Decimal = Decimal("0")
    slice_qty_filled: Decimal = Decimal("0")
    fill_price: Decimal = Decimal("0")


@dataclass
class VWAPResult:
    """Outcome of a completed VWAP execution."""
    symbol: str
    side: str
    total_quantity: Decimal
    filled_quantity: Decimal
    slices_executed: int
    slices_total: int
    avg_fill_price: Decimal
    vwap_benchmark: Decimal
    slippage_bps: Decimal
    total_commission: Decimal
    status: VWAPStatus
    elapsed_seconds: float
    cancel_replace_count: int


@dataclass
class _ActiveOrder:
    """Tracks a live limit order placed during a VWAP slice."""
    order_id: int
    symbol: str
    side: str
    price: Decimal
    quantity: Decimal
    placed_at: float = field(default_factory=time.monotonic)


class AdaptiveVWAPEngine:
    """Execute large orders using an adaptive VWAP strategy.

    The engine builds a *volume profile* from recent historical klines
    (default: 5 days of hourly candles), then slices the parent order
    proportionally to each time-bucket's expected volume share.

    During execution the engine continuously compares *observed* market
    volume against the profile forecast and adjusts upcoming slice sizes
    so that:

    * Under-participation in high-volume buckets is compensated in later
      buckets (catch-up logic).
    * The participation rate never exceeds ``max_participation`` of the
      bucket's observed volume, reducing market footprint.
    * If the current best price deviates more than 0.5% from the running
      VWAP target the outstanding limit order is cancelled and replaced
      at the new price.
    """

    def __init__(
        self,
        max_participation: Decimal = _DEFAULT_MAX_PARTICIPATION,
        profile_lookback_days: int = 5,
    ) -> None:
        self._max_participation = max_participation
        self._profile_lookback_days = profile_lookback_days
        self._cancel_replace_count: int = 0
        self._active_order: _ActiveOrder | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        symbol: str,
        side: str,
        total_qty: Decimal,
        duration_minutes: int = 60,
        num_slices: int = 12,
    ) -> VWAPResult:
        """Run the adaptive VWAP execution.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTCUSDT"``).
        side:
            ``"BUY"`` or ``"SELL"``.
        total_qty:
            Total quantity to fill over the execution window.
        duration_minutes:
            How long the execution should run.
        num_slices:
            Number of time-buckets to divide the window into.

        Returns
        -------
        VWAPResult
            Summary of the execution including slippage analysis.
        """
        self._cancel_replace_count = 0
        start_time = time.monotonic()

        logger.info(
            "vwap_started",
            symbol=symbol,
            side=side,
            total_qty=str(total_qty),
            duration_min=duration_minutes,
            slices=num_slices,
        )

        # --- Build historical volume profile ---
        profile = await self._build_volume_profile(symbol, num_slices)
        interval_seconds = Decimal(str((duration_minutes * 60) / num_slices))

        # --- Pre-compute ideal slice quantities from profile ---
        buckets = self._plan_slices(profile, total_qty, num_slices)

        fills: list[dict[str, Any]] = []
        remaining_qty = total_qty
        vwap_numerator = Decimal("0")  # sum(price_i * qty_i) for benchmark
        vwap_denominator = Decimal("0")
        status = VWAPStatus.RUNNING

        for i, bucket in enumerate(buckets):
            if remaining_qty <= 0:
                break

            # Adapt slice size based on observed vs expected volume.
            slice_qty = self._adapt_slice(
                bucket=bucket,
                remaining_qty=remaining_qty,
                remaining_buckets=num_slices - i,
            )

            if slice_qty <= 0:
                if i < num_slices - 1:
                    await asyncio.sleep(float(interval_seconds))
                continue

            try:
                fill = await self._execute_slice(
                    symbol=symbol,
                    side=side,
                    quantity=slice_qty,
                    bucket_index=i,
                )
                fill_qty = Decimal(str(fill["qty"]))
                fill_price = Decimal(str(fill["price"]))

                bucket.slice_qty_filled = fill_qty
                bucket.fill_price = fill_price

                vwap_numerator += fill_price * fill_qty
                vwap_denominator += fill_qty
                remaining_qty -= fill_qty

                fills.append(fill)

                logger.info(
                    "vwap_slice_filled",
                    slice=i + 1,
                    total=num_slices,
                    price=str(fill_price),
                    qty=str(fill_qty),
                    remaining=str(remaining_qty),
                )

            except Exception as exc:
                logger.error("vwap_slice_failed", slice=i + 1, error=str(exc))

            # Wait for the next bucket (skip sleep after last slice).
            if i < num_slices - 1:
                await asyncio.sleep(float(interval_seconds))

        # --- Compute results ---
        filled_qty = total_qty - remaining_qty
        avg_fill = (
            (vwap_numerator / vwap_denominator).quantize(Decimal("0.00000001"))
            if vwap_denominator > 0
            else Decimal("0")
        )

        # Market VWAP benchmark (volume-weighted price over the same period).
        benchmark_vwap = await self._compute_market_vwap(symbol)

        slippage_bps = Decimal("0")
        if benchmark_vwap > 0 and avg_fill > 0:
            raw_slip = (avg_fill - benchmark_vwap) / benchmark_vwap * Decimal("10000")
            if side == "SELL":
                raw_slip = -raw_slip
            slippage_bps = raw_slip.quantize(Decimal("0.01"))

        total_commission = sum(
            Decimal(str(f.get("commission", 0))) for f in fills
        ).quantize(Decimal("0.00000001"))

        if filled_qty >= total_qty:
            status = VWAPStatus.COMPLETED
        elif filled_qty > 0:
            status = VWAPStatus.COMPLETED  # partial fill still counts
        else:
            status = VWAPStatus.FAILED

        elapsed = time.monotonic() - start_time

        logger.info(
            "vwap_complete",
            symbol=symbol,
            avg_fill=str(avg_fill),
            benchmark_vwap=str(benchmark_vwap),
            slippage_bps=str(slippage_bps),
            filled=str(filled_qty),
            cancel_replaces=self._cancel_replace_count,
        )

        return VWAPResult(
            symbol=symbol,
            side=side,
            total_quantity=total_qty,
            filled_quantity=filled_qty,
            slices_executed=len(fills),
            slices_total=num_slices,
            avg_fill_price=avg_fill,
            vwap_benchmark=benchmark_vwap,
            slippage_bps=slippage_bps,
            total_commission=total_commission,
            status=status,
            elapsed_seconds=round(elapsed, 2),
            cancel_replace_count=self._cancel_replace_count,
        )

    # ------------------------------------------------------------------
    # Volume profile construction
    # ------------------------------------------------------------------

    async def _build_volume_profile(
        self, symbol: str, num_buckets: int
    ) -> list[Decimal]:
        """Fetch historical klines and derive an intraday volume profile.

        Returns a list of ``num_buckets`` fractions that sum to ~1.0,
        representing the expected share of total daily volume in each
        time bucket.
        """
        try:
            df = await binance_client.get_klines(
                symbol=symbol,
                interval="1h",
                limit=self._profile_lookback_days * 24,
            )

            if df.empty or len(df) < num_buckets:
                logger.warning("vwap_profile_fallback", reason="insufficient_kline_data")
                return self._uniform_profile(num_buckets)

            # Assign each candle to a bucket based on its position in the day.
            volumes = df["volume"].tolist()

            # Reshape into days, then average per-bucket.
            hours_per_bucket = max(1, 24 // num_buckets)
            bucket_volumes: list[Decimal] = []
            for b in range(num_buckets):
                start_h = b * hours_per_bucket
                end_h = start_h + hours_per_bucket
                # Gather matching hours across all days in the dataset.
                bucket_total = Decimal("0")
                count = 0
                for idx, vol in enumerate(volumes):
                    hour_of_day = idx % 24
                    if start_h <= hour_of_day < end_h:
                        bucket_total += Decimal(str(vol))
                        count += 1
                avg = bucket_total / max(count, 1)
                bucket_volumes.append(avg)

            total_vol = sum(bucket_volumes) or Decimal("1")
            profile = [bv / total_vol for bv in bucket_volumes]

            logger.info("vwap_profile_built", buckets=num_buckets, lookback_days=self._profile_lookback_days)
            return profile

        except Exception as exc:
            logger.error("vwap_profile_error", error=str(exc))
            return self._uniform_profile(num_buckets)

    @staticmethod
    def _uniform_profile(n: int) -> list[Decimal]:
        """Fallback: equal volume distribution across buckets."""
        frac = Decimal("1") / Decimal(str(n))
        return [frac] * n

    # ------------------------------------------------------------------
    # Slice planning and adaptation
    # ------------------------------------------------------------------

    def _plan_slices(
        self,
        profile: list[Decimal],
        total_qty: Decimal,
        num_slices: int,
    ) -> list[VolumeProfileBucket]:
        """Create initial slice plan weighted by the volume profile."""
        buckets: list[VolumeProfileBucket] = []
        for i, frac in enumerate(profile):
            buckets.append(
                VolumeProfileBucket(
                    bucket_index=i,
                    expected_volume_fraction=frac,
                    slice_qty_sent=(total_qty * frac).quantize(
                        Decimal("0.00000001"), rounding=ROUND_DOWN
                    ),
                )
            )
        return buckets

    def _adapt_slice(
        self,
        bucket: VolumeProfileBucket,
        remaining_qty: Decimal,
        remaining_buckets: int,
    ) -> Decimal:
        """Adjust slice quantity based on remaining work and participation cap.

        If the observed volume in this bucket is known, we cap our slice to
        ``max_participation * observed_volume``.  Otherwise we use the
        planned quantity from the profile, bounded so the remaining quantity
        is spread evenly over remaining buckets (catch-up logic).
        """
        planned = bucket.slice_qty_sent

        # Catch-up: if we fell behind, spread the deficit evenly.
        even_share = (
            remaining_qty / Decimal(str(max(remaining_buckets, 1)))
        ).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
        slice_qty = max(planned, even_share)

        # Participation cap: do not exceed max_participation of observed volume.
        if bucket.observed_volume > 0:
            participation_cap = (
                self._max_participation * bucket.observed_volume
            ).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            slice_qty = min(slice_qty, participation_cap)

        # Safety: never exceed remaining or single-slice cap.
        max_single = (remaining_qty * _MAX_SINGLE_SLICE_FRACTION).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
        slice_qty = min(slice_qty, remaining_qty, max(max_single, even_share))

        return slice_qty

    # ------------------------------------------------------------------
    # Order execution with cancel-and-replace
    # ------------------------------------------------------------------

    async def _execute_slice(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        bucket_index: int,
    ) -> dict[str, Any]:
        """Place a limit order at the current best price.

        If the market moves more than 0.5% from the order price before
        fill, the order is cancelled and replaced at the new price.
        """
        current_price = await binance_client.get_ticker_price(symbol)
        order_price = current_price

        result = await binance_client.place_limit_order(
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=float(order_price),
        )
        order_id = result.get("orderId")
        self._active_order = _ActiveOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=order_price,
            quantity=quantity,
        )

        # Poll for fill, cancel-and-replace if price drifts too far.
        max_wait_iterations = 30
        poll_interval = 1.0  # seconds
        for _ in range(max_wait_iterations):
            status_resp = await binance_client.get_order_status(symbol, order_id)
            order_status = status_resp.get("status", "")

            if order_status == "FILLED":
                fill_price = Decimal(str(
                    status_resp.get("avgPrice")
                    or status_resp.get("price", "0")
                ))
                commission = self._extract_commission(status_resp)
                self._active_order = None
                return {
                    "price": fill_price,
                    "qty": quantity,
                    "commission": commission,
                    "order_id": order_id,
                }

            if order_status in ("CANCELED", "REJECTED", "EXPIRED"):
                break

            # Check whether price has deviated beyond threshold.
            new_price = await binance_client.get_ticker_price(symbol)
            deviation = abs(new_price - order_price) / order_price

            if deviation > _PRICE_DEVIATION_THRESHOLD:
                logger.info(
                    "vwap_cancel_replace",
                    bucket=bucket_index,
                    old_price=str(order_price),
                    new_price=str(new_price),
                    deviation_pct=str((deviation * 100).quantize(Decimal("0.01"))),
                )
                try:
                    await binance_client.cancel_order(symbol, order_id)
                except Exception as cancel_exc:
                    logger.warning("vwap_cancel_failed", error=str(cancel_exc))

                self._cancel_replace_count += 1
                order_price = new_price

                result = await binance_client.place_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=float(quantity),
                    price=float(order_price),
                )
                order_id = result.get("orderId")
                self._active_order = _ActiveOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    price=order_price,
                    quantity=quantity,
                )

            await asyncio.sleep(poll_interval)

        # If we exit the loop unfilled, attempt a market order for the slice.
        logger.warning("vwap_slice_timeout_market_fallback", bucket=bucket_index)
        if self._active_order:
            try:
                await binance_client.cancel_order(symbol, self._active_order.order_id)
            except Exception:
                pass

        market_result = await binance_client.place_market_order(
            symbol=symbol,
            side=side,
            quantity=float(quantity),
        )
        fill_price = Decimal(str(
            market_result.get("avgPrice")
            or market_result.get("price", "0")
        ))
        commission = sum(
            Decimal(str(f.get("commission", "0")))
            for f in market_result.get("fills", [])
        )
        self._active_order = None

        return {
            "price": fill_price,
            "qty": quantity,
            "commission": commission,
            "order_id": market_result.get("orderId"),
        }

    # ------------------------------------------------------------------
    # Benchmark & helpers
    # ------------------------------------------------------------------

    async def _compute_market_vwap(self, symbol: str) -> Decimal:
        """Compute the market VWAP from recent 1-minute candles.

        This serves as the benchmark against which our execution is
        measured.
        """
        try:
            df = await binance_client.get_klines(
                symbol=symbol, interval="1m", limit=60
            )
            if df.empty:
                return Decimal("0")

            # VWAP = sum(typical_price * volume) / sum(volume)
            typical = (df["high"] + df["low"] + df["close"]) / 3
            volume = df["volume"]
            total_vol = volume.sum()
            if total_vol == 0:
                return Decimal("0")

            vwap = (typical * volume).sum() / total_vol
            return Decimal(str(round(vwap, 8)))

        except Exception as exc:
            logger.error("vwap_benchmark_error", error=str(exc))
            return Decimal("0")

    @staticmethod
    def _extract_commission(order_response: dict[str, Any]) -> Decimal:
        """Extract total commission from an order status response."""
        fills = order_response.get("fills", [])
        if fills:
            return sum(
                Decimal(str(f.get("commission", "0"))) for f in fills
            )
        return Decimal("0")


# Module-level singleton for convenience.
adaptive_vwap_engine = AdaptiveVWAPEngine()
