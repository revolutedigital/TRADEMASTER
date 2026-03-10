"""Liquidity risk modeling - illiquidity measurement, market impact, and execution sizing."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LiquidityAssessment:
    """Result of a liquidity risk evaluation for a single asset."""
    symbol: str
    amihud_ratio: float
    market_impact_bps: float
    liquidity_at_risk: float
    spread_z_score: float
    optimal_trade_size_usd: float
    crisis_detected: bool
    timestamp: str


@dataclass
class SpreadSnapshot:
    """A single bid-ask spread observation."""
    symbol: str
    bid: Decimal
    ask: Decimal
    timestamp: str

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread_bps(self) -> float:
        """Spread in basis points relative to mid price."""
        if self.mid == 0:
            return 0.0
        return float((self.spread / self.mid) * Decimal("10000"))


class LiquidityRiskModel:
    """
    Comprehensive liquidity risk analysis for crypto and traditional assets.

    Provides:
    - Amihud illiquidity ratio per asset
    - Market impact estimation (Kyle's lambda proxy)
    - Liquidity-at-Risk (LaR): VaR adjusted for liquidation cost
    - Bid-ask spread monitoring with z-score anomaly detection
    - Optimal execution sizing based on available liquidity
    """

    # Z-score threshold for declaring a liquidity crisis
    CRISIS_Z_THRESHOLD = 3.0
    # Maximum lookback for spread history per asset
    MAX_SPREAD_HISTORY = 500
    # Maximum lookback for price/volume history
    MAX_PRICE_HISTORY = 500

    def __init__(
        self,
        crisis_z_threshold: float = 3.0,
        max_market_impact_bps: float = 50.0,
    ):
        self.crisis_z_threshold = crisis_z_threshold
        self.max_market_impact_bps = max_market_impact_bps

        # Internal state keyed by symbol
        self._price_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[float]] = {}
        self._spread_history: dict[str, list[SpreadSnapshot]] = {}

        logger.info("liquidity_risk_model_initialized",
                    crisis_z=crisis_z_threshold,
                    max_impact_bps=max_market_impact_bps)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_price_volume(
        self, symbol: str, price: float, volume_usd: float
    ) -> None:
        """Record a new price and dollar-volume observation for an asset."""
        if price <= 0 or volume_usd <= 0:
            return

        self._price_history.setdefault(symbol, []).append(price)
        self._volume_history.setdefault(symbol, []).append(volume_usd)

        # Trim to bounded window
        if len(self._price_history[symbol]) > self.MAX_PRICE_HISTORY:
            self._price_history[symbol] = self._price_history[symbol][-self.MAX_PRICE_HISTORY:]
        if len(self._volume_history[symbol]) > self.MAX_PRICE_HISTORY:
            self._volume_history[symbol] = self._volume_history[symbol][-self.MAX_PRICE_HISTORY:]

    def update_spread(
        self, symbol: str, bid: float, ask: float
    ) -> None:
        """Record a bid-ask spread observation.

        Uses Decimal internally to avoid floating-point artifacts in
        spread calculations that affect trading cost accounting.
        """
        if bid <= 0 or ask <= 0 or ask < bid:
            return

        snap = SpreadSnapshot(
            symbol=symbol,
            bid=Decimal(str(bid)),
            ask=Decimal(str(ask)),
            timestamp=datetime.utcnow().isoformat(),
        )

        self._spread_history.setdefault(symbol, []).append(snap)
        if len(self._spread_history[symbol]) > self.MAX_SPREAD_HISTORY:
            self._spread_history[symbol] = self._spread_history[symbol][-self.MAX_SPREAD_HISTORY:]

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def amihud_illiquidity(self, symbol: str, window: int = 30) -> float:
        """Amihud (2002) illiquidity ratio.

        ILLIQ = (1/N) * sum(|r_t| / V_t)

        where r_t is the return on day t and V_t is the dollar volume.
        Higher values indicate lower liquidity (harder to trade without
        moving the price).

        Args:
            symbol: Asset identifier.
            window: Number of most-recent observations to use.

        Returns:
            The Amihud ratio scaled by 1e6 for readability, or 0.0 when
            there is insufficient data.
        """
        prices = self._price_history.get(symbol, [])
        volumes = self._volume_history.get(symbol, [])

        min_len = min(len(prices), len(volumes))
        if min_len < 2:
            return 0.0

        prices_arr = np.array(prices[-window:], dtype=np.float64)
        volumes_arr = np.array(volumes[-window:], dtype=np.float64)

        # Align lengths after slicing
        n = min(len(prices_arr), len(volumes_arr))
        prices_arr = prices_arr[-n:]
        volumes_arr = volumes_arr[-n:]

        if n < 2:
            return 0.0

        returns = np.abs(np.diff(prices_arr) / prices_arr[:-1])
        vol_slice = volumes_arr[1:]

        # Guard against zero volume
        safe_volumes = np.where(vol_slice > 0, vol_slice, np.inf)
        illiq = float(np.mean(returns / safe_volumes))

        # Scale by 1e6 for readability
        return round(illiq * 1e6, 6)

    def estimate_market_impact(
        self,
        symbol: str,
        trade_size_usd: float,
        window: int = 30,
    ) -> float:
        """Estimate the price impact of executing a trade of a given size.

        Uses a square-root market impact model:
            impact_bps = lambda * sqrt(trade_size / ADV)

        where lambda is calibrated from the Amihud ratio and ADV is
        the average daily dollar volume.

        Args:
            symbol: Asset identifier.
            trade_size_usd: Notional value of the intended trade in USD.
            window: Lookback window for average volume.

        Returns:
            Estimated one-way market impact in basis points.
        """
        volumes = self._volume_history.get(symbol, [])
        if len(volumes) < 2 or trade_size_usd <= 0:
            return 0.0

        adv = float(np.mean(volumes[-window:]))
        if adv <= 0:
            return 0.0

        participation_rate = trade_size_usd / adv

        # Empirical lambda calibrated from Amihud ratio
        amihud = self.amihud_illiquidity(symbol, window)
        # lambda in [5, 100] bps range, scaled by illiquidity
        lam = max(5.0, min(100.0, 10.0 + amihud * 1e3))

        impact_bps = lam * np.sqrt(participation_rate)
        return round(float(min(impact_bps, self.max_market_impact_bps * 10)), 2)

    def liquidity_at_risk(
        self,
        symbol: str,
        portfolio_value: float,
        weight: float,
        confidence: float = 0.95,
        window: int = 60,
    ) -> float:
        """Liquidity-at-Risk (LaR): VaR adjusted for liquidation cost.

        LaR = VaR + Liquidation Cost
        Liquidation Cost = half-spread + market impact of unwinding the
        full position.

        Args:
            symbol: Asset identifier.
            portfolio_value: Total portfolio value in USD.
            weight: Fraction of portfolio in this asset (0-1).
            confidence: VaR confidence level.
            window: Lookback for return distribution.

        Returns:
            LaR in USD terms.
        """
        prices = self._price_history.get(symbol, [])
        if len(prices) < 10 or portfolio_value <= 0 or weight <= 0:
            return 0.0

        position_usd = portfolio_value * weight

        # Historical VaR component
        arr = np.array(prices[-window:], dtype=np.float64)
        returns = np.diff(arr) / arr[:-1]
        if len(returns) < 5:
            return 0.0

        percentile = (1 - confidence) * 100
        var_pct = abs(float(np.percentile(returns, percentile)))
        var_usd = var_pct * position_usd

        # Liquidation cost: half-spread + market impact
        half_spread_cost = self._average_half_spread_cost(symbol, position_usd)
        impact_bps = self.estimate_market_impact(symbol, position_usd, window)
        impact_cost = position_usd * (impact_bps / 10_000)

        lar = var_usd + half_spread_cost + impact_cost
        return round(lar, 2)

    def spread_z_score(self, symbol: str, lookback: int = 100) -> float:
        """Compute the z-score of the current spread vs. recent history.

        A z-score above the crisis threshold signals abnormal widening
        (potential liquidity crisis).

        Args:
            symbol: Asset identifier.
            lookback: Number of observations for mean/std calculation.

        Returns:
            Z-score of the latest spread, or 0.0 if insufficient data.
        """
        snaps = self._spread_history.get(symbol, [])
        if len(snaps) < 10:
            return 0.0

        spread_series = np.array(
            [s.spread_bps for s in snaps[-lookback:]],
            dtype=np.float64,
        )

        mean_spread = float(np.mean(spread_series[:-1]))
        std_spread = float(np.std(spread_series[:-1], ddof=1))

        if std_spread < 1e-10:
            return 0.0

        current_spread = spread_series[-1]
        z = (current_spread - mean_spread) / std_spread
        return round(float(z), 4)

    def is_liquidity_crisis(self, symbol: str) -> bool:
        """Return True if the current spread z-score exceeds the crisis threshold."""
        return self.spread_z_score(symbol) > self.crisis_z_threshold

    # ------------------------------------------------------------------
    # Execution guidance
    # ------------------------------------------------------------------

    def optimal_execution_size(
        self,
        symbol: str,
        total_order_usd: float,
        max_impact_bps: float | None = None,
        window: int = 30,
    ) -> dict:
        """Determine the optimal single-clip size to limit market impact.

        Solves for the clip size Q such that:
            lambda * sqrt(Q / ADV) <= max_impact_bps

        If the total order is larger than one clip, the result includes
        a recommended number of clips and a suggested time interval.

        Args:
            symbol: Asset identifier.
            total_order_usd: Full intended order in USD.
            max_impact_bps: Maximum acceptable impact per clip. Defaults
                to the model's max_market_impact_bps.
            window: Lookback for ADV.

        Returns:
            Dictionary with clip sizing details.
        """
        if max_impact_bps is None:
            max_impact_bps = self.max_market_impact_bps

        volumes = self._volume_history.get(symbol, [])
        if len(volumes) < 2 or total_order_usd <= 0:
            return {
                "symbol": symbol,
                "total_order_usd": total_order_usd,
                "clip_size_usd": total_order_usd,
                "n_clips": 1,
                "estimated_impact_bps": 0.0,
                "status": "insufficient_data",
            }

        adv = float(np.mean(volumes[-window:]))
        if adv <= 0:
            return {
                "symbol": symbol,
                "total_order_usd": total_order_usd,
                "clip_size_usd": total_order_usd,
                "n_clips": 1,
                "estimated_impact_bps": 0.0,
                "status": "zero_volume",
            }

        amihud = self.amihud_illiquidity(symbol, window)
        lam = max(5.0, min(100.0, 10.0 + amihud * 1e3))

        # max_impact_bps >= lam * sqrt(Q / ADV)
        # => Q <= ADV * (max_impact_bps / lam)^2
        max_clip = adv * (max_impact_bps / lam) ** 2
        max_clip = max(max_clip, 1.0)  # floor to avoid zero

        clip_size = min(total_order_usd, max_clip)
        n_clips = max(1, int(np.ceil(total_order_usd / clip_size)))

        # Re-estimate impact of chosen clip
        impact = self.estimate_market_impact(symbol, clip_size, window)

        # Participation guidance: keep each clip below 5% of ADV
        participation = clip_size / adv
        if participation > 0.05:
            suggested_interval_min = 15
        elif participation > 0.01:
            suggested_interval_min = 5
        else:
            suggested_interval_min = 1

        return {
            "symbol": symbol,
            "total_order_usd": round(total_order_usd, 2),
            "clip_size_usd": round(clip_size, 2),
            "n_clips": n_clips,
            "estimated_impact_bps": impact,
            "adv_usd": round(adv, 2),
            "participation_rate_pct": round(participation * 100, 4),
            "suggested_interval_min": suggested_interval_min,
            "status": "ok",
        }

    # ------------------------------------------------------------------
    # Composite assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        symbol: str,
        portfolio_value: float = 0.0,
        weight: float = 0.0,
        confidence: float = 0.95,
    ) -> LiquidityAssessment:
        """Run a full liquidity risk assessment for a single asset.

        Args:
            symbol: Asset identifier.
            portfolio_value: Total portfolio value in USD.
            weight: Portfolio weight of this asset (0-1).
            confidence: Confidence level for LaR.

        Returns:
            LiquidityAssessment dataclass.
        """
        amihud = self.amihud_illiquidity(symbol)
        impact = self.estimate_market_impact(
            symbol,
            portfolio_value * weight if portfolio_value > 0 else 0,
        )
        lar = self.liquidity_at_risk(symbol, portfolio_value, weight, confidence)
        z = self.spread_z_score(symbol)
        crisis = z > self.crisis_z_threshold

        # Optimal clip for liquidating the entire position
        position_usd = portfolio_value * weight if portfolio_value > 0 else 0
        exec_info = self.optimal_execution_size(symbol, position_usd)

        assessment = LiquidityAssessment(
            symbol=symbol,
            amihud_ratio=amihud,
            market_impact_bps=impact,
            liquidity_at_risk=lar,
            spread_z_score=z,
            optimal_trade_size_usd=exec_info["clip_size_usd"],
            crisis_detected=crisis,
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            "liquidity_risk_assessed",
            symbol=symbol,
            amihud=amihud,
            impact_bps=impact,
            lar=lar,
            spread_z=z,
            crisis=crisis,
        )

        return assessment

    def assess_portfolio(
        self,
        holdings: dict[str, float],
        portfolio_value: float,
        confidence: float = 0.95,
    ) -> dict:
        """Assess liquidity risk across the entire portfolio.

        Args:
            holdings: Mapping of symbol to portfolio weight (0-1).
            portfolio_value: Total portfolio value in USD.
            confidence: Confidence level for LaR.

        Returns:
            Portfolio-level liquidity risk summary.
        """
        assessments: dict[str, LiquidityAssessment] = {}
        total_lar = Decimal("0")

        for symbol, weight in holdings.items():
            a = self.assess(symbol, portfolio_value, weight, confidence)
            assessments[symbol] = a
            total_lar += Decimal(str(a.liquidity_at_risk))

        total_lar_rounded = float(
            total_lar.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

        any_crisis = any(a.crisis_detected for a in assessments.values())
        worst_symbol = max(
            assessments, key=lambda s: assessments[s].amihud_ratio
        ) if assessments else None

        return {
            "assessments": {
                s: {
                    "amihud_ratio": a.amihud_ratio,
                    "market_impact_bps": a.market_impact_bps,
                    "liquidity_at_risk": a.liquidity_at_risk,
                    "spread_z_score": a.spread_z_score,
                    "crisis_detected": a.crisis_detected,
                    "optimal_trade_size_usd": a.optimal_trade_size_usd,
                }
                for s, a in assessments.items()
            },
            "portfolio_lar": total_lar_rounded,
            "crisis_detected": any_crisis,
            "least_liquid_asset": worst_symbol,
            "confidence": confidence,
            "portfolio_value": portfolio_value,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _average_half_spread_cost(self, symbol: str, position_usd: float) -> float:
        """Estimate half-spread cost for liquidating a position.

        Uses the average observed spread as the expected crossing cost
        for a market order.
        """
        snaps = self._spread_history.get(symbol, [])
        if not snaps or position_usd <= 0:
            return 0.0

        avg_spread_bps = float(np.mean([s.spread_bps for s in snaps[-50:]]))
        # Half spread as cost (we cross half the spread on each side)
        half_spread_fraction = (avg_spread_bps / 2) / 10_000
        return position_usd * half_spread_fraction


# Module-level instance
liquidity_risk_model = LiquidityRiskModel()
