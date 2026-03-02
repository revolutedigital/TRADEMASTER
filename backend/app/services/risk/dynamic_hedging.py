"""Dynamic hedging - delta-neutral strategies and options pricing for crypto."""

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionGreeks:
    """Option greeks for risk management."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class HedgeRecommendation:
    """A recommended hedging action."""
    action: str  # BUY_PUT, SELL_CALL, REDUCE_POSITION, ADD_SHORT, COLLAR
    symbol: str
    quantity: float
    reason: str
    estimated_cost: float
    delta_impact: float
    priority: str  # high, medium, low


class CryptoBlackScholes:
    """
    Black-Scholes model adapted for cryptocurrency options.

    Adjustments for crypto:
    - Higher base volatility (50-150% vs 15-30% for equities)
    - 24/7 trading (365 days, no weekends)
    - No dividends but staking yield considered
    - Fat-tail adjustment via volatility smile
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        d1_val = CryptoBlackScholes.d1(S, K, T, r, sigma)
        if T <= 0 or sigma <= 0:
            return 0.0
        return d1_val - sigma * math.sqrt(T)

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Cumulative normal distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Normal probability density function."""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    def price(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType = OptionType.CALL) -> float:
        """
        Price a European option using Black-Scholes.

        Args:
            S: Current price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: CALL or PUT
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1_val = self.d1(S, K, T, r, sigma)
        d2_val = self.d2(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            price = S * self.norm_cdf(d1_val) - K * math.exp(-r * T) * self.norm_cdf(d2_val)
        else:
            price = K * math.exp(-r * T) * self.norm_cdf(-d2_val) - S * self.norm_cdf(-d1_val)

        return max(0, price)

    def greeks(self, S: float, K: float, T: float, r: float, sigma: float,
               option_type: OptionType = OptionType.CALL) -> OptionGreeks:
        """Calculate option greeks."""
        if T <= 0 or sigma <= 0:
            return OptionGreeks(delta=0, gamma=0, theta=0, vega=0, rho=0)

        d1_val = self.d1(S, K, T, r, sigma)
        d2_val = self.d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)

        # Delta
        if option_type == OptionType.CALL:
            delta = self.norm_cdf(d1_val)
        else:
            delta = self.norm_cdf(d1_val) - 1

        # Gamma (same for calls and puts)
        gamma = self.norm_pdf(d1_val) / (S * sigma * sqrt_T) if S > 0 else 0

        # Theta
        common_theta = -(S * self.norm_pdf(d1_val) * sigma) / (2 * sqrt_T)
        if option_type == OptionType.CALL:
            theta = common_theta - r * K * math.exp(-r * T) * self.norm_cdf(d2_val)
        else:
            theta = common_theta + r * K * math.exp(-r * T) * self.norm_cdf(-d2_val)
        theta /= 365  # Daily theta

        # Vega
        vega = S * sqrt_T * self.norm_pdf(d1_val) / 100  # Per 1% vol change

        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * math.exp(-r * T) * self.norm_cdf(d2_val) / 100
        else:
            rho = -K * T * math.exp(-r * T) * self.norm_cdf(-d2_val) / 100

        return OptionGreeks(
            delta=round(delta, 6),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho, 4),
        )

    def implied_volatility(self, market_price: float, S: float, K: float,
                           T: float, r: float,
                           option_type: OptionType = OptionType.CALL,
                           max_iter: int = 100, tol: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson."""
        sigma = 0.5  # Initial guess for crypto (50%)

        for _ in range(max_iter):
            price = self.price(S, K, T, r, sigma, option_type)
            diff = price - market_price

            if abs(diff) < tol:
                return sigma

            # Vega for Newton-Raphson
            d1_val = self.d1(S, K, T, r, sigma)
            vega = S * math.sqrt(T) * self.norm_pdf(d1_val)

            if vega < 1e-10:
                break

            sigma -= diff / vega
            sigma = max(0.01, min(5.0, sigma))  # Bound sigma

        return sigma


class DynamicHedger:
    """
    Dynamic hedging engine for cryptocurrency portfolios.

    Strategies:
    - Delta-neutral hedging via synthetic positions
    - Protective puts for downside protection
    - Collar strategy (buy put + sell call)
    - Portfolio insurance via put spread
    - Dynamic rebalancing based on delta drift
    """

    def __init__(self):
        self.bs = CryptoBlackScholes()
        self._portfolio_delta: float = 0.0
        self._portfolio_gamma: float = 0.0
        self._hedge_history: list[dict] = []

        logger.info("dynamic_hedger_initialized")

    def analyze_portfolio_greeks(self, positions: list[dict]) -> dict:
        """
        Analyze portfolio-level greeks.

        Each position: {symbol, quantity, current_price, type: "spot"|"option",
                        option_type, strike, expiry_days, iv}
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_notional = 0.0
        position_details = []

        for pos in positions:
            qty = pos.get("quantity", 0)
            price = pos.get("current_price", 0)
            pos_type = pos.get("type", "spot")

            if pos_type == "spot":
                delta = qty  # Delta of spot = 1 per unit
                gamma = 0.0
                theta = 0.0
                vega = 0.0
                notional = qty * price
            elif pos_type == "option":
                option_type = OptionType(pos.get("option_type", "call"))
                strike = pos.get("strike", price)
                T = pos.get("expiry_days", 30) / 365
                r = pos.get("risk_free_rate", 0.05)
                iv = pos.get("iv", 0.6)

                greeks = self.bs.greeks(price, strike, T, r, iv, option_type)
                delta = greeks.delta * qty
                gamma = greeks.gamma * qty
                theta = greeks.theta * qty
                vega = greeks.vega * qty
                notional = self.bs.price(price, strike, T, r, iv, option_type) * qty
            else:
                continue

            total_delta += delta
            total_gamma += gamma
            total_theta += theta
            total_vega += vega
            total_notional += notional

            position_details.append({
                "symbol": pos.get("symbol", "UNKNOWN"),
                "type": pos_type,
                "quantity": qty,
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
                "notional": round(notional, 2),
            })

        self._portfolio_delta = total_delta
        self._portfolio_gamma = total_gamma

        return {
            "portfolio_greeks": {
                "delta": round(total_delta, 4),
                "gamma": round(total_gamma, 6),
                "theta_daily": round(total_theta, 4),
                "vega": round(total_vega, 4),
                "total_notional": round(total_notional, 2),
            },
            "positions": position_details,
            "is_delta_neutral": abs(total_delta) < 0.05 * abs(total_notional / (positions[0].get("current_price", 1) if positions else 1)),
            "delta_bias": "long" if total_delta > 0 else "short" if total_delta < 0 else "neutral",
        }

    def recommend_hedges(self, positions: list[dict],
                         risk_tolerance: str = "moderate") -> list[HedgeRecommendation]:
        """
        Generate hedging recommendations for the portfolio.

        Args:
            risk_tolerance: "conservative", "moderate", "aggressive"
        """
        analysis = self.analyze_portfolio_greeks(positions)
        portfolio_delta = analysis["portfolio_greeks"]["delta"]
        recommendations = []

        # Get main position info
        spot_positions = [p for p in positions if p.get("type") == "spot"]
        if not spot_positions:
            return []

        main_symbol = spot_positions[0].get("symbol", "BTCUSDT")
        main_price = spot_positions[0].get("current_price", 50000)
        total_spot_qty = sum(p.get("quantity", 0) for p in spot_positions)

        # Risk tolerance parameters
        hedge_params = {
            "conservative": {"hedge_ratio": 0.8, "otm_pct": 0.05, "collar_width": 0.10},
            "moderate": {"hedge_ratio": 0.5, "otm_pct": 0.10, "collar_width": 0.15},
            "aggressive": {"hedge_ratio": 0.3, "otm_pct": 0.15, "collar_width": 0.20},
        }
        params = hedge_params.get(risk_tolerance, hedge_params["moderate"])

        # 1. Protective put recommendation
        if portfolio_delta > 0:
            put_strike = main_price * (1 - params["otm_pct"])
            put_qty = total_spot_qty * params["hedge_ratio"]
            put_price = self.bs.price(main_price, put_strike, 30/365, 0.05, 0.6, OptionType.PUT)

            recommendations.append(HedgeRecommendation(
                action="BUY_PUT",
                symbol=main_symbol,
                quantity=round(put_qty, 4),
                reason=f"Protective put at {params['otm_pct']*100:.0f}% OTM to hedge {params['hedge_ratio']*100:.0f}% of position",
                estimated_cost=round(put_price * put_qty, 2),
                delta_impact=round(-put_qty * 0.3, 4),  # Approximate delta of OTM put
                priority="high" if risk_tolerance == "conservative" else "medium",
            ))

        # 2. Collar recommendation (buy put + sell call)
        if portfolio_delta > 0 and total_spot_qty > 0:
            put_strike = main_price * (1 - params["collar_width"] / 2)
            call_strike = main_price * (1 + params["collar_width"] / 2)
            collar_qty = total_spot_qty * params["hedge_ratio"]

            put_cost = self.bs.price(main_price, put_strike, 30/365, 0.05, 0.6, OptionType.PUT) * collar_qty
            call_premium = self.bs.price(main_price, call_strike, 30/365, 0.05, 0.6, OptionType.CALL) * collar_qty
            net_cost = put_cost - call_premium

            recommendations.append(HedgeRecommendation(
                action="COLLAR",
                symbol=main_symbol,
                quantity=round(collar_qty, 4),
                reason=f"Collar: Buy {put_strike:.0f} put + Sell {call_strike:.0f} call "
                       f"for net cost ${net_cost:.2f}",
                estimated_cost=round(net_cost, 2),
                delta_impact=round(-collar_qty * 0.5, 4),
                priority="medium",
            ))

        # 3. Position reduction recommendation
        if abs(portfolio_delta) > total_spot_qty * 0.7:
            reduce_pct = 0.3 if risk_tolerance == "conservative" else 0.2
            reduce_qty = total_spot_qty * reduce_pct

            recommendations.append(HedgeRecommendation(
                action="REDUCE_POSITION",
                symbol=main_symbol,
                quantity=round(reduce_qty, 4),
                reason=f"Reduce spot position by {reduce_pct*100:.0f}% to lower directional risk",
                estimated_cost=0.0,
                delta_impact=round(-reduce_qty, 4),
                priority="high" if risk_tolerance == "conservative" else "low",
            ))

        # 4. Delta-neutral adjustment
        if abs(portfolio_delta) > 0.1:
            neutralize_qty = abs(portfolio_delta)
            side = "SELL" if portfolio_delta > 0 else "BUY"

            recommendations.append(HedgeRecommendation(
                action=f"DELTA_NEUTRAL_{side}",
                symbol=main_symbol,
                quantity=round(neutralize_qty, 4),
                reason=f"{side} {neutralize_qty:.4f} units to achieve delta neutrality",
                estimated_cost=round(neutralize_qty * main_price * 0.001, 2),  # Commission only
                delta_impact=round(-portfolio_delta, 4),
                priority="medium",
            ))

        logger.info("hedge_recommendations_generated",
                    n_recommendations=len(recommendations),
                    portfolio_delta=round(portfolio_delta, 4))

        return recommendations

    def calculate_hedge_effectiveness(self, original_delta: float,
                                      hedged_delta: float,
                                      hedge_cost: float,
                                      portfolio_value: float) -> dict:
        """Calculate hedge effectiveness metrics."""
        delta_reduction = abs(original_delta) - abs(hedged_delta)
        delta_reduction_pct = delta_reduction / abs(original_delta) if original_delta != 0 else 0
        cost_pct = hedge_cost / portfolio_value if portfolio_value > 0 else 0

        return {
            "original_delta": round(original_delta, 4),
            "hedged_delta": round(hedged_delta, 4),
            "delta_reduction": round(delta_reduction, 4),
            "delta_reduction_pct": round(delta_reduction_pct * 100, 2),
            "hedge_cost": round(hedge_cost, 2),
            "cost_as_pct_portfolio": round(cost_pct * 100, 4),
            "effectiveness_ratio": round(
                delta_reduction_pct / (cost_pct + 1e-10), 2
            ),
        }

    def get_dashboard(self, positions: list[dict]) -> dict:
        """Get hedging dashboard data."""
        analysis = self.analyze_portfolio_greeks(positions)
        recommendations = self.recommend_hedges(positions)

        return {
            "portfolio_analysis": analysis,
            "recommendations": [
                {
                    "action": r.action,
                    "symbol": r.symbol,
                    "quantity": r.quantity,
                    "reason": r.reason,
                    "estimated_cost": r.estimated_cost,
                    "delta_impact": r.delta_impact,
                    "priority": r.priority,
                }
                for r in recommendations
            ],
            "n_recommendations": len(recommendations),
            "high_priority": sum(1 for r in recommendations if r.priority == "high"),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Module-level instances
crypto_bs = CryptoBlackScholes()
dynamic_hedger = DynamicHedger()
