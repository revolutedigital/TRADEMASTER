"""Counterparty risk monitoring - exchange health and proof-of-reserves tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class ExchangeHealthStatus(str, Enum):
    HEALTHY = "healthy"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ExchangeProfile:
    """Profile of a cryptocurrency exchange for risk assessment."""
    name: str
    tier: int  # 1=top, 2=major, 3=mid, 4=small
    regulatory_status: str  # "regulated", "partially_regulated", "unregulated"
    proof_of_reserves: bool = False
    last_audit_date: str | None = None
    insurance_fund_usd: float = 0.0
    daily_volume_usd: float = 0.0
    years_operating: int = 0
    incident_history: list[dict] = field(default_factory=list)
    health_score: float = 100.0
    status: ExchangeHealthStatus = ExchangeHealthStatus.UNKNOWN


# Pre-configured exchange profiles
EXCHANGE_PROFILES: dict[str, dict] = {
    "binance": {
        "tier": 1,
        "regulatory_status": "partially_regulated",
        "proof_of_reserves": True,
        "insurance_fund_usd": 1_000_000_000,
        "daily_volume_usd": 20_000_000_000,
        "years_operating": 8,
    },
    "coinbase": {
        "tier": 1,
        "regulatory_status": "regulated",
        "proof_of_reserves": True,
        "insurance_fund_usd": 500_000_000,
        "daily_volume_usd": 5_000_000_000,
        "years_operating": 13,
    },
    "kraken": {
        "tier": 1,
        "regulatory_status": "regulated",
        "proof_of_reserves": True,
        "insurance_fund_usd": 200_000_000,
        "daily_volume_usd": 2_000_000_000,
        "years_operating": 13,
    },
    "bybit": {
        "tier": 2,
        "regulatory_status": "partially_regulated",
        "proof_of_reserves": True,
        "insurance_fund_usd": 300_000_000,
        "daily_volume_usd": 8_000_000_000,
        "years_operating": 6,
    },
    "okx": {
        "tier": 2,
        "regulatory_status": "partially_regulated",
        "proof_of_reserves": True,
        "insurance_fund_usd": 400_000_000,
        "daily_volume_usd": 6_000_000_000,
        "years_operating": 7,
    },
}

# Historical incident database
KNOWN_INCIDENTS = [
    {"exchange": "ftx", "date": "2022-11-11", "type": "bankruptcy",
     "severity": 10, "description": "Complete collapse, $8B customer funds lost"},
    {"exchange": "mtgox", "date": "2014-02-28", "type": "hack",
     "severity": 10, "description": "850,000 BTC stolen"},
    {"exchange": "bitfinex", "date": "2016-08-02", "type": "hack",
     "severity": 7, "description": "119,756 BTC stolen, later reimbursed"},
    {"exchange": "binance", "date": "2019-05-07", "type": "hack",
     "severity": 4, "description": "7,000 BTC stolen, covered by SAFU fund"},
    {"exchange": "kucoin", "date": "2020-09-25", "type": "hack",
     "severity": 5, "description": "$280M stolen, 84% recovered"},
]


class CounterpartyRiskMonitor:
    """
    Monitor counterparty risk for cryptocurrency exchanges.

    Evaluates:
    - Exchange health scoring (0-100)
    - Proof of reserves verification status
    - Regulatory compliance
    - Insurance fund adequacy
    - Historical incident analysis
    - Exposure concentration across exchanges
    - Early warning indicators
    """

    def __init__(self):
        self.exchanges: dict[str, ExchangeProfile] = {}
        self._exposure: dict[str, float] = {}  # exchange -> USD exposure
        self._alerts: list[dict] = []

        # Initialize known exchanges
        for name, config in EXCHANGE_PROFILES.items():
            self.exchanges[name] = ExchangeProfile(name=name, **config)

        logger.info("counterparty_risk_monitor_initialized",
                    exchanges=len(self.exchanges))

    def assess_exchange(self, exchange_name: str) -> dict:
        """Comprehensive risk assessment of a single exchange."""
        name = exchange_name.lower()
        profile = self.exchanges.get(name)

        if not profile:
            return {
                "exchange": exchange_name,
                "status": ExchangeHealthStatus.UNKNOWN.value,
                "health_score": 0,
                "message": "Exchange not in database",
                "recommendation": "DO NOT USE - unverified exchange",
            }

        score = 100.0
        risk_factors = []

        # 1. Tier scoring (0-15 points deduction)
        tier_deduction = {1: 0, 2: 5, 3: 10, 4: 15}
        score -= tier_deduction.get(profile.tier, 15)
        if profile.tier >= 3:
            risk_factors.append(f"Lower tier exchange (tier {profile.tier})")

        # 2. Regulatory status (0-20 points deduction)
        reg_deduction = {"regulated": 0, "partially_regulated": 10, "unregulated": 20}
        score -= reg_deduction.get(profile.regulatory_status, 20)
        if profile.regulatory_status == "unregulated":
            risk_factors.append("Not regulated in any major jurisdiction")

        # 3. Proof of reserves (0-15 points deduction)
        if not profile.proof_of_reserves:
            score -= 15
            risk_factors.append("No proof of reserves")

        # 4. Insurance fund (0-10 points deduction)
        if profile.insurance_fund_usd == 0:
            score -= 10
            risk_factors.append("No insurance fund")
        elif profile.insurance_fund_usd < 100_000_000:
            score -= 5
            risk_factors.append("Small insurance fund (<$100M)")

        # 5. Operating history (0-10 points deduction)
        if profile.years_operating < 2:
            score -= 10
            risk_factors.append("Very new exchange (<2 years)")
        elif profile.years_operating < 5:
            score -= 5
            risk_factors.append("Relatively new exchange (<5 years)")

        # 6. Volume (0-10 points deduction)
        if profile.daily_volume_usd < 100_000_000:
            score -= 10
            risk_factors.append("Low daily volume (<$100M)")
        elif profile.daily_volume_usd < 1_000_000_000:
            score -= 5

        # 7. Incident history (0-20 points deduction)
        exchange_incidents = [i for i in KNOWN_INCIDENTS if i["exchange"] == name]
        for incident in exchange_incidents:
            score -= min(incident["severity"] * 2, 20)
            risk_factors.append(f"Historical incident: {incident['description']}")

        score = max(0, min(100, score))
        profile.health_score = score

        # Determine status
        if score >= 80:
            status = ExchangeHealthStatus.HEALTHY
        elif score >= 60:
            status = ExchangeHealthStatus.CAUTION
        elif score >= 40:
            status = ExchangeHealthStatus.WARNING
        else:
            status = ExchangeHealthStatus.CRITICAL

        profile.status = status

        # Recommendation
        if status == ExchangeHealthStatus.CRITICAL:
            recommendation = "AVOID - High counterparty risk. Move funds immediately."
        elif status == ExchangeHealthStatus.WARNING:
            recommendation = "REDUCE EXPOSURE - Limit funds on this exchange."
        elif status == ExchangeHealthStatus.CAUTION:
            recommendation = "MONITOR - Keep only necessary trading funds."
        else:
            recommendation = "ACCEPTABLE - Standard risk management applies."

        return {
            "exchange": name,
            "status": status.value,
            "health_score": round(score, 1),
            "tier": profile.tier,
            "regulatory_status": profile.regulatory_status,
            "proof_of_reserves": profile.proof_of_reserves,
            "insurance_fund_usd": profile.insurance_fund_usd,
            "daily_volume_usd": profile.daily_volume_usd,
            "years_operating": profile.years_operating,
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "incidents": exchange_incidents,
        }

    def assess_all_exchanges(self) -> dict:
        """Assess all tracked exchanges."""
        assessments = {}
        for name in self.exchanges:
            assessments[name] = self.assess_exchange(name)

        # Sort by health score
        sorted_exchanges = sorted(
            assessments.values(),
            key=lambda x: x["health_score"],
            reverse=True,
        )

        return {
            "assessments": assessments,
            "ranking": [
                {"exchange": e["exchange"], "score": e["health_score"], "status": e["status"]}
                for e in sorted_exchanges
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def update_exposure(self, exchange: str, amount_usd: float) -> None:
        """Update USD exposure on an exchange."""
        self._exposure[exchange.lower()] = amount_usd

    def analyze_exposure(self) -> dict:
        """Analyze exposure concentration across exchanges."""
        total = sum(self._exposure.values())
        if total == 0:
            return {"total_exposure": 0, "exchanges": {}, "risk_level": "none"}

        analysis = {}
        max_concentration = 0.0

        for exchange, amount in self._exposure.items():
            concentration = amount / total
            max_concentration = max(max_concentration, concentration)

            assessment = self.assess_exchange(exchange)
            risk_weighted = amount * (1 - assessment["health_score"] / 100)

            analysis[exchange] = {
                "exposure_usd": amount,
                "concentration_pct": round(concentration * 100, 1),
                "health_score": assessment["health_score"],
                "risk_weighted_exposure": round(risk_weighted, 2),
                "status": assessment["status"],
            }

        # Overall risk level
        if max_concentration > 0.8:
            risk_level = "critical"
            suggestion = "Extremely concentrated - distribute across 3+ exchanges"
        elif max_concentration > 0.5:
            risk_level = "high"
            suggestion = "High concentration - consider adding another exchange"
        elif max_concentration > 0.3:
            risk_level = "moderate"
            suggestion = "Acceptable concentration levels"
        else:
            risk_level = "low"
            suggestion = "Well diversified across exchanges"

        total_risk_weighted = sum(e["risk_weighted_exposure"] for e in analysis.values())

        return {
            "total_exposure": round(total, 2),
            "total_risk_weighted": round(total_risk_weighted, 2),
            "max_concentration_pct": round(max_concentration * 100, 1),
            "risk_level": risk_level,
            "suggestion": suggestion,
            "exchanges": analysis,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def check_early_warnings(self) -> list[dict]:
        """Check for early warning signs across exchanges."""
        warnings = []

        for name, profile in self.exchanges.items():
            assessment = self.assess_exchange(name)

            # Warning: exchange in critical state with user exposure
            exposure = self._exposure.get(name, 0)
            if assessment["status"] == "critical" and exposure > 0:
                warnings.append({
                    "severity": "critical",
                    "exchange": name,
                    "message": f"CRITICAL: {name} health score {assessment['health_score']}, "
                              f"you have ${exposure:,.0f} exposure",
                    "action": "Withdraw funds immediately",
                })

            # Warning: exchange without proof of reserves
            if not profile.proof_of_reserves and exposure > 0:
                warnings.append({
                    "severity": "high",
                    "exchange": name,
                    "message": f"{name} has no proof of reserves with ${exposure:,.0f} exposure",
                    "action": "Verify exchange solvency, consider reducing exposure",
                })

            # Warning: unregulated exchange with significant exposure
            if profile.regulatory_status == "unregulated" and exposure > 10000:
                warnings.append({
                    "severity": "high",
                    "exchange": name,
                    "message": f"Unregulated exchange {name} with ${exposure:,.0f} exposure",
                    "action": "Move funds to regulated exchange",
                })

        self._alerts = warnings
        return warnings

    def get_dashboard(self) -> dict:
        """Get counterparty risk dashboard data."""
        return {
            "exchange_assessments": self.assess_all_exchanges(),
            "exposure_analysis": self.analyze_exposure(),
            "early_warnings": self.check_early_warnings(),
            "summary": {
                "total_exchanges_monitored": len(self.exchanges),
                "healthy_exchanges": sum(
                    1 for e in self.exchanges.values()
                    if e.status == ExchangeHealthStatus.HEALTHY
                ),
                "active_warnings": len(self._alerts),
            },
        }


# Module-level instance
counterparty_monitor = CounterpartyRiskMonitor()
