"""Adaptive ensemble voting: technical + ML + regime with dynamic weights.

Weight distribution adapts based on market regime:
- Bull/Bear with high confidence → ML gets more weight (trending, ML excels)
- Sideways → Technical gets more weight (mean-reversion indicators)
- High volatility → Regime signal gets more weight (risk-off bias)

Also provides regime bias: a directional nudge based on detected regime.
"""

from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VotingResult:
    action: str  # BUY, HOLD, SELL
    weighted_score: float
    confidence: float
    agreement_ratio: float  # How many sources agree on direction
    individual_votes: dict[str, str]
    regime_bias: float  # Directional nudge from regime


# Regime-adaptive weight presets: (technical, ml, regime_bias)
_REGIME_WEIGHTS = {
    "bull": (0.30, 0.50, 0.20),     # Trending → ML excels
    "bear": (0.30, 0.45, 0.25),     # Trending down → ML + regime caution
    "sideways": (0.50, 0.30, 0.20), # Range-bound → technical indicators better
}

# Volatility adjustments: high vol → reduce ML trust, increase regime weight
_VOL_ADJUSTMENTS = {
    "low": (0.0, 0.05, -0.05),      # Low vol → trust ML more
    "normal": (0.0, 0.0, 0.0),       # No adjustment
    "high": (0.05, -0.10, 0.05),     # High vol → less ML, more caution
}

# Regime directional bias (-1 to +1)
_REGIME_BIAS = {
    "bull": 0.15,     # Slight bullish nudge
    "bear": -0.15,    # Slight bearish nudge
    "sideways": 0.0,  # No directional bias
}


class EnsembleVoter:
    """Adaptive weighted voting across technical, ML, and regime signals."""

    def vote(
        self,
        predictions: list[dict],
        regime: str = "sideways",
        volatility: str = "normal",
        regime_confidence: float = 0.5,
    ) -> VotingResult:
        """Combine multiple signal sources via regime-adaptive weighted voting.

        Each prediction dict: {model, action, score, confidence}
        model should be one of: "technical", "ml", "xgboost", "lstm", etc.

        Returns VotingResult with final action, score, and diagnostics.
        """
        if not predictions:
            return VotingResult(
                action="HOLD", weighted_score=0.0, confidence=0.0,
                agreement_ratio=0.0, individual_votes={}, regime_bias=0.0,
            )

        # 1. Get base weights for this regime
        tech_w, ml_w, regime_w = _REGIME_WEIGHTS.get(regime, (0.40, 0.40, 0.20))

        # 2. Apply volatility adjustments
        vol_adj = _VOL_ADJUSTMENTS.get(volatility, (0.0, 0.0, 0.0))
        tech_w = max(0.1, tech_w + vol_adj[0])
        ml_w = max(0.1, ml_w + vol_adj[1])
        regime_w = max(0.05, regime_w + vol_adj[2])

        # Normalize weights
        total_w = tech_w + ml_w + regime_w
        tech_w /= total_w
        ml_w /= total_w
        regime_w /= total_w

        # 3. Classify predictions into technical vs ML
        action_map = {"SELL": -1.0, "HOLD": 0.0, "BUY": 1.0}
        tech_scores = []
        ml_scores = []
        individual_votes = {}

        for pred in predictions:
            model = pred.get("model", "unknown")
            action = pred.get("action", "HOLD")
            score = pred.get("score", 0.0)
            conf = pred.get("confidence", 0.5)
            direction = action_map.get(action, 0.0)
            weighted_val = direction * abs(score) * conf

            individual_votes[model] = action

            if model == "technical":
                tech_scores.append(weighted_val)
            else:
                ml_scores.append(weighted_val)

        # 4. Aggregate per-group scores
        tech_avg = sum(tech_scores) / len(tech_scores) if tech_scores else 0.0
        ml_avg = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0

        # 5. Regime directional bias (scaled by regime confidence)
        bias = _REGIME_BIAS.get(regime, 0.0) * regime_confidence

        # 6. Final weighted score
        weighted_score = tech_avg * tech_w + ml_avg * ml_w + bias * regime_w

        # 7. Confidence: conservative (min of all sources that voted)
        all_confs = [p.get("confidence", 0.5) for p in predictions]
        confidence = min(all_confs) if all_confs else 0.0

        # 8. Agreement: how many sources point in same direction
        all_directions = []
        for p in predictions:
            d = action_map.get(p.get("action", "HOLD"), 0.0)
            if d != 0:
                all_directions.append(d)
        if bias != 0:
            all_directions.append(1.0 if bias > 0 else -1.0)

        if all_directions:
            dominant = max(set(int(d) for d in all_directions), key=lambda x: all_directions.count(x))
            agreement = sum(1 for d in all_directions if int(d) == dominant) / len(all_directions)
        else:
            agreement = 0.0

        # 9. Final action
        if weighted_score > 0.15:
            final_action = "BUY"
        elif weighted_score < -0.15:
            final_action = "SELL"
        else:
            final_action = "HOLD"

        result = VotingResult(
            action=final_action,
            weighted_score=round(weighted_score, 4),
            confidence=round(confidence, 4),
            agreement_ratio=round(agreement, 2),
            individual_votes=individual_votes,
            regime_bias=round(bias, 4),
        )

        logger.debug(
            "ensemble_vote",
            action=final_action,
            score=result.weighted_score,
            regime=regime,
            volatility=volatility,
            weights={"tech": round(tech_w, 2), "ml": round(ml_w, 2), "regime": round(regime_w, 2)},
            votes=individual_votes,
        )

        return result


ensemble_voter = EnsembleVoter()
