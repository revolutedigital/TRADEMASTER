"""Ensemble model voting with configurable weights."""
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VotingResult:
    action: str  # BUY, HOLD, SELL
    weighted_score: float
    confidence: float
    agreement_ratio: float  # How many models agree
    individual_votes: dict[str, str]


class EnsembleVoter:
    """Weighted voting across multiple model predictions."""

    DEFAULT_WEIGHTS = {
        "lstm": 0.35,
        "xgboost": 0.35,
        "transformer": 0.20,
        "sentiment": 0.10,
    }

    def vote(self, predictions: list[dict], weights: dict[str, float] | None = None) -> VotingResult:
        """Combine multiple model predictions via weighted voting.

        Each prediction dict should have: model, action, score, confidence
        """
        w = weights or self.DEFAULT_WEIGHTS

        if not predictions:
            return VotingResult(action="HOLD", weighted_score=0.0, confidence=0.0, agreement_ratio=0.0, individual_votes={})

        # Calculate weighted score (-1 SELL, 0 HOLD, +1 BUY)
        action_map = {"SELL": -1.0, "HOLD": 0.0, "BUY": 1.0}
        total_weight = 0.0
        weighted_score = 0.0
        individual_votes = {}

        for pred in predictions:
            model = pred.get("model", "unknown")
            action = pred.get("action", "HOLD")
            score = pred.get("score", 0.0)
            model_weight = w.get(model, 0.1)

            weighted_score += action_map.get(action, 0.0) * score * model_weight
            total_weight += model_weight
            individual_votes[model] = action

        if total_weight > 0:
            weighted_score /= total_weight

        # Conservative confidence: minimum across models
        confidence = min(p.get("confidence", 0.5) for p in predictions)

        # Agreement ratio
        actions = [p.get("action", "HOLD") for p in predictions]
        most_common = max(set(actions), key=actions.count)
        agreement = actions.count(most_common) / len(actions)

        # Final action
        if weighted_score > 0.2:
            action = "BUY"
        elif weighted_score < -0.2:
            action = "SELL"
        else:
            action = "HOLD"

        return VotingResult(
            action=action,
            weighted_score=round(weighted_score, 4),
            confidence=round(confidence, 4),
            agreement_ratio=round(agreement, 2),
            individual_votes=individual_votes,
        )


ensemble_voter = EnsembleVoter()
