"""Neuro-symbolic trading engine: interpretable rules with neural weight adaptation."""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from app.core.logging import get_logger

logger = get_logger(__name__)


class RuleAction(str, Enum):
    """Actions a trading rule can recommend."""
    CONFIDENCE_BOOST = "confidence_boost"
    CONFIDENCE_REDUCE = "confidence_reduce"
    SIGNAL_BUY = "signal_buy"
    SIGNAL_SELL = "signal_sell"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    STOP_LOSS = "stop_loss"


@dataclass
class TradingRule:
    """An interpretable trading rule with adaptive weight.

    Attributes:
        name: Human-readable rule identifier.
        condition: Callable that takes a feature dict and returns True/False.
        condition_desc: Human-readable description of the condition logic.
        action: The action this rule recommends when activated.
        weight: Neural-adjusted importance weight (0-1 range, updated by learning).
        confidence: How reliable this rule has been historically (0-1).
        activation_count: Total number of times this rule has fired.
        success_count: Times the rule fired and the predicted outcome was correct.
    """
    name: str
    condition: Callable[[dict], bool]
    condition_desc: str
    action: RuleAction
    weight: float = 0.5
    confidence: float = 0.5
    activation_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.success_count / self.activation_count


@dataclass
class RuleActivation:
    """Record of a single rule firing during evaluation."""
    rule_name: str
    condition_desc: str
    action: RuleAction
    weight: float
    confidence: float
    contribution: float  # weight * confidence, signed by action direction


@dataclass
class DecisionExplanation:
    """Full explanation of a neuro-symbolic decision."""
    final_score: float  # Aggregated score from all activated rules
    recommended_action: str  # BUY, SELL, HOLD
    activated_rules: list[RuleActivation]
    total_rules_checked: int
    activation_ratio: float  # Fraction of rules that fired


class NeuroSymbolicEngine:
    """Combines interpretable rule-based reasoning with neural weight adaptation.

    The symbolic component is a set of TradingRule objects with human-readable
    conditions.  The neural component maintains and updates rule weights using
    an online gradient-free learning signal derived from trading outcomes.

    Architecture:
    - Rule Engine: evaluates conditions against current market features.
    - Neural Adapter: adjusts rule weights toward rules that historically
      correlate with profitable outcomes.
    - Explainability: every decision is fully decomposed into rule activations.
    - Auto-Discovery: mines new candidate rules from feature statistics.
    """

    # Action direction mapping for score aggregation
    _ACTION_DIRECTION: dict[RuleAction, float] = {
        RuleAction.CONFIDENCE_BOOST: 0.5,
        RuleAction.CONFIDENCE_REDUCE: -0.5,
        RuleAction.SIGNAL_BUY: 1.0,
        RuleAction.SIGNAL_SELL: -1.0,
        RuleAction.INCREASE_POSITION: 0.7,
        RuleAction.DECREASE_POSITION: -0.7,
        RuleAction.STOP_LOSS: -1.0,
    }

    def __init__(self, learning_rate: float = 0.05, decay: float = 0.995):
        """Initialize the engine.

        Args:
            learning_rate: Step size for neural weight updates.
            decay: Multiplicative decay applied each update to prevent
                   overfitting to recent outcomes.
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.rules: list[TradingRule] = []
        self._weight_history: list[dict[str, float]] = []
        self._init_default_rules()

    # ------------------------------------------------------------------
    # Default rule set
    # ------------------------------------------------------------------

    def _init_default_rules(self) -> None:
        """Register a curated set of well-known technical trading heuristics."""
        self.add_rule(TradingRule(
            name="rsi_oversold",
            condition=lambda f: f.get("rsi", 50) < 30,
            condition_desc="RSI < 30 (oversold)",
            action=RuleAction.SIGNAL_BUY,
            weight=0.6,
        ))
        self.add_rule(TradingRule(
            name="rsi_overbought",
            condition=lambda f: f.get("rsi", 50) > 70,
            condition_desc="RSI > 70 (overbought)",
            action=RuleAction.SIGNAL_SELL,
            weight=0.6,
        ))
        self.add_rule(TradingRule(
            name="volume_spike_buy",
            condition=lambda f: (
                f.get("volume_ratio", 1.0) > 2.0
                and f.get("price_change_pct", 0) > 0
            ),
            condition_desc="Volume > 2x average AND price rising",
            action=RuleAction.CONFIDENCE_BOOST,
            weight=0.5,
        ))
        self.add_rule(TradingRule(
            name="volume_spike_sell",
            condition=lambda f: (
                f.get("volume_ratio", 1.0) > 2.0
                and f.get("price_change_pct", 0) < 0
            ),
            condition_desc="Volume > 2x average AND price falling",
            action=RuleAction.CONFIDENCE_REDUCE,
            weight=0.5,
        ))
        self.add_rule(TradingRule(
            name="macd_bullish_cross",
            condition=lambda f: (
                f.get("macd", 0) > f.get("macd_signal", 0)
                and f.get("macd_prev", 0) <= f.get("macd_signal_prev", 0)
            ),
            condition_desc="MACD crosses above signal line",
            action=RuleAction.SIGNAL_BUY,
            weight=0.55,
        ))
        self.add_rule(TradingRule(
            name="macd_bearish_cross",
            condition=lambda f: (
                f.get("macd", 0) < f.get("macd_signal", 0)
                and f.get("macd_prev", 0) >= f.get("macd_signal_prev", 0)
            ),
            condition_desc="MACD crosses below signal line",
            action=RuleAction.SIGNAL_SELL,
            weight=0.55,
        ))
        self.add_rule(TradingRule(
            name="bollinger_lower_touch",
            condition=lambda f: f.get("price", 0) <= f.get("bb_lower", float("-inf")),
            condition_desc="Price at or below lower Bollinger Band",
            action=RuleAction.SIGNAL_BUY,
            weight=0.45,
        ))
        self.add_rule(TradingRule(
            name="bollinger_upper_touch",
            condition=lambda f: f.get("price", 0) >= f.get("bb_upper", float("inf")),
            condition_desc="Price at or above upper Bollinger Band",
            action=RuleAction.SIGNAL_SELL,
            weight=0.45,
        ))
        self.add_rule(TradingRule(
            name="strong_downtrend_stop",
            condition=lambda f: f.get("price_change_pct", 0) < -5.0,
            condition_desc="Price dropped more than 5% (stop-loss trigger)",
            action=RuleAction.STOP_LOSS,
            weight=0.7,
        ))
        self.add_rule(TradingRule(
            name="rsi_volume_oversold",
            condition=lambda f: (
                f.get("rsi", 50) < 30
                and f.get("volume_ratio", 1.0) > 1.5
            ),
            condition_desc="RSI < 30 AND volume_spike (high-conviction oversold)",
            action=RuleAction.INCREASE_POSITION,
            weight=0.55,
        ))

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: TradingRule) -> None:
        """Register a new trading rule."""
        existing_names = {r.name for r in self.rules}
        if rule.name in existing_names:
            logger.warning("Rule '%s' already exists; skipping", rule.name)
            return
        self.rules.append(rule)
        logger.debug("Added rule: %s", rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        for i, r in enumerate(self.rules):
            if r.name == name:
                self.rules.pop(i)
                logger.info("Removed rule: %s", name)
                return True
        return False

    # ------------------------------------------------------------------
    # Evaluation (symbolic inference)
    # ------------------------------------------------------------------

    def evaluate(self, features: dict) -> DecisionExplanation:
        """Evaluate all rules against current market features.

        Returns an explained decision with every rule's contribution.

        Args:
            features: Dictionary of current market indicators (e.g. rsi, macd,
                      volume_ratio, price, bb_lower, bb_upper, etc.).

        Returns:
            DecisionExplanation with aggregated score, action, and per-rule detail.
        """
        activations: list[RuleActivation] = []
        total_score = 0.0
        total_weight = 0.0

        for rule in self.rules:
            try:
                fired = rule.condition(features)
            except Exception as exc:
                logger.warning("Rule '%s' raised %s: %s", rule.name, type(exc).__name__, exc)
                continue

            if not fired:
                continue

            direction = self._ACTION_DIRECTION.get(rule.action, 0.0)
            contribution = rule.weight * rule.confidence * direction
            total_score += contribution
            total_weight += rule.weight

            activations.append(RuleActivation(
                rule_name=rule.name,
                condition_desc=rule.condition_desc,
                action=rule.action,
                weight=round(rule.weight, 4),
                confidence=round(rule.confidence, 4),
                contribution=round(contribution, 4),
            ))

        # Normalise score to [-1, 1]
        if total_weight > 0:
            normalised = np.clip(total_score / total_weight, -1.0, 1.0)
        else:
            normalised = 0.0

        if normalised > 0.15:
            action = "BUY"
        elif normalised < -0.15:
            action = "SELL"
        else:
            action = "HOLD"

        activation_ratio = len(activations) / max(len(self.rules), 1)

        explanation = DecisionExplanation(
            final_score=round(float(normalised), 4),
            recommended_action=action,
            activated_rules=sorted(activations, key=lambda a: abs(a.contribution), reverse=True),
            total_rules_checked=len(self.rules),
            activation_ratio=round(activation_ratio, 3),
        )
        logger.debug(
            "Evaluated %d rules, %d activated -> %s (score=%.4f)",
            len(self.rules), len(activations), action, normalised,
        )
        return explanation

    # ------------------------------------------------------------------
    # Neural weight adaptation
    # ------------------------------------------------------------------

    def update_weights(self, activated_rule_names: list[str], outcome: float) -> None:
        """Update rule weights based on a trading outcome.

        Uses a simple reward-modulated Hebbian-style update:
        - Rules that fired before a positive outcome get a weight increase.
        - Rules that fired before a negative outcome get a weight decrease.
        - All weights are then decayed and re-normalised to [0.05, 1.0].

        Args:
            activated_rule_names: Names of rules that were active for this trade.
            outcome: Realised return of the trade (positive = profit, negative = loss).
        """
        reward = np.tanh(outcome * 10)  # Squash to [-1, 1]

        name_to_rule = {r.name: r for r in self.rules}
        for name in activated_rule_names:
            rule = name_to_rule.get(name)
            if rule is None:
                continue
            rule.activation_count += 1
            if outcome > 0:
                rule.success_count += 1

            direction = self._ACTION_DIRECTION.get(rule.action, 0.0)
            # If the rule's direction matched the outcome sign, reinforce
            alignment = direction * reward
            delta = self.learning_rate * alignment
            rule.weight = float(np.clip(rule.weight + delta, 0.05, 1.0))

            # Update confidence using exponential moving average of success rate
            rule.confidence = 0.9 * rule.confidence + 0.1 * rule.success_rate

        # Apply decay to prevent runaway weights
        for rule in self.rules:
            rule.weight = float(np.clip(rule.weight * self.decay, 0.05, 1.0))

        # Record snapshot
        self._weight_history.append({r.name: r.weight for r in self.rules})
        logger.info(
            "Updated weights for %d activated rules (outcome=%.4f, reward=%.4f)",
            len(activated_rule_names), outcome, reward,
        )

    def get_weight_history(self) -> list[dict[str, float]]:
        """Return the full history of weight snapshots."""
        return self._weight_history

    # ------------------------------------------------------------------
    # Auto-discovery of new rules
    # ------------------------------------------------------------------

    def discover_rules(
        self,
        feature_history: list[dict],
        outcome_history: list[float],
        min_samples: int = 30,
        min_correlation: float = 0.15,
    ) -> list[TradingRule]:
        """Automatically discover candidate rules from historical data patterns.

        Scans each numeric feature for threshold-based rules where extreme
        values (top/bottom quintile) correlate with subsequent positive or
        negative outcomes.

        Args:
            feature_history: List of feature dicts, one per time step.
            outcome_history: Corresponding future returns.
            min_samples: Minimum activations required to consider a rule.
            min_correlation: Minimum abs(correlation) to accept.

        Returns:
            List of newly discovered TradingRule objects (also added to engine).
        """
        if len(feature_history) != len(outcome_history):
            logger.error("Feature and outcome history length mismatch")
            return []

        n = len(feature_history)
        if n < min_samples:
            logger.warning("Not enough samples (%d) for rule discovery", n)
            return []

        outcomes = np.array(outcome_history, dtype=np.float64)

        # Collect all numeric feature keys
        all_keys: set[str] = set()
        for fdict in feature_history:
            for k, v in fdict.items():
                if isinstance(v, (int, float)):
                    all_keys.add(k)

        existing_names = {r.name for r in self.rules}
        discovered: list[TradingRule] = []

        for key in sorted(all_keys):
            values = np.array(
                [f.get(key, np.nan) for f in feature_history], dtype=np.float64,
            )
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() < min_samples:
                continue

            valid_vals = values[valid_mask]
            valid_outcomes = outcomes[valid_mask]

            # Test low-threshold rule (bottom quintile)
            low_thresh = float(np.percentile(valid_vals, 20))
            low_mask = valid_vals <= low_thresh
            if low_mask.sum() >= min_samples:
                low_mean = float(np.mean(valid_outcomes[low_mask]))
                overall_mean = float(np.mean(valid_outcomes))
                if abs(low_mean - overall_mean) > 0:
                    # Point-biserial correlation
                    corr = self._point_biserial(low_mask, valid_outcomes)
                    if abs(corr) >= min_correlation:
                        rule_name = f"auto_{key}_low"
                        if rule_name not in existing_names:
                            action = (
                                RuleAction.SIGNAL_BUY if low_mean > overall_mean
                                else RuleAction.SIGNAL_SELL
                            )
                            threshold = low_thresh
                            rule = TradingRule(
                                name=rule_name,
                                condition=self._make_threshold_condition(key, "<=", threshold),
                                condition_desc=f"{key} <= {threshold:.4f} (auto-discovered)",
                                action=action,
                                weight=min(abs(corr), 0.6),
                                confidence=min(abs(corr) * 2, 0.8),
                            )
                            discovered.append(rule)
                            existing_names.add(rule_name)

            # Test high-threshold rule (top quintile)
            high_thresh = float(np.percentile(valid_vals, 80))
            high_mask = valid_vals >= high_thresh
            if high_mask.sum() >= min_samples:
                high_mean = float(np.mean(valid_outcomes[high_mask]))
                overall_mean = float(np.mean(valid_outcomes))
                if abs(high_mean - overall_mean) > 0:
                    corr = self._point_biserial(high_mask, valid_outcomes)
                    if abs(corr) >= min_correlation:
                        rule_name = f"auto_{key}_high"
                        if rule_name not in existing_names:
                            action = (
                                RuleAction.SIGNAL_BUY if high_mean > overall_mean
                                else RuleAction.SIGNAL_SELL
                            )
                            threshold = high_thresh
                            rule = TradingRule(
                                name=rule_name,
                                condition=self._make_threshold_condition(key, ">=", threshold),
                                condition_desc=f"{key} >= {threshold:.4f} (auto-discovered)",
                                action=action,
                                weight=min(abs(corr), 0.6),
                                confidence=min(abs(corr) * 2, 0.8),
                            )
                            discovered.append(rule)
                            existing_names.add(rule_name)

        # Register discovered rules
        for rule in discovered:
            self.add_rule(rule)

        logger.info("Auto-discovered %d new rules from %d samples", len(discovered), n)
        return discovered

    @staticmethod
    def _make_threshold_condition(
        feature_key: str, operator: str, threshold: float,
    ) -> Callable[[dict], bool]:
        """Create a threshold condition closure.

        Uses default arguments to capture values at creation time, avoiding
        late-binding closure issues.
        """
        if operator == "<=":
            def condition(f: dict, _k: str = feature_key, _t: float = threshold) -> bool:
                return f.get(_k, _t + 1) <= _t
        elif operator == ">=":
            def condition(f: dict, _k: str = feature_key, _t: float = threshold) -> bool:
                return f.get(_k, _t - 1) >= _t
        else:
            def condition(f: dict) -> bool:
                return False
        return condition

    @staticmethod
    def _point_biserial(binary: np.ndarray, continuous: np.ndarray) -> float:
        """Compute point-biserial correlation between a boolean mask and continuous values."""
        binary = binary.astype(np.float64)
        n = len(binary)
        n1 = binary.sum()
        n0 = n - n1
        if n0 == 0 or n1 == 0 or n < 3:
            return 0.0
        mean1 = np.mean(continuous[binary > 0.5])
        mean0 = np.mean(continuous[binary < 0.5])
        std_all = np.std(continuous, ddof=1)
        if std_all == 0:
            return 0.0
        return float((mean1 - mean0) / std_all * np.sqrt(n1 * n0 / (n * n)))

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_rule_summary(self) -> list[dict]:
        """Return a summary of all rules and their current state."""
        return [
            {
                "name": r.name,
                "condition": r.condition_desc,
                "action": r.action.value,
                "weight": round(r.weight, 4),
                "confidence": round(r.confidence, 4),
                "activations": r.activation_count,
                "success_rate": round(r.success_rate, 4),
            }
            for r in self.rules
        ]

    def get_top_rules(self, n: int = 5) -> list[dict]:
        """Return the top-N rules ranked by weight * confidence."""
        ranked = sorted(self.rules, key=lambda r: r.weight * r.confidence, reverse=True)
        return [
            {
                "name": r.name,
                "condition": r.condition_desc,
                "action": r.action.value,
                "score": round(r.weight * r.confidence, 4),
            }
            for r in ranked[:n]
        ]


neuro_symbolic_engine = NeuroSymbolicEngine()
