"""A/B testing framework for comparing ML model performance."""

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ABTestResult:
    champion_trades: int = 0
    challenger_trades: int = 0
    champion_pnl: float = 0.0
    challenger_pnl: float = 0.0
    champion_win_rate: float = 0.0
    challenger_win_rate: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ABTestManager:
    """Routes predictions between champion and challenger models."""

    def __init__(self, challenger_traffic_pct: float = 0.2):
        self.challenger_traffic_pct = challenger_traffic_pct
        self.champion_model = None
        self.challenger_model = None
        self._results = ABTestResult()
        self._active = False

    def start_test(self, champion, challenger, traffic_pct: float = 0.2):
        """Start an A/B test between two models."""
        self.champion_model = champion
        self.challenger_model = challenger
        self.challenger_traffic_pct = traffic_pct
        self._results = ABTestResult()
        self._active = True
        logger.info("ab_test_started", traffic_pct=traffic_pct)

    def get_model_for_prediction(self):
        """Route to champion or challenger based on traffic split."""
        if not self._active or not self.challenger_model:
            return self.champion_model, "champion"

        if random.random() < self.challenger_traffic_pct:
            return self.challenger_model, "challenger"
        return self.champion_model, "champion"

    def record_result(self, variant: str, pnl: float, is_win: bool):
        """Record the outcome of a trade for A/B test evaluation."""
        if variant == "champion":
            self._results.champion_trades += 1
            self._results.champion_pnl += pnl
            if is_win:
                total_wins = self._results.champion_win_rate * (self._results.champion_trades - 1) + 1
                self._results.champion_win_rate = total_wins / self._results.champion_trades
        else:
            self._results.challenger_trades += 1
            self._results.challenger_pnl += pnl
            if is_win:
                total_wins = self._results.challenger_win_rate * (self._results.challenger_trades - 1) + 1
                self._results.challenger_win_rate = total_wins / self._results.challenger_trades

    def evaluate_test(self) -> ABTestResult:
        """Get current A/B test results."""
        return self._results

    def stop_test(self, promote_challenger: bool = False):
        """Stop the A/B test, optionally promoting the challenger."""
        if promote_challenger and self.challenger_model:
            self.champion_model = self.challenger_model
            logger.info("challenger_promoted")
        self.challenger_model = None
        self._active = False
        logger.info("ab_test_stopped", promote=promote_challenger)


ab_test_manager = ABTestManager()
