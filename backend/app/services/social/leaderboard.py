"""Social trading leaderboard and copy-trading support."""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LeaderboardEntry:
    user_id: str
    username: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    rank: int
    badge: str  # "gold", "silver", "bronze", ""


class Leaderboard:
    """Public performance leaderboard with ranking and badges."""

    BADGE_THRESHOLDS = {
        "gold": {"min_return": 0.50, "min_sharpe": 2.0, "min_trades": 100},
        "silver": {"min_return": 0.20, "min_sharpe": 1.0, "min_trades": 50},
        "bronze": {"min_return": 0.05, "min_sharpe": 0.5, "min_trades": 20},
    }

    def _assign_badge(self, entry: dict) -> str:
        for badge, thresholds in self.BADGE_THRESHOLDS.items():
            if (
                entry.get("total_return", 0) >= thresholds["min_return"]
                and entry.get("sharpe_ratio", 0) >= thresholds["min_sharpe"]
                and entry.get("total_trades", 0) >= thresholds["min_trades"]
            ):
                return badge
        return ""

    async def get_rankings(self, period: str = "30d", limit: int = 50) -> list[LeaderboardEntry]:
        """Get ranked leaderboard entries."""
        # In production, this queries the database for aggregated performance
        logger.info("leaderboard_query", period=period, limit=limit)
        return []

    async def get_user_rank(self, user_id: str) -> LeaderboardEntry | None:
        """Get a specific user's ranking."""
        logger.info("user_rank_query", user_id=user_id)
        return None


class CopyTrading:
    """Copy-trading: follow and replicate top trader strategies."""

    async def follow_trader(self, follower_id: str, leader_id: str, allocation_pct: float) -> dict:
        """Start copying a trader's positions."""
        if allocation_pct <= 0 or allocation_pct > 1.0:
            raise ValueError("Allocation must be between 0 and 1.0")
        logger.info("copy_trading_follow", follower=follower_id, leader=leader_id, allocation=allocation_pct)
        return {"status": "following", "leader_id": leader_id, "allocation_pct": allocation_pct}

    async def unfollow_trader(self, follower_id: str, leader_id: str) -> dict:
        """Stop copying a trader."""
        logger.info("copy_trading_unfollow", follower=follower_id, leader=leader_id)
        return {"status": "unfollowed", "leader_id": leader_id}

    async def get_following(self, user_id: str) -> list[dict]:
        """Get list of traders being followed."""
        return []


leaderboard = Leaderboard()
copy_trading = CopyTrading()
