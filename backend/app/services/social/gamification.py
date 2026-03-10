"""Gamification engine: achievements, XP levels, and daily/weekly challenges.

Provides a progression system that rewards consistent trading behaviour,
risk management discipline, and community engagement.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# Enums
# ======================================================================


class AchievementCategory(str, Enum):
    FIRST_STEPS = "first_steps"
    RISK_MASTER = "risk_master"
    CONSISTENCY = "consistency"
    BIG_WINS = "big_wins"
    LEARNING = "learning"
    STRATEGY = "strategy"
    ANALYTICS = "analytics"
    COMMUNITY = "community"


class ChallengeType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


class ChallengeStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"


class Level(str, Enum):
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"
    LEGEND = "legend"


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class Achievement:
    """Definition of a single achievement."""

    id: str
    name: str
    description: str
    category: AchievementCategory
    xp_reward: int
    icon: str = ""
    secret: bool = False  # Hidden until unlocked


@dataclass
class Challenge:
    """A time-limited challenge a user can attempt."""

    id: str
    name: str
    description: str
    challenge_type: ChallengeType
    xp_reward: int
    criteria: dict = field(default_factory=dict)
    status: ChallengeStatus = ChallengeStatus.ACTIVE
    expires_at: datetime | None = None
    progress: float = 0.0  # 0.0 – 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserProgress:
    """Tracks a user's gamification state."""

    user_id: str
    xp: int = 0
    level: Level = Level.NOVICE
    unlocked_achievements: list[str] = field(default_factory=list)
    active_challenges: list[str] = field(default_factory=list)
    completed_challenges: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    streak_days: int = 0
    last_active_date: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ======================================================================
# Level thresholds
# ======================================================================

LEVEL_THRESHOLDS: list[tuple[Level, int]] = [
    (Level.NOVICE, 0),
    (Level.BEGINNER, 500),
    (Level.INTERMEDIATE, 2_000),
    (Level.ADVANCED, 5_000),
    (Level.EXPERT, 12_000),
    (Level.MASTER, 25_000),
    (Level.LEGEND, 50_000),
]


# ======================================================================
# Achievement catalogue (50+)
# ======================================================================

_ACHIEVEMENT_DEFS: list[dict] = [
    # --- First Steps (1-8) ---
    {"id": "first_trade", "name": "First Trade", "description": "Execute your first trade", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 50},
    {"id": "first_profit", "name": "First Profit", "description": "Close your first profitable trade", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 75},
    {"id": "first_loss", "name": "Lesson Learned", "description": "Close your first losing trade — it happens to everyone", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 50},
    {"id": "first_strategy", "name": "Strategy Architect", "description": "Create your first trading strategy", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 100},
    {"id": "first_backtest", "name": "Time Traveller", "description": "Run your first backtest", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 100},
    {"id": "first_alert", "name": "Watchful Eye", "description": "Create your first price alert", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 50},
    {"id": "profile_complete", "name": "All Set", "description": "Complete your profile setup", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 30},
    {"id": "first_deposit", "name": "Skin in the Game", "description": "Make your first deposit", "category": AchievementCategory.FIRST_STEPS, "xp_reward": 75},
    # --- Risk Master (9-16) ---
    {"id": "stop_loss_10", "name": "Safety Net", "description": "Use stop-losses on 10 trades", "category": AchievementCategory.RISK_MASTER, "xp_reward": 100},
    {"id": "stop_loss_100", "name": "Risk Disciplinarian", "description": "Use stop-losses on 100 trades", "category": AchievementCategory.RISK_MASTER, "xp_reward": 300},
    {"id": "max_dd_5", "name": "Steady Hand", "description": "Keep max drawdown below 5% for 30 days", "category": AchievementCategory.RISK_MASTER, "xp_reward": 500},
    {"id": "risk_reward_2", "name": "Risk/Reward Pro", "description": "Maintain average R:R >= 2.0 over 20 trades", "category": AchievementCategory.RISK_MASTER, "xp_reward": 250},
    {"id": "no_overleverage", "name": "Level Headed", "description": "Never exceed 10x leverage for 30 days", "category": AchievementCategory.RISK_MASTER, "xp_reward": 200},
    {"id": "hedging_first", "name": "Hedgehog", "description": "Open your first hedging position", "category": AchievementCategory.RISK_MASTER, "xp_reward": 150},
    {"id": "position_sizing", "name": "Measured Approach", "description": "Use position sizing on 50 consecutive trades", "category": AchievementCategory.RISK_MASTER, "xp_reward": 200},
    {"id": "circuit_breaker_survive", "name": "Circuit Survivor", "description": "Recover from a circuit-breaker event", "category": AchievementCategory.RISK_MASTER, "xp_reward": 300},
    # --- Consistency (17-24) ---
    {"id": "streak_7", "name": "Week Warrior", "description": "Log in 7 consecutive days", "category": AchievementCategory.CONSISTENCY, "xp_reward": 100},
    {"id": "streak_30", "name": "Monthly Regular", "description": "Log in 30 consecutive days", "category": AchievementCategory.CONSISTENCY, "xp_reward": 300},
    {"id": "streak_100", "name": "Iron Habit", "description": "Log in 100 consecutive days", "category": AchievementCategory.CONSISTENCY, "xp_reward": 750},
    {"id": "trades_50", "name": "Getting Started", "description": "Execute 50 trades", "category": AchievementCategory.CONSISTENCY, "xp_reward": 150},
    {"id": "trades_200", "name": "Active Trader", "description": "Execute 200 trades", "category": AchievementCategory.CONSISTENCY, "xp_reward": 400},
    {"id": "trades_1000", "name": "Trade Machine", "description": "Execute 1,000 trades", "category": AchievementCategory.CONSISTENCY, "xp_reward": 1000},
    {"id": "journal_30", "name": "Diary Keeper", "description": "Write 30 trade journal entries", "category": AchievementCategory.CONSISTENCY, "xp_reward": 200},
    {"id": "dca_12", "name": "Steady Accumulator", "description": "Complete 12 DCA buys in a row", "category": AchievementCategory.CONSISTENCY, "xp_reward": 250},
    # --- Big Wins (25-32) ---
    {"id": "profit_100", "name": "Triple Digits", "description": "Earn $100 in realized profit", "category": AchievementCategory.BIG_WINS, "xp_reward": 100},
    {"id": "profit_1k", "name": "Grand Slam", "description": "Earn $1,000 in realized profit", "category": AchievementCategory.BIG_WINS, "xp_reward": 300},
    {"id": "profit_10k", "name": "Five Figures", "description": "Earn $10,000 in realized profit", "category": AchievementCategory.BIG_WINS, "xp_reward": 750},
    {"id": "win_streak_5", "name": "Hot Hand", "description": "Win 5 trades in a row", "category": AchievementCategory.BIG_WINS, "xp_reward": 200},
    {"id": "win_streak_10", "name": "On Fire", "description": "Win 10 trades in a row", "category": AchievementCategory.BIG_WINS, "xp_reward": 500},
    {"id": "single_trade_10pct", "name": "Home Run", "description": "Profit 10%+ on a single trade", "category": AchievementCategory.BIG_WINS, "xp_reward": 200},
    {"id": "monthly_positive_3", "name": "Quarterly Positive", "description": "3 consecutive profitable months", "category": AchievementCategory.BIG_WINS, "xp_reward": 500},
    {"id": "monthly_positive_12", "name": "Year of Green", "description": "12 consecutive profitable months", "category": AchievementCategory.BIG_WINS, "xp_reward": 2000, "secret": True},
    # --- Learning (33-39) ---
    {"id": "tutorial_first", "name": "Eager Student", "description": "Complete your first tutorial", "category": AchievementCategory.LEARNING, "xp_reward": 50},
    {"id": "tutorial_all", "name": "Scholar", "description": "Complete all available tutorials", "category": AchievementCategory.LEARNING, "xp_reward": 500},
    {"id": "paper_trade_20", "name": "Practice Makes Perfect", "description": "Complete 20 paper trades", "category": AchievementCategory.LEARNING, "xp_reward": 150},
    {"id": "read_docs", "name": "RTFM", "description": "Read the full risk management guide", "category": AchievementCategory.LEARNING, "xp_reward": 75},
    {"id": "quiz_pass", "name": "Quiz Whiz", "description": "Pass a trading knowledge quiz", "category": AchievementCategory.LEARNING, "xp_reward": 100},
    {"id": "webinar_attend", "name": "Live Learner", "description": "Attend a live webinar or Q&A session", "category": AchievementCategory.LEARNING, "xp_reward": 75},
    {"id": "glossary_master", "name": "Walking Dictionary", "description": "Look up 50 terms in the glossary", "category": AchievementCategory.LEARNING, "xp_reward": 100},
    # --- Strategy (40-46) ---
    {"id": "strategy_5", "name": "Multi-Strategy", "description": "Create 5 different strategies", "category": AchievementCategory.STRATEGY, "xp_reward": 200},
    {"id": "backtest_50", "name": "Data Miner", "description": "Run 50 backtests", "category": AchievementCategory.STRATEGY, "xp_reward": 300},
    {"id": "sharpe_above_2", "name": "Sharp Shooter", "description": "Achieve a Sharpe ratio above 2.0", "category": AchievementCategory.STRATEGY, "xp_reward": 500},
    {"id": "multi_asset", "name": "Diversifier", "description": "Trade 5 different assets", "category": AchievementCategory.STRATEGY, "xp_reward": 150},
    {"id": "algo_trade", "name": "Algo Trader", "description": "Execute your first automated trade", "category": AchievementCategory.STRATEGY, "xp_reward": 200},
    {"id": "ml_model_deploy", "name": "AI Trader", "description": "Deploy a custom ML model to live trading", "category": AchievementCategory.STRATEGY, "xp_reward": 400},
    {"id": "strategy_publish", "name": "Open Source", "description": "Publish a strategy to the marketplace", "category": AchievementCategory.STRATEGY, "xp_reward": 300},
    # --- Analytics (47-51) ---
    {"id": "report_first", "name": "Number Cruncher", "description": "Generate your first performance report", "category": AchievementCategory.ANALYTICS, "xp_reward": 75},
    {"id": "report_pdf", "name": "Report Card", "description": "Export a PDF performance report", "category": AchievementCategory.ANALYTICS, "xp_reward": 100},
    {"id": "tax_report", "name": "Tax Ready", "description": "Generate your first tax report", "category": AchievementCategory.ANALYTICS, "xp_reward": 150},
    {"id": "custom_dashboard", "name": "Dashboard Designer", "description": "Customise your analytics dashboard", "category": AchievementCategory.ANALYTICS, "xp_reward": 100},
    {"id": "heatmap_view", "name": "Heat Seeker", "description": "View the correlation heatmap", "category": AchievementCategory.ANALYTICS, "xp_reward": 50},
    # --- Community (52-57) ---
    {"id": "referral_1", "name": "Recruiter", "description": "Refer your first friend", "category": AchievementCategory.COMMUNITY, "xp_reward": 150},
    {"id": "referral_10", "name": "Ambassador", "description": "Refer 10 friends", "category": AchievementCategory.COMMUNITY, "xp_reward": 500},
    {"id": "competition_join", "name": "Challenger", "description": "Join a trading competition", "category": AchievementCategory.COMMUNITY, "xp_reward": 100},
    {"id": "competition_win", "name": "Champion", "description": "Win a trading competition", "category": AchievementCategory.COMMUNITY, "xp_reward": 750},
    {"id": "copy_trader", "name": "Trendsetter", "description": "Have another user copy your trades", "category": AchievementCategory.COMMUNITY, "xp_reward": 200},
    {"id": "leaderboard_top10", "name": "Hall of Fame", "description": "Reach the top 10 on the leaderboard", "category": AchievementCategory.COMMUNITY, "xp_reward": 1000, "secret": True},
]

# ======================================================================
# Challenge templates
# ======================================================================

_DAILY_CHALLENGE_TEMPLATES: list[dict] = [
    {"name": "Quick Trader", "description": "Execute 3 trades today", "xp_reward": 50, "criteria": {"trades": 3}},
    {"name": "Profit Hunter", "description": "Close a profitable trade today", "xp_reward": 40, "criteria": {"profitable_trades": 1}},
    {"name": "Market Scout", "description": "Analyse 3 different assets today", "xp_reward": 30, "criteria": {"assets_analysed": 3}},
    {"name": "Journal Entry", "description": "Write a trade journal entry today", "xp_reward": 25, "criteria": {"journal_entries": 1}},
    {"name": "Alert Setter", "description": "Create 2 price alerts", "xp_reward": 25, "criteria": {"alerts_created": 2}},
    {"name": "News Reader", "description": "Read 5 market news articles", "xp_reward": 20, "criteria": {"news_read": 5}},
    {"name": "Risk Reviewer", "description": "Review your risk metrics", "xp_reward": 30, "criteria": {"risk_review": 1}},
]

_WEEKLY_CHALLENGE_TEMPLATES: list[dict] = [
    {"name": "Weekly Warrior", "description": "Execute 15 trades this week", "xp_reward": 200, "criteria": {"trades": 15}},
    {"name": "Win Streak", "description": "Win 5 trades in a row this week", "xp_reward": 250, "criteria": {"win_streak": 5}},
    {"name": "Diversify", "description": "Trade 3 different assets this week", "xp_reward": 150, "criteria": {"unique_assets": 3}},
    {"name": "Backtest Sprint", "description": "Run 5 backtests this week", "xp_reward": 175, "criteria": {"backtests": 5}},
    {"name": "Positive Week", "description": "Finish the week in profit", "xp_reward": 300, "criteria": {"weekly_profit": True}},
    {"name": "Community Spirit", "description": "Share a strategy or idea this week", "xp_reward": 100, "criteria": {"shares": 1}},
    {"name": "Full Attendance", "description": "Log in every day this week", "xp_reward": 150, "criteria": {"login_days": 7}},
]


# ======================================================================
# Engine
# ======================================================================


class GamificationEngine:
    """Manages achievements, XP, levels, and challenges for all users.

    Usage::

        engine = GamificationEngine()
        progress = engine.get_or_create_progress("user_123")
        new = await engine.check_and_unlock(
            "user_123", "first_trade",
        )
    """

    def __init__(self) -> None:
        self._achievements: dict[str, Achievement] = {}
        self._user_progress: dict[str, UserProgress] = {}
        self._active_challenges: dict[str, Challenge] = {}  # challenge_id -> Challenge
        self._load_achievements()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_achievements(self) -> None:
        """Populate the catalogue from the built-in definitions."""
        for defn in _ACHIEVEMENT_DEFS:
            ach = Achievement(
                id=defn["id"],
                name=defn["name"],
                description=defn["description"],
                category=defn["category"],
                xp_reward=defn["xp_reward"],
                icon=defn.get("icon", ""),
                secret=defn.get("secret", False),
            )
            self._achievements[ach.id] = ach
        logger.info("gamification_achievements_loaded", count=len(self._achievements))

    # ------------------------------------------------------------------
    # User progress
    # ------------------------------------------------------------------

    def get_or_create_progress(self, user_id: str) -> UserProgress:
        """Retrieve or initialise progress for a user."""
        if user_id not in self._user_progress:
            self._user_progress[user_id] = UserProgress(user_id=user_id)
            logger.info("gamification_progress_created", user_id=user_id)
        return self._user_progress[user_id]

    def get_progress(self, user_id: str) -> UserProgress | None:
        """Return existing progress or ``None``."""
        return self._user_progress.get(user_id)

    # ------------------------------------------------------------------
    # XP and levelling
    # ------------------------------------------------------------------

    def _level_for_xp(self, xp: int) -> Level:
        """Determine the level corresponding to an XP total."""
        current_level = Level.NOVICE
        for level, threshold in LEVEL_THRESHOLDS:
            if xp >= threshold:
                current_level = level
            else:
                break
        return current_level

    def _xp_for_next_level(self, current_level: Level) -> int | None:
        """Return the XP required to reach the next level, or ``None`` if max."""
        found = False
        for level, threshold in LEVEL_THRESHOLDS:
            if found:
                return threshold
            if level == current_level:
                found = True
        return None  # Already at Legend

    async def award_xp(self, user_id: str, amount: int, reason: str = "") -> UserProgress:
        """Award XP to a user and potentially level them up.

        Args:
            user_id: Target user.
            amount: XP to grant (must be positive).
            reason: Optional description for logging.

        Returns:
            Updated :class:`UserProgress`.
        """
        if amount <= 0:
            raise ValueError("XP amount must be positive")

        progress = self.get_or_create_progress(user_id)
        old_level = progress.level
        progress.xp += amount
        progress.level = self._level_for_xp(progress.xp)

        logger.info(
            "gamification_xp_awarded",
            user_id=user_id,
            amount=amount,
            total_xp=progress.xp,
            reason=reason,
        )

        if progress.level != old_level:
            logger.info(
                "gamification_level_up",
                user_id=user_id,
                old_level=old_level.value,
                new_level=progress.level.value,
                xp=progress.xp,
            )

        return progress

    # ------------------------------------------------------------------
    # Achievements
    # ------------------------------------------------------------------

    def get_achievement(self, achievement_id: str) -> Achievement | None:
        """Retrieve an achievement definition by id."""
        return self._achievements.get(achievement_id)

    def list_achievements(self, category: AchievementCategory | None = None) -> list[Achievement]:
        """Return all achievements, optionally filtered by category."""
        if category is None:
            return list(self._achievements.values())
        return [a for a in self._achievements.values() if a.category == category]

    async def check_and_unlock(self, user_id: str, achievement_id: str) -> Achievement | None:
        """Unlock an achievement for *user_id* if not already unlocked.

        Returns the :class:`Achievement` on success, or ``None`` if already
        unlocked or the achievement id is invalid.
        """
        achievement = self._achievements.get(achievement_id)
        if achievement is None:
            logger.warning("gamification_unknown_achievement", achievement_id=achievement_id)
            return None

        progress = self.get_or_create_progress(user_id)
        if achievement_id in progress.unlocked_achievements:
            return None  # Already unlocked

        progress.unlocked_achievements.append(achievement_id)
        await self.award_xp(user_id, achievement.xp_reward, reason=f"achievement:{achievement_id}")

        logger.info(
            "gamification_achievement_unlocked",
            user_id=user_id,
            achievement=achievement_id,
            name=achievement.name,
            xp=achievement.xp_reward,
        )
        return achievement

    def get_user_achievements(self, user_id: str) -> list[Achievement]:
        """Return the list of achievements a user has unlocked."""
        progress = self.get_or_create_progress(user_id)
        return [
            self._achievements[aid]
            for aid in progress.unlocked_achievements
            if aid in self._achievements
        ]

    def get_locked_achievements(self, user_id: str) -> list[Achievement]:
        """Return achievements the user has **not** yet unlocked (excluding secrets)."""
        progress = self.get_or_create_progress(user_id)
        unlocked = set(progress.unlocked_achievements)
        return [
            a
            for a in self._achievements.values()
            if a.id not in unlocked and not a.secret
        ]

    # ------------------------------------------------------------------
    # Streaks
    # ------------------------------------------------------------------

    async def record_activity(self, user_id: str) -> int:
        """Record daily activity for streak tracking.

        Returns the current streak length in days.
        """
        progress = self.get_or_create_progress(user_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if progress.last_active_date == today:
            return progress.streak_days  # Already recorded

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        if progress.last_active_date == yesterday:
            progress.streak_days += 1
        else:
            progress.streak_days = 1

        progress.last_active_date = today

        # Auto-unlock streak achievements
        streak = progress.streak_days
        if streak >= 7:
            await self.check_and_unlock(user_id, "streak_7")
        if streak >= 30:
            await self.check_and_unlock(user_id, "streak_30")
        if streak >= 100:
            await self.check_and_unlock(user_id, "streak_100")

        logger.debug("gamification_streak", user_id=user_id, streak=streak)
        return streak

    # ------------------------------------------------------------------
    # Challenges
    # ------------------------------------------------------------------

    async def generate_daily_challenge(self, user_id: str) -> Challenge:
        """Generate a random daily challenge for the user."""
        import random

        template = random.choice(_DAILY_CHALLENGE_TEMPLATES)
        challenge = Challenge(
            id=f"ch_{uuid.uuid4().hex[:10]}",
            name=template["name"],
            description=template["description"],
            challenge_type=ChallengeType.DAILY,
            xp_reward=template["xp_reward"],
            criteria=dict(template["criteria"]),
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        self._active_challenges[challenge.id] = challenge

        progress = self.get_or_create_progress(user_id)
        progress.active_challenges.append(challenge.id)

        logger.info(
            "gamification_daily_challenge_created",
            user_id=user_id,
            challenge_id=challenge.id,
            name=challenge.name,
        )
        return challenge

    async def generate_weekly_challenge(self, user_id: str) -> Challenge:
        """Generate a random weekly challenge for the user."""
        import random

        template = random.choice(_WEEKLY_CHALLENGE_TEMPLATES)
        challenge = Challenge(
            id=f"ch_{uuid.uuid4().hex[:10]}",
            name=template["name"],
            description=template["description"],
            challenge_type=ChallengeType.WEEKLY,
            xp_reward=template["xp_reward"],
            criteria=dict(template["criteria"]),
            expires_at=datetime.now(timezone.utc) + timedelta(weeks=1),
        )
        self._active_challenges[challenge.id] = challenge

        progress = self.get_or_create_progress(user_id)
        progress.active_challenges.append(challenge.id)

        logger.info(
            "gamification_weekly_challenge_created",
            user_id=user_id,
            challenge_id=challenge.id,
            name=challenge.name,
        )
        return challenge

    async def update_challenge_progress(
        self,
        user_id: str,
        challenge_id: str,
        progress_value: float,
    ) -> Challenge:
        """Update progress on a challenge (0.0-1.0).

        Automatically completes the challenge and awards XP when progress
        reaches 1.0.
        """
        challenge = self._active_challenges.get(challenge_id)
        if challenge is None:
            raise KeyError(f"Challenge not found: {challenge_id}")

        now = datetime.now(timezone.utc)
        if challenge.expires_at and now > challenge.expires_at:
            challenge.status = ChallengeStatus.EXPIRED
            user_prog = self.get_or_create_progress(user_id)
            if challenge_id in user_prog.active_challenges:
                user_prog.active_challenges.remove(challenge_id)
            logger.info("gamification_challenge_expired", challenge_id=challenge_id)
            return challenge

        challenge.progress = min(progress_value, 1.0)

        if challenge.progress >= 1.0:
            challenge.status = ChallengeStatus.COMPLETED
            user_prog = self.get_or_create_progress(user_id)
            if challenge_id in user_prog.active_challenges:
                user_prog.active_challenges.remove(challenge_id)
            user_prog.completed_challenges.append(challenge_id)
            await self.award_xp(user_id, challenge.xp_reward, reason=f"challenge:{challenge_id}")
            logger.info(
                "gamification_challenge_completed",
                user_id=user_id,
                challenge_id=challenge_id,
                name=challenge.name,
                xp=challenge.xp_reward,
            )

        return challenge

    def get_active_challenges(self, user_id: str) -> list[Challenge]:
        """Return all active (non-expired) challenges for a user."""
        progress = self.get_or_create_progress(user_id)
        now = datetime.now(timezone.utc)
        result: list[Challenge] = []
        for cid in list(progress.active_challenges):
            ch = self._active_challenges.get(cid)
            if ch is None:
                continue
            if ch.expires_at and now > ch.expires_at:
                ch.status = ChallengeStatus.EXPIRED
                progress.active_challenges.remove(cid)
                continue
            result.append(ch)
        return result

    # ------------------------------------------------------------------
    # Summary / leaderboard helper
    # ------------------------------------------------------------------

    def get_level_info(self, user_id: str) -> dict:
        """Return a summary of the user's level progression."""
        progress = self.get_or_create_progress(user_id)
        next_level_xp = self._xp_for_next_level(progress.level)

        # Find current level threshold
        current_threshold = 0
        for level, threshold in LEVEL_THRESHOLDS:
            if level == progress.level:
                current_threshold = threshold
                break

        if next_level_xp is not None:
            xp_in_level = progress.xp - current_threshold
            xp_needed = next_level_xp - current_threshold
            pct = (xp_in_level / xp_needed * 100) if xp_needed > 0 else 100.0
        else:
            xp_in_level = progress.xp - current_threshold
            xp_needed = 0
            pct = 100.0

        return {
            "user_id": user_id,
            "level": progress.level.value,
            "xp": progress.xp,
            "xp_in_level": xp_in_level,
            "xp_needed_for_next": xp_needed,
            "progress_pct": round(pct, 1),
            "next_level": None if next_level_xp is None else self._level_for_xp(next_level_xp).value,
            "achievements_unlocked": len(progress.unlocked_achievements),
            "achievements_total": len(self._achievements),
            "streak_days": progress.streak_days,
        }


gamification_engine = GamificationEngine()
