"""Paper trading competitions with leaderboards and prizes."""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class CompetitionStatus(str, Enum):
    UPCOMING = "upcoming"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class Competition:
    id: str
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    initial_capital: Decimal
    status: CompetitionStatus
    max_participants: int = 100
    participants: list[str] = field(default_factory=list)
    prizes: list[dict] = field(default_factory=list)


@dataclass
class CompetitionResult:
    user_id: str
    username: str
    final_equity: Decimal
    total_return: float
    total_trades: int
    win_rate: float
    rank: int


class CompetitionManager:
    """Manage paper trading competitions."""

    def __init__(self):
        self._competitions: dict[str, Competition] = {}

    async def create_competition(self, name: str, duration_days: int, initial_capital: float = 10000.0, max_participants: int = 100) -> Competition:
        """Create a new trading competition."""
        comp_id = f"comp_{int(datetime.now(timezone.utc).timestamp())}"
        now = datetime.now(timezone.utc)
        competition = Competition(
            id=comp_id,
            name=name,
            description=f"{duration_days}-day paper trading competition",
            start_time=now,
            end_time=now + timedelta(days=duration_days),
            initial_capital=Decimal(str(initial_capital)),
            status=CompetitionStatus.ACTIVE,
            max_participants=max_participants,
            prizes=[
                {"rank": 1, "prize": "Gold Trophy + Featured Trader Badge"},
                {"rank": 2, "prize": "Silver Trophy"},
                {"rank": 3, "prize": "Bronze Trophy"},
            ],
        )
        self._competitions[comp_id] = competition
        logger.info("competition_created", id=comp_id, name=name, duration=duration_days)
        return competition

    async def join_competition(self, competition_id: str, user_id: str) -> dict:
        """Join an active competition."""
        comp = self._competitions.get(competition_id)
        if not comp:
            return {"error": "Competition not found"}
        if comp.status != CompetitionStatus.ACTIVE:
            return {"error": "Competition is not active"}
        if len(comp.participants) >= comp.max_participants:
            return {"error": "Competition is full"}
        if user_id in comp.participants:
            return {"error": "Already joined"}
        comp.participants.append(user_id)
        logger.info("competition_joined", comp=competition_id, user=user_id)
        return {"status": "joined", "competition_id": competition_id}

    async def get_leaderboard(self, competition_id: str) -> list[CompetitionResult]:
        """Get competition leaderboard."""
        comp = self._competitions.get(competition_id)
        if not comp:
            return []
        return []  # Would query DB for participant performance

    async def list_competitions(self, status: CompetitionStatus | None = None) -> list[Competition]:
        """List all competitions, optionally filtered by status."""
        comps = list(self._competitions.values())
        if status:
            comps = [c for c in comps if c.status == status]
        return comps


competition_manager = CompetitionManager()
