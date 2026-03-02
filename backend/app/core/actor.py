"""Actor model: fault-isolated concurrent processing.

Each trading pair runs as an independent actor with:
- Own message queue
- Fault isolation (one actor crashing doesn't affect others)
- Supervision tree for automatic recovery
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class ActorState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class Message:
    type: str
    data: dict = field(default_factory=dict)
    reply_to: str | None = None


class Actor(ABC):
    """Base actor with message processing loop."""

    def __init__(self, name: str, mailbox_size: int = 1000):
        self.name = name
        self.state = ActorState.IDLE
        self._mailbox: asyncio.Queue = asyncio.Queue(maxsize=mailbox_size)
        self._task: asyncio.Task | None = None
        self._message_count = 0
        self._error_count = 0

    async def start(self):
        """Start the actor's message processing loop."""
        self.state = ActorState.RUNNING
        self._task = asyncio.create_task(self._run(), name=f"actor-{self.name}")
        logger.info("actor_started", name=self.name)

    async def stop(self):
        """Stop the actor."""
        self.state = ActorState.STOPPED
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("actor_stopped", name=self.name)

    async def send(self, message: Message):
        """Send a message to this actor."""
        await self._mailbox.put(message)

    async def _run(self):
        """Main processing loop."""
        while self.state == ActorState.RUNNING:
            try:
                message = await asyncio.wait_for(self._mailbox.get(), timeout=1.0)
                await self.handle_message(message)
                self._message_count += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._error_count += 1
                logger.error("actor_error", name=self.name, error=str(e))
                if self._error_count > 10:
                    self.state = ActorState.FAILED
                    break

    @abstractmethod
    async def handle_message(self, message: Message):
        """Handle an incoming message. Override in subclasses."""
        pass

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "mailbox_size": self._mailbox.qsize(),
            "messages_processed": self._message_count,
            "errors": self._error_count,
        }


class TradingPairActor(Actor):
    """Actor for processing a single trading pair."""

    def __init__(self, symbol: str):
        super().__init__(name=f"pair-{symbol}")
        self.symbol = symbol
        self._last_price: float = 0.0

    async def handle_message(self, message: Message):
        if message.type == "price_update":
            self._last_price = message.data.get("price", 0.0)
        elif message.type == "check_signals":
            logger.debug("actor_check_signals", symbol=self.symbol, price=self._last_price)
        elif message.type == "execute_trade":
            logger.info("actor_execute_trade", symbol=self.symbol, data=message.data)


class ActorSupervisor:
    """Supervision tree: monitors actors and restarts on failure."""

    def __init__(self):
        self._actors: dict[str, Actor] = {}

    async def spawn(self, actor: Actor) -> None:
        """Spawn an actor under supervision."""
        self._actors[actor.name] = actor
        await actor.start()

    async def stop_all(self) -> None:
        """Stop all supervised actors."""
        for actor in self._actors.values():
            await actor.stop()
        self._actors.clear()

    async def check_health(self) -> list[str]:
        """Check health and restart failed actors."""
        restarted = []
        for name, actor in list(self._actors.items()):
            if actor.state == ActorState.FAILED:
                logger.warning("actor_failed_restarting", name=name)
                await actor.stop()
                actor.state = ActorState.IDLE
                actor._error_count = 0
                await actor.start()
                restarted.append(name)
        return restarted

    def get_status(self) -> dict:
        return {
            "total_actors": len(self._actors),
            "actors": {name: actor.get_status() for name, actor in self._actors.items()},
        }


supervisor = ActorSupervisor()
