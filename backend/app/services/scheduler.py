"""Asyncio-based periodic task scheduler for background jobs.

Replaces Celery for Railway deployment (no separate worker process needed).
All tasks run as asyncio.Tasks inside the FastAPI event loop.
"""

import asyncio
from datetime import datetime, timezone
from typing import Callable, Coroutine, Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class PeriodicTask:
    """A task that runs repeatedly at a fixed interval."""

    def __init__(
        self,
        name: str,
        coro_fn: Callable[[], Coroutine[Any, Any, None]],
        interval_seconds: float,
        run_immediately: bool = False,
    ) -> None:
        self.name = name
        self.coro_fn = coro_fn
        self.interval_seconds = interval_seconds
        self.run_immediately = run_immediately
        self._task: asyncio.Task | None = None
        self._running = False
        self.last_run: datetime | None = None
        self.run_count: int = 0
        self.error_count: int = 0

    async def _loop(self) -> None:
        if not self.run_immediately:
            await asyncio.sleep(self.interval_seconds)

        while self._running:
            try:
                await self.coro_fn()
                self.last_run = datetime.now(timezone.utc)
                self.run_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                logger.error(
                    "periodic_task_error",
                    task=self.name,
                    error=str(e),
                    error_count=self.error_count,
                )
            await asyncio.sleep(self.interval_seconds)

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name=f"periodic_{self.name}")
        logger.info("periodic_task_started", task=self.name, interval=self.interval_seconds)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("periodic_task_stopped", task=self.name)


class Scheduler:
    """Manages multiple periodic tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, PeriodicTask] = {}

    def add_task(
        self,
        name: str,
        coro_fn: Callable[[], Coroutine[Any, Any, None]],
        interval_seconds: float,
        run_immediately: bool = False,
    ) -> None:
        self._tasks[name] = PeriodicTask(name, coro_fn, interval_seconds, run_immediately)

    def start_all(self) -> None:
        for task in self._tasks.values():
            task.start()
        logger.info("scheduler_started", tasks=list(self._tasks.keys()))

    async def stop_all(self) -> None:
        for task in self._tasks.values():
            await task.stop()
        logger.info("scheduler_stopped")

    def get_status(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "interval_seconds": t.interval_seconds,
                "running": t._running,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "run_count": t.run_count,
                "error_count": t.error_count,
            }
            for t in self._tasks.values()
        ]


scheduler = Scheduler()
