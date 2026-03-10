"""Saga orchestrator: distributed multi-step transaction coordination.

Implements the Saga pattern to manage long-running transactions that span
multiple services or subsystems. Each saga consists of a sequence of steps,
each with a compensating action for rollback on failure. Provides full
audit trail, timeout/retry policies, and dead letter queue handling.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

from app.core.event_sourcing import EventStore, EventType, create_event
from app.core.logging import get_logger

logger = get_logger(__name__)

# Type alias for async step functions
StepFn = Callable[..., Coroutine[Any, Any, dict]]


class SagaState(str, Enum):
    """Lifecycle states for a saga instance."""

    PENDING = "pending"
    RUNNING = "running"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DEAD_LETTERED = "dead_lettered"


@dataclass
class SagaStep:
    """Definition of a single step within a saga.

    Attributes:
        name: Human-readable identifier for the step.
        action: Async callable that performs the forward action.
        compensate: Async callable that reverses the action on rollback.
        timeout_seconds: Max wall-clock time allowed for the action.
        max_retries: Number of retry attempts before the step is marked failed.
        retry_delay_seconds: Base delay between retries (exponential backoff applied).
    """

    name: str
    action: StepFn
    compensate: StepFn | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class SagaStepResult:
    """Outcome of executing a single saga step."""

    step_name: str
    status: str  # "success", "failed", "compensated"
    result: dict = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_ms: float = 0.0


@dataclass
class SagaLog:
    """Full audit record for a saga execution.

    Immutable once the saga reaches a terminal state. Captures every step
    attempt, compensation, and the final outcome.
    """

    saga_id: str
    saga_type: str
    state: SagaState
    context: dict = field(default_factory=dict)
    step_results: list[SagaStepResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "saga_id": self.saga_id,
            "saga_type": self.saga_type,
            "state": self.state.value,
            "context": self.context,
            "step_results": [
                {
                    "step_name": sr.step_name,
                    "status": sr.status,
                    "result": sr.result,
                    "error": sr.error,
                    "attempts": sr.attempts,
                    "started_at": sr.started_at.isoformat(),
                    "completed_at": sr.completed_at.isoformat() if sr.completed_at else None,
                    "duration_ms": sr.duration_ms,
                }
                for sr in self.step_results
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "retry_count": self.retry_count,
        }


class DeadLetterQueue:
    """Holds sagas that exhausted all retries without completing.

    Entries stay in the DLQ until manually inspected and resolved by an
    operator. The queue is bounded to prevent unbounded memory growth.
    """

    def __init__(self, max_size: int = 1000):
        self._queue: deque[SagaLog] = deque(maxlen=max_size)

    async def enqueue(self, saga_log: SagaLog) -> None:
        saga_log.state = SagaState.DEAD_LETTERED
        saga_log.updated_at = datetime.now(timezone.utc)
        self._queue.append(saga_log)
        logger.error(
            "saga_dead_lettered",
            saga_id=saga_log.saga_id,
            saga_type=saga_log.saga_type,
            error=saga_log.error,
        )

    async def peek(self, limit: int = 10) -> list[SagaLog]:
        """Return up to *limit* entries without removing them."""
        return list(self._queue)[:limit]

    async def dequeue(self) -> SagaLog | None:
        """Remove and return the oldest entry, or None if empty."""
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    @property
    def size(self) -> int:
        return len(self._queue)


class SagaOrchestrator:
    """Coordinates multi-step distributed transactions using the Saga pattern.

    Usage::

        orchestrator = SagaOrchestrator(event_store)
        saga_log = await orchestrator.execute_trade_saga(trade_context)
    """

    def __init__(
        self,
        event_store: EventStore,
        *,
        max_saga_retries: int = 3,
        default_timeout: float = 30.0,
    ):
        self._event_store = event_store
        self._max_saga_retries = max_saga_retries
        self._default_timeout = default_timeout
        self._saga_logs: dict[str, SagaLog] = {}
        self._dead_letter_queue = DeadLetterQueue()
        self._active_sagas: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, saga_type: str, steps: list[SagaStep], context: dict) -> SagaLog:
        """Run an ordered sequence of steps as a saga.

        On failure, previously completed steps are compensated in reverse
        order. If the entire saga fails after *max_saga_retries*, the log
        is moved to the dead letter queue.
        """
        saga_id = str(uuid4())
        saga_log = SagaLog(
            saga_id=saga_id,
            saga_type=saga_type,
            state=SagaState.PENDING,
            context=dict(context),
        )
        self._saga_logs[saga_id] = saga_log
        logger.info("saga_started", saga_id=saga_id, saga_type=saga_type)

        await self._emit_event(saga_id, saga_type, "saga.started", context)

        completed_steps: list[tuple[SagaStep, SagaStepResult]] = []

        try:
            saga_log.state = SagaState.RUNNING
            saga_log.updated_at = datetime.now(timezone.utc)

            for step in steps:
                step_result = await self._execute_step(step, context)
                saga_log.step_results.append(step_result)
                saga_log.updated_at = datetime.now(timezone.utc)

                if step_result.status == "success":
                    completed_steps.append((step, step_result))
                    # Merge step output into the running context so
                    # downstream steps can use it.
                    context.update(step_result.result)
                else:
                    raise _SagaStepError(step.name, step_result.error or "unknown")

            # All steps succeeded
            saga_log.state = SagaState.COMPLETED
            saga_log.completed_at = datetime.now(timezone.utc)
            saga_log.updated_at = saga_log.completed_at
            logger.info("saga_completed", saga_id=saga_id, saga_type=saga_type)
            await self._emit_event(saga_id, saga_type, "saga.completed", context)

        except _SagaStepError as exc:
            logger.warning(
                "saga_step_failed",
                saga_id=saga_id,
                step=exc.step_name,
                error=exc.message,
            )
            saga_log.error = f"Step '{exc.step_name}' failed: {exc.message}"

            # Compensate in reverse order
            await self._compensate(saga_id, saga_type, completed_steps, context, saga_log)

            # Decide whether to dead-letter
            saga_log.retry_count += 1
            if saga_log.retry_count >= self._max_saga_retries:
                await self._dead_letter_queue.enqueue(saga_log)
            else:
                saga_log.state = SagaState.FAILED

            saga_log.updated_at = datetime.now(timezone.utc)
            await self._emit_event(saga_id, saga_type, "saga.failed", {"error": saga_log.error})

        return saga_log

    async def execute_trade_saga(self, trade_context: dict) -> SagaLog:
        """Pre-built saga for the standard trade lifecycle.

        Steps: validate -> reserve_capital -> place_order -> confirm -> update_portfolio

        *trade_context* must include at minimum:
            - symbol: str
            - side: "buy" | "sell"
            - quantity: float
            - price: float
            - portfolio_id: str
        """
        steps = [
            SagaStep(
                name="validate",
                action=self._trade_validate,
                compensate=None,  # validation is idempotent, no rollback needed
                timeout_seconds=5.0,
                max_retries=1,
            ),
            SagaStep(
                name="reserve_capital",
                action=self._trade_reserve_capital,
                compensate=self._trade_release_capital,
                timeout_seconds=10.0,
                max_retries=2,
            ),
            SagaStep(
                name="place_order",
                action=self._trade_place_order,
                compensate=self._trade_cancel_order,
                timeout_seconds=self._default_timeout,
                max_retries=3,
            ),
            SagaStep(
                name="confirm",
                action=self._trade_confirm,
                compensate=self._trade_revert_confirmation,
                timeout_seconds=15.0,
                max_retries=2,
            ),
            SagaStep(
                name="update_portfolio",
                action=self._trade_update_portfolio,
                compensate=self._trade_revert_portfolio,
                timeout_seconds=10.0,
                max_retries=2,
            ),
        ]

        return await self.execute("trade", steps, trade_context)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_saga_log(self, saga_id: str) -> SagaLog | None:
        return self._saga_logs.get(saga_id)

    async def list_sagas(
        self,
        *,
        state: SagaState | None = None,
        saga_type: str | None = None,
        limit: int = 50,
    ) -> list[SagaLog]:
        logs = list(self._saga_logs.values())
        if state is not None:
            logs = [s for s in logs if s.state == state]
        if saga_type is not None:
            logs = [s for s in logs if s.saga_type == saga_type]
        return logs[-limit:]

    async def get_dead_letter_entries(self, limit: int = 10) -> list[SagaLog]:
        return await self._dead_letter_queue.peek(limit)

    def get_stats(self) -> dict:
        by_state: dict[str, int] = {}
        for log in self._saga_logs.values():
            by_state[log.state.value] = by_state.get(log.state.value, 0) + 1
        return {
            "total_sagas": len(self._saga_logs),
            "by_state": by_state,
            "dead_letter_queue_size": self._dead_letter_queue.size,
            "active_sagas": len(self._active_sagas),
        }

    # ------------------------------------------------------------------
    # Internal: step execution
    # ------------------------------------------------------------------

    async def _execute_step(self, step: SagaStep, context: dict) -> SagaStepResult:
        """Execute a single saga step with retry and timeout policies."""
        step_result = SagaStepResult(step_name=step.name, status="failed")
        last_error: str | None = None

        for attempt in range(1, step.max_retries + 1):
            step_result.attempts = attempt
            start = datetime.now(timezone.utc)

            try:
                result = await asyncio.wait_for(
                    step.action(context),
                    timeout=step.timeout_seconds,
                )
                step_result.status = "success"
                step_result.result = result if isinstance(result, dict) else {}
                step_result.completed_at = datetime.now(timezone.utc)
                step_result.duration_ms = (
                    step_result.completed_at - start
                ).total_seconds() * 1000
                logger.debug(
                    "saga_step_succeeded",
                    step=step.name,
                    attempt=attempt,
                    duration_ms=step_result.duration_ms,
                )
                return step_result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {step.timeout_seconds}s"
                logger.warning(
                    "saga_step_timeout",
                    step=step.name,
                    attempt=attempt,
                    timeout=step.timeout_seconds,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "saga_step_error",
                    step=step.name,
                    attempt=attempt,
                    error=last_error,
                )

            # Exponential backoff before retry
            if attempt < step.max_retries:
                delay = step.retry_delay_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        step_result.error = last_error
        step_result.completed_at = datetime.now(timezone.utc)
        step_result.duration_ms = (
            step_result.completed_at - step_result.started_at
        ).total_seconds() * 1000
        return step_result

    async def _compensate(
        self,
        saga_id: str,
        saga_type: str,
        completed_steps: list[tuple[SagaStep, SagaStepResult]],
        context: dict,
        saga_log: SagaLog,
    ) -> None:
        """Run compensating transactions in reverse order."""
        saga_log.state = SagaState.COMPENSATING
        saga_log.updated_at = datetime.now(timezone.utc)
        logger.info("saga_compensating", saga_id=saga_id, steps=len(completed_steps))

        for step, _step_result in reversed(completed_steps):
            if step.compensate is None:
                continue
            comp_result = SagaStepResult(
                step_name=f"compensate_{step.name}",
                status="failed",
            )
            try:
                await asyncio.wait_for(
                    step.compensate(context),
                    timeout=step.timeout_seconds,
                )
                comp_result.status = "compensated"
                logger.debug("saga_step_compensated", step=step.name, saga_id=saga_id)
            except Exception as exc:
                comp_result.error = str(exc)
                logger.error(
                    "saga_compensation_failed",
                    step=step.name,
                    saga_id=saga_id,
                    error=str(exc),
                )
            comp_result.completed_at = datetime.now(timezone.utc)
            comp_result.duration_ms = (
                comp_result.completed_at - comp_result.started_at
            ).total_seconds() * 1000
            saga_log.step_results.append(comp_result)

        await self._emit_event(saga_id, saga_type, "saga.compensated", context)

    # ------------------------------------------------------------------
    # Internal: event emission
    # ------------------------------------------------------------------

    async def _emit_event(
        self, saga_id: str, saga_type: str, event_name: str, data: dict
    ) -> None:
        """Publish a saga lifecycle event to the event store."""
        # Use a generic event type that maps well to the existing enum.
        # CONFIG_CHANGED is the closest catch-all; in production you'd
        # extend EventType with saga-specific variants.
        event = create_event(
            event_type=EventType.CONFIG_CHANGED,
            aggregate_id=saga_id,
            aggregate_type=f"saga:{saga_type}",
            data={"saga_event": event_name, **data},
            saga_id=saga_id,
        )
        await self._event_store.append(event)

    # ------------------------------------------------------------------
    # Trade saga step implementations (stubs)
    # ------------------------------------------------------------------

    async def _trade_validate(self, ctx: dict) -> dict:
        """Validate trade parameters (symbol exists, market open, etc.)."""
        symbol = ctx.get("symbol")
        quantity = ctx.get("quantity", 0)
        price = ctx.get("price", 0)
        if not symbol:
            raise ValueError("Missing symbol")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if price <= 0:
            raise ValueError("Price must be positive")
        logger.debug("trade_validated", symbol=symbol, quantity=quantity, price=price)
        return {"validated": True, "notional": quantity * price}

    async def _trade_reserve_capital(self, ctx: dict) -> dict:
        """Reserve capital in the portfolio for the trade."""
        notional = ctx.get("notional", 0)
        portfolio_id = ctx.get("portfolio_id")
        reservation_id = str(uuid4())
        logger.debug(
            "capital_reserved",
            portfolio_id=portfolio_id,
            amount=notional,
            reservation_id=reservation_id,
        )
        return {"reservation_id": reservation_id, "reserved_amount": notional}

    async def _trade_release_capital(self, ctx: dict) -> dict:
        """Compensate: release the reserved capital."""
        reservation_id = ctx.get("reservation_id")
        logger.debug("capital_released", reservation_id=reservation_id)
        return {"released": True}

    async def _trade_place_order(self, ctx: dict) -> dict:
        """Place the order with the broker."""
        order_id = str(uuid4())
        logger.debug(
            "order_placed",
            order_id=order_id,
            symbol=ctx.get("symbol"),
            side=ctx.get("side"),
            quantity=ctx.get("quantity"),
        )
        return {"order_id": order_id, "order_status": "submitted"}

    async def _trade_cancel_order(self, ctx: dict) -> dict:
        """Compensate: cancel the placed order."""
        order_id = ctx.get("order_id")
        logger.debug("order_cancelled", order_id=order_id)
        return {"cancelled": True}

    async def _trade_confirm(self, ctx: dict) -> dict:
        """Wait for order fill confirmation."""
        order_id = ctx.get("order_id")
        fill_price = ctx.get("price", 0)
        logger.debug("order_confirmed", order_id=order_id, fill_price=fill_price)
        return {"fill_price": fill_price, "confirmed": True}

    async def _trade_revert_confirmation(self, ctx: dict) -> dict:
        """Compensate: mark confirmation as reverted."""
        logger.debug("confirmation_reverted", order_id=ctx.get("order_id"))
        return {"confirmation_reverted": True}

    async def _trade_update_portfolio(self, ctx: dict) -> dict:
        """Apply the filled order to the portfolio state."""
        portfolio_id = ctx.get("portfolio_id")
        symbol = ctx.get("symbol")
        quantity = ctx.get("quantity")
        fill_price = ctx.get("fill_price", ctx.get("price", 0))
        logger.debug(
            "portfolio_updated",
            portfolio_id=portfolio_id,
            symbol=symbol,
            quantity=quantity,
            fill_price=fill_price,
        )
        return {"portfolio_updated": True}

    async def _trade_revert_portfolio(self, ctx: dict) -> dict:
        """Compensate: undo portfolio changes."""
        logger.debug("portfolio_reverted", portfolio_id=ctx.get("portfolio_id"))
        return {"portfolio_reverted": True}


class _SagaStepError(Exception):
    """Internal sentinel raised when a step exhausts its retries."""

    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        self.message = message
        super().__init__(f"{step_name}: {message}")
