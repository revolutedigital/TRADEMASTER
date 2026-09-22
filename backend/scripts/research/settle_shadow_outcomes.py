"""Settle mature research-only shadow outcomes from historical trade replay.

Default mode is a dry-run: it reads pending mature shadow signals and replays
their paths, but does not write outcomes. Use --commit to append immutable
outcome evidence to the research ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.models.base import async_session_factory
from app.services.backtest.trailing_portfolio import (
    V1_TRAILING_POLICIES,
    HistoricalTrailingSimulator,
    TrailingPolicy,
)
from app.services.research.research_dataset import load_trade_interval
from app.services.research.shadow_outcome_settlement import (
    ShadowOutcomeSettlement,
    list_mature_pending_shadow_signals,
    settle_shadow_signals,
)
from app.services.research.shadow_recorder import research_shadow_recorder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--trade-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "normalized"
        / "aggTrades",
    )
    parser.add_argument(
        "--policy-name",
        required=True,
        choices=tuple(policy.name for policy in V1_TRAILING_POLICIES),
    )
    parser.add_argument("--now", type=_parse_datetime)
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-round-trip-bps", type=float, default=12.0)
    parser.add_argument("--stress-round-trip-bps", type=float, default=24.0)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Append immutable replay outcomes to the research ledger.",
    )
    arguments = parser.parse_args()
    if arguments.limit <= 0:
        parser.error("--limit must be positive")
    if arguments.latency_ms < 0:
        parser.error("--latency-ms cannot be negative")
    if arguments.expected_round_trip_bps < 0 or arguments.stress_round_trip_bps < 0:
        parser.error("round-trip costs cannot be negative")
    if arguments.stress_round_trip_bps < arguments.expected_round_trip_bps:
        parser.error("--stress-round-trip-bps cannot be below expected cost")

    report = asyncio.run(
        _run(
            experiment_id=arguments.experiment_id,
            trade_root=arguments.trade_root,
            policy=_policy_by_name(arguments.policy_name),
            now=arguments.now,
            limit=arguments.limit,
            latency_ms=arguments.latency_ms,
            expected_round_trip_bps=arguments.expected_round_trip_bps,
            stress_round_trip_bps=arguments.stress_round_trip_bps,
            commit=arguments.commit,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))  # noqa: T201
    return 0


async def _run(
    *,
    experiment_id: str,
    trade_root: Path,
    policy: TrailingPolicy,
    now: datetime | None,
    limit: int,
    latency_ms: int,
    expected_round_trip_bps: float,
    stress_round_trip_bps: float,
    commit: bool,
) -> dict[str, object]:
    settlement_time = _normalize_utc(now or datetime.now(UTC))
    async with async_session_factory() as session:
        try:
            signals = await list_mature_pending_shadow_signals(
                session,
                experiment_id=experiment_id,
                now=settlement_time,
                limit=limit,
            )
            if not signals:
                if commit:
                    await session.commit()
                return _empty_report(policy=policy, settlement_time=settlement_time, commit=commit)

            start = min(_normalize_utc(signal.decision_time) for signal in signals) - timedelta(
                seconds=1
            )
            end = max(
                _normalize_utc(signal.decision_time) + timedelta(seconds=signal.horizon_seconds)
                for signal in signals
            ) + timedelta(seconds=1)
            trades = load_trade_interval(
                trade_root,
                start,
                end,
                expected_product="usdm_perpetual",
            )
            simulator = HistoricalTrailingSimulator(
                trades["event_time_ms"].to_numpy(),
                trades["price"].to_numpy(),
                latency_ms=latency_ms,
                expected_round_trip_bps=expected_round_trip_bps,
                stress_round_trip_bps=stress_round_trip_bps,
            )
            settlements = settle_shadow_signals(list(signals), simulator=simulator, policy=policy)
            if commit:
                for settlement in settlements:
                    await research_shadow_recorder.record_outcome(
                        session,
                        **settlement.to_recorder_kwargs(),
                    )
                await session.commit()
            report = _settlement_report(
                settlements=settlements,
                policy=policy,
                settlement_time=settlement_time,
                commit=commit,
                trade_start=start,
                trade_end=end,
            )
            if not commit:
                await session.rollback()
            return report
        except Exception:
            await session.rollback()
            raise


def _settlement_report(
    *,
    settlements: tuple[ShadowOutcomeSettlement, ...],
    policy: TrailingPolicy,
    settlement_time: datetime,
    commit: bool,
    trade_start: datetime,
    trade_end: datetime,
) -> dict[str, object]:
    expected_values = [settlement.expected_net_bps for settlement in settlements]
    stress_values = [settlement.stress_net_bps for settlement in settlements]
    outcome_count = len(settlements)
    decision_dates = {
        settlement.decision_time.astimezone(UTC).date().isoformat()
        for settlement in settlements
    }
    decision_day_count = len(decision_dates)
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "committed": commit,
        "dry_run": not commit,
        "settlement_time": settlement_time.isoformat(),
        "policy": asdict(policy),
        "signal_count": outcome_count,
        "outcome_count": outcome_count,
        "decision_day_count": decision_day_count,
        "outcome_day_count": decision_day_count,
        "complete": True,
        "incomplete_signal_ids": [],
        "expected_mean_bps": _mean(expected_values),
        "stress_mean_bps": _mean(stress_values),
        "trade_interval": {
            "start": trade_start.isoformat(),
            "end": trade_end.isoformat(),
        },
        "outcomes": [
            {
                "signal_id": settlement.signal_id,
                "decision_time": settlement.decision_time.isoformat(),
                "would_enter": settlement.would_enter,
                "expected_net_bps": settlement.expected_net_bps,
                "stress_net_bps": settlement.stress_net_bps,
                "label_sha256": settlement.label_sha256,
                "policy_name": settlement.policy_name,
            }
            for settlement in settlements
        ],
    }


def _empty_report(
    *,
    policy: TrailingPolicy,
    settlement_time: datetime,
    commit: bool,
) -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "committed": commit,
        "dry_run": not commit,
        "settlement_time": settlement_time.isoformat(),
        "policy": asdict(policy),
        "signal_count": 0,
        "outcome_count": 0,
        "decision_day_count": 0,
        "outcome_day_count": 0,
        "complete": True,
        "incomplete_signal_ids": [],
        "expected_mean_bps": None,
        "stress_mean_bps": None,
        "outcomes": [],
    }


def _policy_by_name(name: str) -> TrailingPolicy:
    for policy in V1_TRAILING_POLICIES:
        if policy.name == name:
            return policy
    raise ValueError(f"unknown trailing policy: {name}")


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return _normalize_utc(datetime.fromisoformat(normalized))


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
