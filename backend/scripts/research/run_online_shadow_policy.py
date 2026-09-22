"""Score frozen top-p policy from chronological microstructure events.

Default mode is a dry-run: it reads normalized archive/WAL partitions, derives
causal online features, scores the frozen research policy, and prints the shadow
entries that would be recorded. Use --commit only after the prospective shadow
partition has been explicitly opened in the research ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.models.base import async_session_factory
from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.research.microstructure_features import TRADE_WINDOWS_SECONDS
from app.services.research.online_shadow_policy import score_online_frozen_top_p_shadow_events
from app.services.research.research_dataset import (
    load_book_interval,
    load_liquidation_interval,
    load_mark_interval,
    load_trade_interval,
)
from app.services.research.shadow_policy_runner import record_frozen_top_p_shadow_selection


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_TRADE_ROOT = DEFAULT_DATA_ROOT / "normalized" / "aggTrades"
DEFAULT_DECISION_STRIDE_SECONDS = 5


@dataclass(frozen=True)
class EventRoots:
    trade: Path
    book: Path | None = None
    mark: Path | None = None
    liquidation: Path | None = None
    spot: Path | None = None

    def as_report(self) -> dict[str, str | None]:
        return {
            "trade": str(self.trade),
            "book": str(self.book) if self.book is not None else None,
            "mark": str(self.mark) if self.mark is not None else None,
            "liquidation": str(self.liquidation) if self.liquidation is not None else None,
            "spot": str(self.spot) if self.spot is not None else None,
        }


@dataclass(frozen=True)
class LoadedOnlineEvents:
    events: tuple[MicrostructureEvent, ...]
    counts: dict[str, int]
    roots: EventRoots
    load_start: datetime
    load_end: datetime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--policy-artifact", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", type=_parse_datetime, required=True)
    parser.add_argument("--end", type=_parse_datetime, required=True)
    parser.add_argument("--decision-stride-seconds", type=int, default=DEFAULT_DECISION_STRIDE_SECONDS)
    parser.add_argument(
        "--history-seconds",
        type=int,
        help="Lookback loaded before --start. Defaults to the max online feature window.",
    )
    parser.add_argument(
        "--wal-root",
        type=Path,
        help=(
            "Prospective recorder root with trade/depth/mark_price/liquidation/spot_trade "
            "children. Explicit roots override these derived paths."
        ),
    )
    parser.add_argument("--trade-root", type=Path)
    parser.add_argument("--book-root", type=Path)
    parser.add_argument("--mark-root", type=Path)
    parser.add_argument("--liquidation-root", type=Path)
    parser.add_argument("--spot-root", type=Path)
    parser.add_argument("--sides", nargs="+", default=["BUY", "SELL"])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to atomically write the scored online shadow report.",
    )
    parser.add_argument(
        "--include-non-entries",
        action="store_true",
        help="Record every scored decision, not just probability >= top-p threshold.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Append selected online shadow signals to the research ledger.",
    )
    arguments = parser.parse_args()
    if arguments.end <= arguments.start:
        parser.error("--end must be after --start")
    if arguments.decision_stride_seconds <= 0:
        parser.error("--decision-stride-seconds must be positive")
    if arguments.history_seconds is not None and arguments.history_seconds < 0:
        parser.error("--history-seconds cannot be negative")
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be positive")
    roots = _resolve_roots(arguments)
    artifact = json.loads(arguments.policy_artifact.read_text(encoding="utf-8"))
    report = asyncio.run(
        _run(
            experiment_id=arguments.experiment_id,
            artifact=artifact,
            symbol=arguments.symbol.upper(),
            start=arguments.start,
            end=arguments.end,
            decision_stride_seconds=arguments.decision_stride_seconds,
            history_seconds=arguments.history_seconds,
            roots=roots,
            sides=tuple(arguments.sides),
            include_non_entries=arguments.include_non_entries,
            limit=arguments.limit,
            commit=arguments.commit,
        )
    )
    if arguments.output is not None:
        _write_json_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))  # noqa: T201
    return 0


async def _run(
    *,
    experiment_id: str,
    artifact: dict[str, Any],
    symbol: str,
    start: datetime,
    end: datetime,
    decision_stride_seconds: int,
    history_seconds: int | None,
    roots: EventRoots,
    sides: tuple[str, ...],
    include_non_entries: bool,
    limit: int | None,
    commit: bool,
) -> dict[str, object]:
    loaded = load_online_microstructure_events(
        artifact=artifact,
        symbol=symbol,
        start=start,
        end=end,
        history_seconds=history_seconds,
        roots=roots,
    )
    decision_times_ms = decision_times_between(start, end, stride_seconds=decision_stride_seconds)
    selection = score_online_frozen_top_p_shadow_events(
        loaded.events,
        artifact=artifact,
        decision_times_ms=decision_times_ms,
        sides=sides,
        include_non_entries=include_non_entries,
    )
    if commit:
        async with async_session_factory() as session:
            try:
                result = await record_frozen_top_p_shadow_selection(
                    session,
                    experiment_id=experiment_id,
                    selection=selection,
                    limit=limit,
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return _report(
            selection=selection,
            loaded=loaded,
            start=start,
            end=end,
            decision_stride_seconds=decision_stride_seconds,
            decision_times_ms=decision_times_ms,
            limit=limit,
            commit=True,
            commit_result=asdict(result),
        )
    return _report(
        selection=selection,
        loaded=loaded,
        start=start,
        end=end,
        decision_stride_seconds=decision_stride_seconds,
        decision_times_ms=decision_times_ms,
        limit=limit,
        commit=False,
        commit_result=None,
    )


def load_online_microstructure_events(
    *,
    artifact: dict[str, Any],
    symbol: str,
    start: datetime,
    end: datetime,
    roots: EventRoots,
    history_seconds: int | None = None,
) -> LoadedOnlineEvents:
    """Load normalized research partitions as chronological online events."""
    required_sources = required_event_sources(artifact)
    _raise_for_missing_required_roots(required_sources, roots)
    lookback_seconds = max(TRADE_WINDOWS_SECONDS) if history_seconds is None else history_seconds
    load_start = start - timedelta(seconds=lookback_seconds)
    trade_frame = load_trade_interval(
        roots.trade,
        load_start,
        end,
        expected_product="usdm_perpetual",
    )
    book_frame = _load_optional_frame(
        "book",
        roots.book,
        load_book_interval,
        load_start,
        end,
        required=required_sources["book"],
    )
    mark_frame = _load_optional_frame(
        "mark",
        roots.mark,
        load_mark_interval,
        load_start,
        end,
        required=required_sources["mark"],
    )
    liquidation_frame = _load_optional_frame(
        "liquidation",
        roots.liquidation,
        load_liquidation_interval,
        load_start,
        end,
        required=required_sources["liquidation"],
    )
    spot_frame = _load_optional_trade_frame(
        "spot",
        roots.spot,
        load_start,
        end,
        required=required_sources["spot"],
    )
    _raise_for_missing_required_events(
        required_sources,
        {
            "book": len(book_frame),
            "mark": len(mark_frame),
            "liquidation": len(liquidation_frame),
            "spot": len(spot_frame),
        },
    )
    events = _events_from_frames(
        symbol=symbol,
        trade_frame=trade_frame,
        book_frame=book_frame,
        mark_frame=mark_frame,
        liquidation_frame=liquidation_frame,
        spot_frame=spot_frame,
    )
    return LoadedOnlineEvents(
        events=events,
        counts={
            "perp_trade": len(trade_frame),
            "book": len(book_frame),
            "mark": len(mark_frame),
            "liquidation": len(liquidation_frame),
            "spot_trade": len(spot_frame),
            "total_events": len(events),
        },
        roots=roots,
        load_start=load_start,
        load_end=end,
    )


def decision_times_between(
    start: datetime,
    end: datetime,
    *,
    stride_seconds: int,
) -> tuple[int, ...]:
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    start_ms = int(_normalize_utc(start).timestamp() * 1000)
    end_ms = int(_normalize_utc(end).timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")
    stride_ms = stride_seconds * 1000
    decision_times: list[int] = []
    current = start_ms
    while current < end_ms:
        decision_times.append(current)
        current += stride_ms
    return tuple(decision_times)


def required_event_sources(artifact: dict[str, Any]) -> dict[str, bool]:
    columns = tuple(str(column) for column in artifact.get("feature_columns", ()))
    book_base = {
        "book_available",
        "book_update_age_ms",
        "spread_bps",
        "depth_imbalance",
        "microprice_displacement_bps",
        "directed_depth_imbalance",
        "directed_microprice_displacement_bps",
    }
    mark_base = {
        "mark_available",
        "mark_update_age_ms",
        "mark_index_basis_bps",
        "directed_mark_index_basis_bps",
        "funding_rate",
        "directed_funding_rate",
    }
    return {
        "book": any(
            column in book_base
            or column.startswith("book_")
            or column.startswith("directed_book_")
            for column in columns
        ),
        "mark": any(column in mark_base for column in columns),
        "liquidation": any(
            column.startswith("liquidation_") or column.startswith("directed_liquidation_")
            for column in columns
        ),
        "spot": any(column.startswith("spot_") or column.startswith("directed_spot_") for column in columns),
    }


def _resolve_roots(arguments: argparse.Namespace) -> EventRoots:
    wal_root = arguments.wal_root
    default_trade_root = wal_root / "trade" if wal_root is not None else DEFAULT_TRADE_ROOT
    return EventRoots(
        trade=arguments.trade_root or default_trade_root,
        book=arguments.book_root or (wal_root / "depth" if wal_root is not None else None),
        mark=arguments.mark_root or (wal_root / "mark_price" if wal_root is not None else None),
        liquidation=arguments.liquidation_root
        or (wal_root / "liquidation" if wal_root is not None else None),
        spot=arguments.spot_root or (wal_root / "spot_trade" if wal_root is not None else None),
    )


def _load_optional_frame(
    name: str,
    root: Path | None,
    loader: Any,
    start: datetime,
    end: datetime,
    *,
    required: bool,
) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()
    if not root.exists():
        if required:
            raise FileNotFoundError(f"{name} root does not exist: {root}")
        return pd.DataFrame()
    try:
        return loader(root, start, end)
    except FileNotFoundError:
        if required:
            raise
        return pd.DataFrame()


def _load_optional_trade_frame(
    name: str,
    root: Path | None,
    start: datetime,
    end: datetime,
    *,
    required: bool,
) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()
    if not root.exists():
        if required:
            raise FileNotFoundError(f"{name} root does not exist: {root}")
        return pd.DataFrame()
    try:
        return load_trade_interval(root, start, end, expected_product="spot")
    except FileNotFoundError:
        if required:
            raise
        return pd.DataFrame()


def _events_from_frames(
    *,
    symbol: str,
    trade_frame: pd.DataFrame,
    book_frame: pd.DataFrame,
    mark_frame: pd.DataFrame,
    liquidation_frame: pd.DataFrame,
    spot_frame: pd.DataFrame,
) -> tuple[MicrostructureEvent, ...]:
    indexed_events: list[tuple[int, int, int, MicrostructureEvent]] = []
    indexed_events.extend(_trade_events(trade_frame, symbol=symbol, product="usdm_perpetual", rank=10))
    indexed_events.extend(_trade_events(spot_frame, symbol=symbol, product="spot", rank=20))
    indexed_events.extend(_book_events(book_frame, symbol=symbol, rank=30))
    indexed_events.extend(_mark_events(mark_frame, symbol=symbol, rank=40))
    indexed_events.extend(_liquidation_events(liquidation_frame, symbol=symbol, rank=50))
    indexed_events.sort(key=lambda item: (item[0], item[2], item[1]))
    return tuple(event for _event_time_ms, _sequence_id, _rank, event in indexed_events)


def _trade_events(
    frame: pd.DataFrame,
    *,
    symbol: str,
    product: str,
    rank: int,
) -> list[tuple[int, int, int, MicrostructureEvent]]:
    events: list[tuple[int, int, int, MicrostructureEvent]] = []
    for row in frame.to_dict("records"):
        event_time_ms = int(row["event_time_ms"])
        sequence_id = _sequence_id(row)
        price = _positive_float(row["price"], column="price")
        quantity = _non_negative_float(row["quantity"], column="quantity")
        is_buyer_maker = bool(row["is_buyer_maker"])
        event = MicrostructureEvent(
            product=product,
            symbol=symbol,
            event_type=MarketEventType.TRADE,
            event_time=_datetime_from_ms(event_time_ms),
            sequence_id=sequence_id,
            price=price,
            quantity=quantity,
            quote_quantity=price * quantity,
            is_buyer_maker=is_buyer_maker,
            side="SELL" if is_buyer_maker else "BUY",
        )
        events.append((event_time_ms, sequence_id or 0, rank, event))
    return events


def _book_events(
    frame: pd.DataFrame,
    *,
    symbol: str,
    rank: int,
) -> list[tuple[int, int, int, MicrostructureEvent]]:
    events: list[tuple[int, int, int, MicrostructureEvent]] = []
    for row in frame.to_dict("records"):
        event_time_ms = int(row["event_time_ms"])
        sequence_id = _sequence_id(row)
        event = MicrostructureEvent(
            product="usdm_perpetual",
            symbol=symbol,
            event_type=MarketEventType.DEPTH,
            event_time=_datetime_from_ms(event_time_ms),
            sequence_id=sequence_id,
            bid_price=_positive_float(row["bid_price"], column="bid_price"),
            bid_quantity=_non_negative_float(row["bid_quantity"], column="bid_quantity"),
            ask_price=_positive_float(row["ask_price"], column="ask_price"),
            ask_quantity=_non_negative_float(row["ask_quantity"], column="ask_quantity"),
        )
        events.append((event_time_ms, sequence_id or 0, rank, event))
    return events


def _mark_events(
    frame: pd.DataFrame,
    *,
    symbol: str,
    rank: int,
) -> list[tuple[int, int, int, MicrostructureEvent]]:
    events: list[tuple[int, int, int, MicrostructureEvent]] = []
    for row in frame.to_dict("records"):
        event_time_ms = int(row["event_time_ms"])
        event = MicrostructureEvent(
            product="usdm_perpetual",
            symbol=symbol,
            event_type=MarketEventType.MARK_PRICE,
            event_time=_datetime_from_ms(event_time_ms),
            price=_positive_float(row["mark_price"], column="mark_price"),
            payload={
                "index_price": _positive_float(row["index_price"], column="index_price"),
                "funding_rate": float(row["funding_rate"]),
            },
        )
        events.append((event_time_ms, 0, rank, event))
    return events


def _liquidation_events(
    frame: pd.DataFrame,
    *,
    symbol: str,
    rank: int,
) -> list[tuple[int, int, int, MicrostructureEvent]]:
    events: list[tuple[int, int, int, MicrostructureEvent]] = []
    for row in frame.to_dict("records"):
        event_time_ms = int(row["event_time_ms"])
        sequence_id = _sequence_id(row)
        price = _optional_positive_float(row.get("price"), column="price")
        event = MicrostructureEvent(
            product="usdm_perpetual",
            symbol=symbol,
            event_type=MarketEventType.LIQUIDATION,
            event_time=_datetime_from_ms(event_time_ms),
            sequence_id=sequence_id,
            price=price,
            quantity=_non_negative_float(row["quantity"], column="quantity"),
            side=str(row["side"]).upper(),
        )
        events.append((event_time_ms, sequence_id or 0, rank, event))
    return events


def _report(
    *,
    selection: Any,
    loaded: LoadedOnlineEvents,
    start: datetime,
    end: datetime,
    decision_stride_seconds: int,
    decision_times_ms: tuple[int, ...],
    limit: int | None,
    commit: bool,
    commit_result: dict[str, object] | None,
) -> dict[str, object]:
    decisions = selection.decisions[:limit] if limit is not None else selection.decisions
    report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": selection.order_submission_allowed,
        "execution_authorization": selection.execution_authorization,
        "committed": commit,
        "dry_run": not commit,
        "model_sha256": selection.model_sha256,
        "scored_rows": selection.scored_rows,
        "selected_count": selection.selected_count,
        "selected_count_after_limit": len(decisions),
        "decision_count": len(decision_times_ms),
        "decision_stride_seconds": decision_stride_seconds,
        "decision_interval": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "loaded_interval": {
            "start": loaded.load_start.isoformat(),
            "end": loaded.load_end.isoformat(),
        },
        "event_counts": loaded.counts,
        "event_roots": loaded.roots.as_report(),
        "commit_required_to_write_ledger": not commit,
        "decisions": [
            {
                "decision_time": decision.decision_time.isoformat(),
                "side": decision.side,
                "horizon_seconds": decision.horizon_seconds,
                "probability": decision.probability,
                "threshold": decision.threshold,
                "would_enter": decision.would_enter,
                "model_sha256": decision.model_sha256,
                "feature_vector_sha256": decision.feature_vector_sha256,
            }
            for decision in decisions
        ],
    }
    if commit_result is not None:
        report["commit_result"] = commit_result
    return report


def _raise_for_missing_required_roots(
    required_sources: dict[str, bool],
    roots: EventRoots,
) -> None:
    missing = [
        name
        for name, required in required_sources.items()
        if required and getattr(roots, name) is None
    ]
    if missing:
        raise ValueError(f"policy artifact requires missing event roots: {sorted(missing)}")


def _raise_for_missing_required_events(
    required_sources: dict[str, bool],
    event_counts: dict[str, int],
) -> None:
    missing = [
        name
        for name, required in required_sources.items()
        if required and event_counts.get(name, 0) <= 0
    ]
    if missing:
        raise ValueError(f"policy artifact requires unavailable event data: {sorted(missing)}")


def _sequence_id(row: dict[str, object]) -> int | None:
    value = row.get("sequence_id")
    if value is None or pd.isna(value):
        return None
    return int(value)


def _positive_float(value: object, *, column: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{column} must be positive")
    return parsed


def _optional_positive_float(value: object, *, column: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = float(value)
    if parsed <= 0:
        return None
    return parsed


def _non_negative_float(value: object, *, column: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{column} cannot be negative")
    return parsed


def _datetime_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return _normalize_utc(datetime.fromisoformat(normalized))


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
