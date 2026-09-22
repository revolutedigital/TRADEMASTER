"""Materialize causal features and exact path labels into immutable partitions."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.services.backtest.event_replay import OrderSide
from app.services.research.microstructure_features import (
    TRADE_WINDOWS_SECONDS,
    materialize_book_features,
    materialize_liquidation_features,
    materialize_mark_features,
    materialize_spot_perp_features,
    materialize_trade_flow_features,
)
from app.services.research.path_labels import EventPathLabeler, PathLabelConfig


@dataclass(frozen=True)
class ResearchDatasetConfig:
    decision_stride_seconds: int = 5
    horizons_seconds: tuple[int, ...] = (120, 300)
    expected_cost_bps: float = 12.0
    stress_cost_bps: float = 24.0
    initial_stop_bps: float = 20.0
    feature_windows_seconds: tuple[int, ...] = TRADE_WINDOWS_SECONDS
    require_book_features: bool = False
    max_book_staleness_ms: int = 1_000

    def path_label_config(self) -> PathLabelConfig:
        return PathLabelConfig(
            horizons_seconds=self.horizons_seconds,
            expected_cost_bps=self.expected_cost_bps,
            stress_cost_bps=self.stress_cost_bps,
            initial_stop_bps=self.initial_stop_bps,
        )

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ResearchPartitionResult:
    utc_date: str
    row_count: int
    decision_count: int
    sha256: str
    output_path: str


def build_research_rows(
    trades: pd.DataFrame,
    decision_times_ms: np.ndarray,
    config: ResearchDatasetConfig | None = None,
    *,
    book_events: pd.DataFrame | None = None,
    mark_events: pd.DataFrame | None = None,
    liquidation_events: pd.DataFrame | None = None,
    spot_trade_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build side/horizon rows; all event data must include the full label horizon."""
    dataset_config = config or ResearchDatasetConfig()
    if dataset_config.max_book_staleness_ms < 0:
        raise ValueError("max book staleness cannot be negative")
    required = {
        "event_time_ms",
        "sequence_id",
        "price",
        "quantity",
        "is_buyer_maker",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trade frame is missing columns: {sorted(missing)}")
    ordered = trades.sort_values(["event_time_ms", "sequence_id"], kind="stable")
    times = ordered["event_time_ms"].to_numpy(dtype=np.int64)
    sequences = ordered["sequence_id"].to_numpy(dtype=np.int64)
    prices = ordered["price"].to_numpy(dtype=np.float64)
    decisions = np.asarray(decision_times_ms, dtype=np.int64)
    features = materialize_trade_flow_features(
        ordered,
        decisions,
        windows_seconds=dataset_config.feature_windows_seconds,
    )
    if book_events is not None or dataset_config.require_book_features:
        if book_events is None:
            raise ValueError("book features are required but no book events were provided")
        book_features = materialize_book_features(
            book_events,
            decisions,
            windows_seconds=dataset_config.feature_windows_seconds,
        )
        if dataset_config.require_book_features:
            _require_complete_book_features(
                book_features,
                max_staleness_ms=dataset_config.max_book_staleness_ms,
            )
        features = features.merge(book_features, on="decision_time_ms", how="left")
    if mark_events is not None:
        mark_features = materialize_mark_features(mark_events, decisions)
        features = features.merge(mark_features, on="decision_time_ms", how="left")
    if liquidation_events is not None:
        liquidation_features = materialize_liquidation_features(
            liquidation_events,
            decisions,
            windows_seconds=dataset_config.feature_windows_seconds,
        )
        features = features.merge(liquidation_features, on="decision_time_ms", how="left")
    if spot_trade_events is not None:
        spot_perp_features = materialize_spot_perp_features(
            spot_trade_events,
            ordered,
            decisions,
            windows_seconds=dataset_config.feature_windows_seconds,
        )
        features = features.merge(spot_perp_features, on="decision_time_ms", how="left")
    labeler = EventPathLabeler(
        times,
        sequences,
        prices,
        config=dataset_config.path_label_config(),
    )
    label_rows: list[dict[str, object]] = []
    for decision_time_ms in decisions:
        for side in (OrderSide.BUY, OrderSide.SELL):
            for horizon in dataset_config.horizons_seconds:
                label = labeler.label(int(decision_time_ms), side, horizon)
                row = label.to_dict()
                row["side"] = side.value
                row["first_touch"] = label.first_touch.value
                label_rows.append(row)
    labels = pd.DataFrame(label_rows)
    rows = labels.merge(features, on="decision_time_ms", how="left", validate="many_to_one")
    rows["side_sign"] = np.where(rows["side"] == OrderSide.BUY.value, 1.0, -1.0)
    for window in dataset_config.feature_windows_seconds:
        rows[f"directed_flow_imbalance_{window}s"] = (
            rows[f"flow_imbalance_{window}s"] * rows["side_sign"]
        )
        rows[f"directed_return_{window}s_bps"] = rows[f"return_{window}s_bps"] * rows["side_sign"]
        directional_columns = (
            f"book_pressure_imbalance_{window}s",
            f"book_depth_imbalance_change_{window}s",
            f"book_microprice_displacement_change_bps_{window}s",
            f"liquidation_net_qty_{window}s",
            f"liquidation_net_notional_{window}s",
            f"spot_flow_imbalance_{window}s",
            f"spot_return_{window}s_bps",
            f"spot_perp_flow_gap_{window}s",
            f"spot_perp_return_gap_{window}s_bps",
        )
        for column in directional_columns:
            if column in rows.columns:
                rows[f"directed_{column}"] = rows[column] * rows["side_sign"]
    if "depth_imbalance" in rows.columns:
        rows["directed_depth_imbalance"] = rows["depth_imbalance"] * rows["side_sign"]
    if "microprice_displacement_bps" in rows.columns:
        rows["directed_microprice_displacement_bps"] = (
            rows["microprice_displacement_bps"] * rows["side_sign"]
        )
    if "mark_index_basis_bps" in rows.columns:
        rows["directed_mark_index_basis_bps"] = rows["mark_index_basis_bps"] * rows["side_sign"]
    if "funding_rate" in rows.columns:
        rows["directed_funding_rate"] = -rows["funding_rate"] * rows["side_sign"]
    if "spot_perp_basis_bps" in rows.columns:
        rows["directed_spot_perp_basis_bps"] = rows["spot_perp_basis_bps"] * rows["side_sign"]
    rows["target"] = rows["paid_expected_before_stop"].astype(np.int8)
    return rows.sort_values(
        ["decision_time_ms", "side", "horizon_seconds"], kind="stable"
    ).reset_index(drop=True)


def build_research_partition(
    *,
    source_root: Path,
    output_root: Path,
    utc_date: date,
    config: ResearchDatasetConfig | None = None,
    book_source_root: Path | None = None,
    mark_source_root: Path | None = None,
    liquidation_source_root: Path | None = None,
    spot_source_root: Path | None = None,
) -> ResearchPartitionResult:
    dataset_config = config or ResearchDatasetConfig()
    day_start = datetime.combine(utc_date, datetime.min.time(), tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    history_start = day_start - timedelta(seconds=max(dataset_config.feature_windows_seconds))
    label_end = day_end + timedelta(seconds=max(dataset_config.horizons_seconds))
    trades = load_trade_interval(
        source_root,
        history_start,
        label_end,
        expected_product="usdm_perpetual",
    )
    stride_ms = dataset_config.decision_stride_seconds * 1000
    decisions = np.arange(
        int(day_start.timestamp() * 1000),
        int(day_end.timestamp() * 1000),
        stride_ms,
        dtype=np.int64,
    )
    event_times = trades["event_time_ms"].to_numpy(dtype=np.int64)
    book_events = (
        load_book_interval(book_source_root, history_start, day_end)
        if book_source_root is not None
        else None
    )
    mark_events = (
        load_mark_interval(mark_source_root, history_start, day_end)
        if mark_source_root is not None
        else None
    )
    liquidation_events = (
        load_liquidation_interval(liquidation_source_root, history_start, day_end)
        if liquidation_source_root is not None
        else None
    )
    spot_trade_events = (
        load_trade_interval(
            spot_source_root,
            history_start,
            day_end,
            expected_product="spot",
        )
        if spot_source_root is not None
        else None
    )
    history_ready = decisions - max(dataset_config.feature_windows_seconds) * 1000
    complete = (history_ready >= event_times[0]) & (
        decisions + max(dataset_config.horizons_seconds) * 1000 <= event_times[-1]
    )
    decisions = decisions[complete]
    if not len(decisions):
        raise ValueError(f"no complete decisions for {utc_date.isoformat()}")
    rows = build_research_rows(
        trades,
        decisions,
        dataset_config,
        book_events=book_events,
        mark_events=mark_events,
        liquidation_events=liquidation_events,
        spot_trade_events=spot_trade_events,
    )
    partition_dir = output_root / f"date={utc_date.isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_dir / "research_rows.parquet"
    temporary_path = partition_dir / ".research_rows.parquet.tmp"
    rows.to_parquet(temporary_path, index=False, compression="zstd")
    os.replace(temporary_path, output_path)
    digest = _sha256(output_path)
    manifest = {
        "schema_version": 2,
        "utc_date": utc_date.isoformat(),
        "row_count": len(rows),
        "decision_count": len(decisions),
        "config": asdict(dataset_config),
        "config_sha256": dataset_config.sha256,
        "research_rows_sha256": digest,
        "normalized_sha256": digest,
        "input_sources": {
            "futures_trades": _source_metadata(
                role="futures_trades",
                source_root=source_root,
                frame=trades,
                interval_start=history_start,
                interval_end=label_end,
            ),
            "book": _source_metadata(
                role="book",
                source_root=book_source_root,
                frame=book_events,
                interval_start=history_start,
                interval_end=day_end,
            ),
            "mark": _source_metadata(
                role="mark",
                source_root=mark_source_root,
                frame=mark_events,
                interval_start=history_start,
                interval_end=day_end,
            ),
            "liquidation": _source_metadata(
                role="liquidation",
                source_root=liquidation_source_root,
                frame=liquidation_events,
                interval_start=history_start,
                interval_end=day_end,
            ),
            "spot_trades": _source_metadata(
                role="spot_trades",
                source_root=spot_source_root,
                frame=spot_trade_events,
                interval_start=history_start,
                interval_end=day_end,
            ),
        },
        "feature_families": _feature_family_manifest(rows),
        "source_partition_count": (
            datetime.fromtimestamp(event_times[-1] / 1000, tz=UTC).date()
            - datetime.fromtimestamp(event_times[0] / 1000, tz=UTC).date()
        ).days
        + 1,
    }
    manifest_path = partition_dir / "research_rows.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ResearchPartitionResult(
        utc_date=utc_date.isoformat(),
        row_count=len(rows),
        decision_count=len(decisions),
        sha256=digest,
        output_path=str(output_path),
    )


def load_trade_interval(
    source_root: Path,
    interval_start: datetime,
    interval_end: datetime,
    *,
    expected_product: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = interval_start.date()
    while cursor <= interval_end.date():
        partition_dir = source_root / f"date={cursor.isoformat()}"
        parquet_path = partition_dir / "events.parquet"
        jsonl_path = partition_dir / "events.jsonl.gz"
        if parquet_path.exists():
            columns = [
                "event_time",
                "sequence_id",
                "price",
                "quantity",
                "is_buyer_maker",
            ]
            if expected_product is not None:
                parquet_columns = set(pq.read_schema(parquet_path).names)
                if "product" not in parquet_columns:
                    raise ValueError(
                        f"trade partition {parquet_path} is missing product column"
                    )
                columns.append("product")
            table = pq.read_table(
                parquet_path,
                columns=columns,
            )
            frame = table.to_pandas()
        elif jsonl_path.exists():
            frame = _read_trade_wal_jsonl(jsonl_path)
        else:
            cursor += timedelta(days=1)
            continue
        frame = _prepare_trade_frame(
            frame,
            interval_start,
            interval_end,
            expected_product=expected_product,
        )
        if not frame.empty:
            frames.append(frame)
        cursor += timedelta(days=1)
    if not frames:
        raise FileNotFoundError("no normalized trade partitions overlap the interval")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["event_time_ms", "sequence_id"], kind="stable"
    )


def load_book_interval(
    source_root: Path, interval_start: datetime, interval_end: datetime
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = interval_start.date()
    while cursor <= interval_end.date():
        partition_dir = source_root / f"date={cursor.isoformat()}"
        parquet_path = partition_dir / "events.parquet"
        jsonl_path = partition_dir / "events.jsonl.gz"
        if parquet_path.exists():
            table = pq.read_table(
                parquet_path,
                columns=[
                    "event_time",
                    "sequence_id",
                    "bid_price",
                    "bid_quantity",
                    "ask_price",
                    "ask_quantity",
                ],
            )
            frame = table.to_pandas()
        elif jsonl_path.exists():
            frame = _read_book_wal_jsonl(jsonl_path)
        else:
            cursor += timedelta(days=1)
            continue
        frame = _prepare_book_frame(frame, interval_start, interval_end)
        if not frame.empty:
            frames.append(frame)
        cursor += timedelta(days=1)
    if not frames:
        raise FileNotFoundError("no top-of-book partitions overlap the interval")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["event_time_ms", "sequence_id"], kind="stable"
    )


def load_mark_interval(
    source_root: Path, interval_start: datetime, interval_end: datetime
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = interval_start.date()
    while cursor <= interval_end.date():
        partition_dir = source_root / f"date={cursor.isoformat()}"
        parquet_path = partition_dir / "events.parquet"
        jsonl_path = partition_dir / "events.jsonl.gz"
        if parquet_path.exists():
            table = pq.read_table(
                parquet_path,
                columns=["event_time", "price", "index_price", "funding_rate"],
            )
            frame = table.to_pandas()
        elif jsonl_path.exists():
            frame = _read_mark_wal_jsonl(jsonl_path)
        else:
            cursor += timedelta(days=1)
            continue
        frame = _prepare_mark_frame(frame, interval_start, interval_end)
        if not frame.empty:
            frames.append(frame)
        cursor += timedelta(days=1)
    if not frames:
        raise FileNotFoundError("no mark-price partitions overlap the interval")
    return pd.concat(frames, ignore_index=True).sort_values("event_time_ms", kind="stable")


def load_liquidation_interval(
    source_root: Path, interval_start: datetime, interval_end: datetime
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = interval_start.date()
    while cursor <= interval_end.date():
        partition_dir = source_root / f"date={cursor.isoformat()}"
        parquet_path = partition_dir / "events.parquet"
        jsonl_path = partition_dir / "events.jsonl.gz"
        if parquet_path.exists():
            table = pq.read_table(
                parquet_path,
                columns=["event_time", "sequence_id", "price", "quantity", "side"],
            )
            frame = table.to_pandas()
        elif jsonl_path.exists():
            frame = _read_liquidation_wal_jsonl(jsonl_path)
        else:
            cursor += timedelta(days=1)
            continue
        frame = _prepare_liquidation_frame(frame, interval_start, interval_end)
        if not frame.empty:
            frames.append(frame)
        cursor += timedelta(days=1)
    if not frames:
        raise FileNotFoundError("no liquidation partitions overlap the interval")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["event_time_ms", "sequence_id"], kind="stable"
    )


def _read_trade_wal_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                {
                    "event_time": payload.get("event_time"),
                    "product": payload.get("product"),
                    "sequence_id": payload.get("sequence_id"),
                    "price": payload.get("price"),
                    "quantity": payload.get("quantity"),
                    "is_buyer_maker": payload.get("is_buyer_maker"),
                }
            )
    return pd.DataFrame(rows)


def _read_book_wal_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                {
                    "event_time": payload.get("event_time"),
                    "sequence_id": payload.get("sequence_id"),
                    "bid_price": payload.get("bid_price"),
                    "bid_quantity": payload.get("bid_quantity"),
                    "ask_price": payload.get("ask_price"),
                    "ask_quantity": payload.get("ask_quantity"),
                }
            )
    return pd.DataFrame(rows)


def _read_mark_wal_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
            rows.append(
                {
                    "event_time": payload.get("event_time"),
                    "price": payload.get("price"),
                    "index_price": nested.get("index_price"),
                    "funding_rate": nested.get("funding_rate"),
                }
            )
    return pd.DataFrame(rows)


def _read_liquidation_wal_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                {
                    "event_time": payload.get("event_time"),
                    "sequence_id": payload.get("sequence_id"),
                    "price": payload.get("price"),
                    "quantity": payload.get("quantity"),
                    "side": payload.get("side"),
                }
            )
    return pd.DataFrame(rows)


def _prepare_trade_frame(
    frame: pd.DataFrame,
    interval_start: datetime,
    interval_end: datetime,
    *,
    expected_product: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["event_time_ms", "sequence_id", "price", "quantity", "is_buyer_maker"]
        )
    prepared = frame.copy()
    prepared["event_time"] = pd.to_datetime(prepared["event_time"], utc=True)
    prepared["event_time_ms"] = _datetime_to_epoch_ms(prepared["event_time"])
    start_ms = int(interval_start.timestamp() * 1000)
    end_ms = int(interval_end.timestamp() * 1000)
    prepared = prepared[
        (prepared["event_time_ms"] >= start_ms) & (prepared["event_time_ms"] <= end_ms)
    ]
    if expected_product is not None and not prepared.empty:
        if "product" not in prepared.columns:
            raise ValueError("trade frame is missing product column")
        observed_products = set(prepared["product"].astype(str))
        if observed_products != {expected_product}:
            raise ValueError(
                f"trade frame expected product {expected_product!r}, "
                f"found {sorted(observed_products)!r}"
            )
    if "sequence_id" in prepared.columns:
        prepared["sequence_id"] = pd.to_numeric(prepared["sequence_id"], errors="coerce").fillna(0)
    else:
        prepared["sequence_id"] = 0
    prepared["price"] = pd.to_numeric(prepared["price"], errors="raise")
    prepared["quantity"] = pd.to_numeric(prepared["quantity"], errors="raise")
    prepared["is_buyer_maker"] = prepared["is_buyer_maker"].map(_parse_boolean)
    return prepared[
        [
            "event_time_ms",
            "sequence_id",
            "price",
            "quantity",
            "is_buyer_maker",
        ]
    ]


def _prepare_book_frame(
    frame: pd.DataFrame,
    interval_start: datetime,
    interval_end: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_time_ms",
                "sequence_id",
                "bid_price",
                "bid_quantity",
                "ask_price",
                "ask_quantity",
            ]
        )
    prepared = frame.copy()
    prepared["event_time"] = pd.to_datetime(prepared["event_time"], utc=True)
    prepared["event_time_ms"] = _datetime_to_epoch_ms(prepared["event_time"])
    start_ms = int(interval_start.timestamp() * 1000)
    end_ms = int(interval_end.timestamp() * 1000)
    prepared = prepared[
        (prepared["event_time_ms"] >= start_ms) & (prepared["event_time_ms"] <= end_ms)
    ]
    for column in ("bid_price", "bid_quantity", "ask_price", "ask_quantity"):
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    if "sequence_id" in prepared.columns:
        prepared["sequence_id"] = pd.to_numeric(prepared["sequence_id"], errors="coerce").fillna(0)
    else:
        prepared["sequence_id"] = 0
    return prepared[
        [
            "event_time_ms",
            "sequence_id",
            "bid_price",
            "bid_quantity",
            "ask_price",
            "ask_quantity",
        ]
    ]


def _prepare_mark_frame(
    frame: pd.DataFrame,
    interval_start: datetime,
    interval_end: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["event_time_ms", "mark_price", "index_price", "funding_rate"]
        )
    prepared = frame.copy()
    prepared["event_time"] = pd.to_datetime(prepared["event_time"], utc=True)
    prepared["event_time_ms"] = _datetime_to_epoch_ms(prepared["event_time"])
    start_ms = int(interval_start.timestamp() * 1000)
    end_ms = int(interval_end.timestamp() * 1000)
    prepared = prepared[
        (prepared["event_time_ms"] >= start_ms) & (prepared["event_time_ms"] <= end_ms)
    ]
    mark_column = "mark_price" if "mark_price" in prepared.columns else "price"
    prepared["mark_price"] = pd.to_numeric(prepared[mark_column], errors="raise")
    prepared["index_price"] = pd.to_numeric(prepared["index_price"], errors="raise")
    prepared["funding_rate"] = pd.to_numeric(prepared["funding_rate"], errors="raise")
    return prepared[["event_time_ms", "mark_price", "index_price", "funding_rate"]]


def _prepare_liquidation_frame(
    frame: pd.DataFrame,
    interval_start: datetime,
    interval_end: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["event_time_ms", "sequence_id", "price", "quantity", "side"]
        )
    prepared = frame.copy()
    prepared["event_time"] = pd.to_datetime(prepared["event_time"], utc=True)
    prepared["event_time_ms"] = _datetime_to_epoch_ms(prepared["event_time"])
    start_ms = int(interval_start.timestamp() * 1000)
    end_ms = int(interval_end.timestamp() * 1000)
    prepared = prepared[
        (prepared["event_time_ms"] >= start_ms) & (prepared["event_time_ms"] <= end_ms)
    ]
    if "sequence_id" in prepared.columns:
        prepared["sequence_id"] = pd.to_numeric(prepared["sequence_id"], errors="coerce").fillna(0)
    else:
        prepared["sequence_id"] = 0
    prepared["price"] = pd.to_numeric(prepared.get("price", 0), errors="coerce").fillna(0)
    prepared["quantity"] = pd.to_numeric(prepared["quantity"], errors="raise")
    prepared["side"] = prepared["side"].astype(str).str.upper()
    return prepared[["event_time_ms", "sequence_id", "price", "quantity", "side"]]


def _require_complete_book_features(features: pd.DataFrame, *, max_staleness_ms: int) -> None:
    unavailable_count = int((features["book_available"] < 1).sum())
    stale_count = int((features["book_update_age_ms"] > max_staleness_ms).sum())
    if unavailable_count or stale_count:
        raise ValueError(
            "book features are incomplete: "
            f"{unavailable_count} decisions without book, "
            f"{stale_count} decisions older than {max_staleness_ms}ms"
        )


def _parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _source_metadata(
    *,
    role: str,
    source_root: Path | None,
    frame: pd.DataFrame | None,
    interval_start: datetime,
    interval_end: datetime,
) -> dict[str, object]:
    if source_root is None or frame is None:
        return {
            "role": role,
            "present": False,
        }
    metadata: dict[str, object] = {
        "role": role,
        "present": True,
        "source_root": str(source_root),
        "requested_interval_start": interval_start.isoformat(),
        "requested_interval_end": interval_end.isoformat(),
        "row_count": int(len(frame)),
    }
    if "event_time_ms" in frame.columns and not frame.empty:
        event_times = frame["event_time_ms"].to_numpy(dtype=np.int64)
        metadata["first_event_time"] = _epoch_ms_to_iso(int(event_times.min()))
        metadata["last_event_time"] = _epoch_ms_to_iso(int(event_times.max()))
        metadata["utc_partition_count"] = (
            datetime.fromtimestamp(int(event_times.max()) / 1000, tz=UTC).date()
            - datetime.fromtimestamp(int(event_times.min()) / 1000, tz=UTC).date()
        ).days + 1
    return metadata


def _feature_family_manifest(rows: pd.DataFrame) -> dict[str, object]:
    columns = tuple(rows.columns)
    return {
        "flow": any(column.startswith("flow_imbalance_") for column in columns),
        "price": any(column.startswith("return_") for column in columns),
        "session": {"hour_sin", "hour_cos", "side_sign"}.issubset(columns),
        "book": "book_available" in columns,
        "mark_funding": "mark_available" in columns or "funding_rate" in columns,
        "liquidation": any(column.startswith("liquidation_") for column in columns),
        "spot_perp": "spot_perp_basis_bps" in columns,
        "directional": any(column.startswith("directed_") for column in columns),
        "column_count": len(columns),
    }


def _epoch_ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _datetime_to_epoch_ms(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.as_unit("ms").astype("int64")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
