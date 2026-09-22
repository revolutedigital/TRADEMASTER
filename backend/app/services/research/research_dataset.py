"""Materialize causal features and exact path labels into immutable partitions."""

from __future__ import annotations

import hashlib
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
) -> pd.DataFrame:
    """Build side/horizon rows; all event data must include the full label horizon."""
    dataset_config = config or ResearchDatasetConfig()
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
) -> ResearchPartitionResult:
    dataset_config = config or ResearchDatasetConfig()
    day_start = datetime.combine(utc_date, datetime.min.time(), tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    history_start = day_start - timedelta(seconds=max(dataset_config.feature_windows_seconds))
    label_end = day_end + timedelta(seconds=max(dataset_config.horizons_seconds))
    trades = load_trade_interval(source_root, history_start, label_end)
    stride_ms = dataset_config.decision_stride_seconds * 1000
    decisions = np.arange(
        int(day_start.timestamp() * 1000),
        int(day_end.timestamp() * 1000),
        stride_ms,
        dtype=np.int64,
    )
    event_times = trades["event_time_ms"].to_numpy(dtype=np.int64)
    history_ready = decisions - max(dataset_config.feature_windows_seconds) * 1000
    complete = (history_ready >= event_times[0]) & (
        decisions + max(dataset_config.horizons_seconds) * 1000 <= event_times[-1]
    )
    decisions = decisions[complete]
    if not len(decisions):
        raise ValueError(f"no complete decisions for {utc_date.isoformat()}")
    rows = build_research_rows(trades, decisions, dataset_config)
    partition_dir = output_root / f"date={utc_date.isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_dir / "research_rows.parquet"
    temporary_path = partition_dir / ".research_rows.parquet.tmp"
    rows.to_parquet(temporary_path, index=False, compression="zstd")
    os.replace(temporary_path, output_path)
    digest = _sha256(output_path)
    manifest = {
        "schema_version": 1,
        "utc_date": utc_date.isoformat(),
        "row_count": len(rows),
        "decision_count": len(decisions),
        "config_sha256": dataset_config.sha256,
        "normalized_sha256": digest,
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
    source_root: Path, interval_start: datetime, interval_end: datetime
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = interval_start.date()
    while cursor <= interval_end.date():
        path = source_root / f"date={cursor.isoformat()}" / "events.parquet"
        if path.exists():
            table = pq.read_table(
                path,
                columns=[
                    "event_time",
                    "sequence_id",
                    "price",
                    "quantity",
                    "is_buyer_maker",
                ],
            )
            frame = table.to_pandas()
            frame["event_time_ms"] = frame["event_time"].astype("int64", copy=False)
            start_ms = int(interval_start.timestamp() * 1000)
            end_ms = int(interval_end.timestamp() * 1000)
            frame = frame[(frame["event_time_ms"] >= start_ms) & (frame["event_time_ms"] <= end_ms)]
            frames.append(frame.drop(columns="event_time"))
        cursor += timedelta(days=1)
    if not frames:
        raise FileNotFoundError("no normalized trade partitions overlap the interval")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["event_time_ms", "sequence_id"], kind="stable"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
