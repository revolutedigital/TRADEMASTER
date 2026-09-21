"""Download verified 1-second BTCUSDT data and screen executed-flow hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.stats import t

from scripts.research.fx_fast10_tournament import (
    Candidate,
    TournamentData,
    _top_training_atoms,
    enumerate_cluster_regimes,
    enumerate_intersections,
    enumerate_model_rankings,
    enumerate_tail_candidates,
    enumerate_tree_regimes,
    mean_metrics,
    select_for_audit,
)
from scripts.research.fx_fast4_materialize import _sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast11"
SYMBOL = "BTCUSDT"
DAYS = tuple(f"2026-09-{day:02d}" for day in range(13, 20))
HORIZONS_SECONDS = (5, 30, 120)
BASE_COST_BPS = 5.0
STRESS_COST_BPS = 20.0
FEATURE_WINDOWS = (5, 15, 60, 300)
BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
DIRECTIONAL_FEATURES = frozenset(
    {
        *(f"return_bps_{window}" for window in FEATURE_WINDOWS),
        *(f"flow_imbalance_{window}" for window in FEATURE_WINDOWS),
        *(f"close_position_{window}" for window in FEATURE_WINDOWS),
        *(f"vwap_distance_bps_{window}" for window in FEATURE_WINDOWS),
        "flow_acceleration_5_60",
        "flow_acceleration_15_300",
    }
)


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def download_verified_days(raw_root: Path) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for day in DAYS:
        filename = f"{SYMBOL}-1s-{day}.zip"
        folder = f"{BASE_URL}/{SYMBOL}/1s"
        archive = raw_root / filename
        checksum_path = raw_root / f"{filename}.CHECKSUM"
        _download(f"{folder}/{filename}", archive)
        _download(f"{folder}/{filename}.CHECKSUM", checksum_path)
        checksum_text = checksum_path.read_text(encoding="utf-8").strip()
        expected = checksum_text.split()[0]
        observed = _sha256(archive)
        if observed != expected:
            raise ValueError(f"checksum mismatch for {filename}: {observed} != {expected}")
        manifest[day] = {
            "archive": filename,
            "bytes": archive.stat().st_size,
            "sha256": observed,
            "official_checksum": expected,
        }
        sys.stdout.write(f"{day}: verified {archive.stat().st_size / 1_000_000:.2f} MB\n")
        sys.stdout.flush()
    return manifest


def read_day(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path.name}, found {members}")
        raw = archive.read(members[0])
    frame = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[numeric["open_time"].notna()].copy()
    timestamps = numeric["open_time"].to_numpy(dtype=np.int64)
    unit = "us" if np.median(timestamps) >= 100_000_000_000_000 else "ms"
    numeric.index = pd.to_datetime(timestamps, unit=unit, utc=True)
    numeric.index.name = "open_time"
    numeric = numeric.drop(columns=["open_time", "close_time", "ignore"])
    if numeric.index.has_duplicates or not numeric.index.is_monotonic_increasing:
        raise ValueError(f"invalid timestamps in {path.name}")
    return numeric


def load_market(raw_root: Path) -> pd.DataFrame:
    parts = [read_day(raw_root / f"{SYMBOL}-1s-{day}.zip") for day in DAYS]
    frame = pd.concat(parts).sort_index()
    full_index = pd.date_range(
        pd.Timestamp(f"{DAYS[0]}T00:00:00Z"),
        pd.Timestamp(f"2026-09-20T00:00:00Z") - pd.Timedelta(seconds=1),
        freq="s",
    )
    return frame.reindex(full_index)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    log_return = np.log(close).diff()
    features: dict[str, pd.Series] = {}
    quote_sums: dict[int, pd.Series] = {}
    trade_sums: dict[int, pd.Series] = {}
    flow: dict[int, pd.Series] = {}
    for window in FEATURE_WINDOWS:
        high = frame["high"].rolling(window, min_periods=window).max()
        low = frame["low"].rolling(window, min_periods=window).min()
        quote = frame["quote_volume"].rolling(window, min_periods=window).sum()
        base = frame["volume"].rolling(window, min_periods=window).sum()
        trades = frame["trades"].rolling(window, min_periods=window).sum()
        taker_buy = frame["taker_buy_quote_volume"].rolling(window, min_periods=window).sum()
        imbalance = 2 * taker_buy / quote.replace(0, np.nan) - 1
        vwap = quote / base.replace(0, np.nan)
        spread = high - low
        features[f"return_bps_{window}"] = np.log(close / close.shift(window)) * 10_000
        features[f"volatility_bps_{window}"] = (
            log_return.rolling(window, min_periods=window).std() * 10_000
        )
        features[f"range_bps_{window}"] = (high / low - 1) * 10_000
        features[f"log_quote_volume_{window}"] = np.log1p(quote)
        features[f"log_trades_{window}"] = np.log1p(trades)
        features[f"flow_imbalance_{window}"] = imbalance
        features[f"close_position_{window}"] = (
            (2 * (close - low) / spread - 1).where(spread != 0, 0.0)
        )
        features[f"vwap_distance_bps_{window}"] = (close / vwap - 1) * 10_000
        quote_sums[window] = quote
        trade_sums[window] = trades
        flow[window] = imbalance
    features["volume_acceleration_5_60"] = quote_sums[5] / (quote_sums[60] / 12) - 1
    features["volume_acceleration_15_300"] = quote_sums[15] / (quote_sums[300] / 20) - 1
    features["trade_acceleration_5_60"] = trade_sums[5] / (trade_sums[60] / 12) - 1
    features["trade_acceleration_15_300"] = trade_sums[15] / (trade_sums[300] / 20) - 1
    features["flow_acceleration_5_60"] = flow[5] - flow[60]
    features["flow_acceleration_15_300"] = flow[15] - flow[300]
    return pd.DataFrame(features, index=frame.index).replace([np.inf, -np.inf], np.nan)


def build_horizon_data(
    market: pd.DataFrame, features: pd.DataFrame, horizon_seconds: int
) -> TournamentData:
    decisions = np.asarray(features.index.second % 5 == 0)
    entry = market["open"].shift(-1)
    exit_price = market["close"].shift(-horizon_seconds)
    gross_long = (exit_price / entry - 1) * 10_000
    feature_names = list(features.columns)
    finite_features = np.isfinite(features.to_numpy(dtype=np.float32)).all(axis=1)
    present = market["close"].notna().to_numpy()
    missing_prefix = np.concatenate(([0], np.cumsum(~present)))
    positions = np.arange(len(market))
    complete_path = np.zeros(len(market), dtype=bool)
    eligible = positions + horizon_seconds < len(market)
    eligible_positions = positions[eligible]
    complete_path[eligible] = (
        missing_prefix[eligible_positions + horizon_seconds + 1]
        - missing_prefix[eligible_positions + 1]
        == 0
    )
    valid = (
        decisions
        & finite_features
        & entry.notna().to_numpy()
        & exit_price.notna().to_numpy()
        & complete_path
    )
    selected_features = features.loc[valid]
    long_return = gross_long.loc[valid].to_numpy(dtype=np.float32)
    matrices: list[np.ndarray] = []
    indexes: list[pd.DatetimeIndex] = []
    base_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    for side in (1, -1):
        matrix = selected_features.to_numpy(dtype=np.float32, copy=True)
        for column, name in enumerate(feature_names):
            if name in DIRECTIONAL_FEATURES:
                matrix[:, column] *= side
        matrices.append(
            np.concatenate((matrix, np.full((len(matrix), 1), side, dtype=np.float32)), axis=1)
        )
        indexes.append(pd.DatetimeIndex(selected_features.index))
        gross = side * long_return
        base_parts.append(gross - BASE_COST_BPS)
        stress_parts.append(gross - STRESS_COST_BPS)
    matrix = np.concatenate(matrices)
    index = indexes[0].append(indexes[1:])
    base = np.concatenate(base_parts)
    stress = np.concatenate(stress_parts)
    order = np.argsort(index.asi8, kind="stable")
    return TournamentData(
        matrix=matrix[order],
        names=tuple([*feature_names, "side"]),
        index=index[order],
        base_r=base[order],
        stress_r=stress[order],
    )


def crypto_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray(index < pd.Timestamp("2026-09-17T00:00:00Z")),
        "selection": np.asarray(
            (index >= pd.Timestamp("2026-09-17T00:00:00Z"))
            & (index < pd.Timestamp("2026-09-18T00:00:00Z"))
        ),
        "audit": np.asarray(index >= pd.Timestamp("2026-09-18T00:00:00Z")),
    }


def enumerate_horizon(
    data: TournamentData, masks: dict[str, np.ndarray]
) -> tuple[list[Candidate], list[dict[str, object]], dict[str, int]]:
    tails = enumerate_tail_candidates(data, masks["train"])
    atoms = _top_training_atoms(tails, masks["train"], data)
    families = {
        "tail": tails,
        "intersection": enumerate_intersections(atoms, masks["train"], data),
        "tree": enumerate_tree_regimes(data, masks["train"]),
        "cluster": enumerate_cluster_regimes(data, masks["train"]),
        "model": enumerate_model_rankings(data, masks["train"]),
    }
    candidates = [candidate for values in families.values() for candidate in values]
    finalists, selection = select_for_audit(candidates, masks, data)
    return finalists, selection, {name: len(values) for name, values in families.items()}


def utc_clustered_lower_bound(
    values: np.ndarray, index: pd.DatetimeIndex, alpha: float
) -> tuple[float, int]:
    realized = np.asarray(values, dtype=np.float64)
    days = index.normalize().asi8
    unique_days, inverse = np.unique(days, return_inverse=True)
    clusters = len(unique_days)
    mean = float(realized.mean())
    if clusters < 2:
        return float("nan"), clusters
    scores = np.bincount(inverse, weights=realized - mean)
    variance = clusters / (clusters - 1) * float(np.square(scores).sum()) / len(realized) ** 2
    critical = float(t.ppf(1 - alpha, df=clusters - 1))
    return mean - critical * np.sqrt(max(variance, 0.0)), clusters


def audit_all(
    finalists: list[tuple[int, Candidate, TournamentData, dict[str, np.ndarray]]]
) -> list[dict[str, object]]:
    if not finalists:
        return []
    alpha = 0.05 / len(finalists)
    reports = []
    for horizon, candidate, data, masks in finalists:
        mask = candidate.mask & masks["audit"]
        metrics = mean_metrics(mask, data.base_r, data.stress_r)
        if metrics["rows"]:
            base_lcb, days = utc_clustered_lower_bound(
                data.base_r[mask], data.index[mask], alpha
            )
            stress_lcb, stress_days = utc_clustered_lower_bound(
                data.stress_r[mask], data.index[mask], alpha
            )
        else:
            base_lcb, stress_lcb = float("nan"), float("nan")
            days, stress_days = 0, 0
        gate = (
            metrics["rows"] >= 100
            and days == 2
            and stress_days == 2
            and metrics["mean_base_r"] > 0
            and metrics["mean_stress_r"] > 0
            and base_lcb > 0
            and stress_lcb > 0
        )
        reports.append(
            {
                "horizon_seconds": horizon,
                "name": candidate.name,
                "family": candidate.family,
                "specification": candidate.specification,
                **metrics,
                "utc_days": days,
                "bonferroni_alpha": alpha,
                "lower_95_base_bps": base_lcb,
                "lower_95_stress_bps": stress_lcb,
                "pilot_gate": gate,
            }
        )
    return reports


def run(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    raw_root = root / "raw"
    source_manifest = download_verified_days(raw_root)
    market = load_market(raw_root)
    features = build_features(market)
    all_finalists: list[tuple[int, Candidate, TournamentData, dict[str, np.ndarray]]] = []
    selection_reports: dict[str, list[dict[str, object]]] = {}
    generated: dict[str, dict[str, int]] = {}
    row_counts: dict[str, dict[str, int]] = {}
    for horizon in HORIZONS_SECONDS:
        data = build_horizon_data(market, features, horizon)
        masks = crypto_masks(data.index)
        finalists, selection, by_family = enumerate_horizon(data, masks)
        all_finalists.extend((horizon, candidate, data, masks) for candidate in finalists)
        selection_reports[str(horizon)] = selection
        generated[str(horizon)] = by_family
        row_counts[str(horizon)] = {name: int(mask.sum()) for name, mask in masks.items()}
        sys.stdout.write(
            f"h{horizon}: {sum(by_family.values())} hypotheses, {len(finalists)} finalists\n"
        )
        sys.stdout.flush()
    audit = audit_all(all_finalists)
    passed = [item for item in audit if item["pilot_gate"]]
    winner = max(passed, key=lambda item: item["mean_stress_r"], default=None)
    report = {
        "kind": "round11_centralized_executed_flow_pilot",
        "symbol": SYMBOL,
        "source": "Binance Vision official public 1s klines",
        "source_manifest": source_manifest,
        "protected_samples_opened": False,
        "orders_sent": False,
        "cost_hurdles_bps": [BASE_COST_BPS, 10.0, STRESS_COST_BPS],
        "rows": row_counts,
        "generated_by_horizon_and_family": generated,
        "audited_by_family": dict(Counter(item[1].family for item in all_finalists)),
        "selection": selection_reports,
        "audit": audit,
        "winner": winner,
        "status": "p0_pass" if winner else "p0_rejected",
    }
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "executed-flow-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_path = output / "source-manifest.json"
    source_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        f"round11: {len(audit)} audited, {len(passed)} passed, status={report['status']}, "
        f"sha256={_sha256(report_path)}\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args(argv)
    run(arguments.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
