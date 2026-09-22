"""Generate the microstructure v1 movement-versus-cost feasibility report."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from app.services.backtest.cost_model import RoundTripCostModel, stressed_cost_model
from app.services.backtest.micro_edge_budget import EdgeBudgetReport, compute_edge_budget


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "backend" / "data" / "lab_fast11" / "raw"
DEFAULT_REPORT = REPO_ROOT / "docs" / "research" / "micro-edge-budget-report.md"
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)


def load_verified_second_bars(raw_directory: Path) -> pd.DataFrame:
    archives = sorted(raw_directory.glob("BTCUSDT-1s-*.zip"))
    if not archives:
        raise FileNotFoundError(f"No BTCUSDT 1s archives found in {raw_directory}")
    frames: list[pd.DataFrame] = []
    for archive_path in archives:
        checksum_path = archive_path.with_name(f"{archive_path.name}.CHECKSUM")
        if not checksum_path.exists():
            raise FileNotFoundError(f"Missing official checksum for {archive_path.name}")
        import hashlib

        expected_hash = checksum_path.read_text(encoding="utf-8").split()[0]
        observed_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if observed_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected one CSV in {archive_path.name}")
            frame = pd.read_csv(
                io.BytesIO(archive.read(members[0])),
                names=KLINE_COLUMNS,
                header=None,
            )
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frames.append(frame[["open_time", "high", "low", "close"]].dropna())

    market = pd.concat(frames, ignore_index=True).sort_values("open_time")
    timestamps = market["open_time"].astype("int64")
    timestamp_unit = "us" if timestamps.median() >= 100_000_000_000_000 else "ms"
    market.index = pd.to_datetime(timestamps, unit=timestamp_unit, utc=True)
    market = market.drop(columns="open_time")
    if market.index.has_duplicates or not market.index.is_monotonic_increasing:
        raise ValueError("Second-bar timestamps must be unique and increasing")
    return market


def render_markdown(report: EdgeBudgetReport) -> str:
    lines = [
        "# Microstructure v1 — economic edge budget",
        "",
        f"Source: {report.source_label}",
        "",
        f"> {report.oracle_warning}",
        "",
        "| Horizon | Paths | Expected cost | Stress cost | MFE p50 | MFE p90 | MFE p99 | "
        "Clears expected | Clears stress | Stress p99 net | Feasible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for horizon in report.horizons:
        lines.append(
            f"| {horizon.horizon_seconds}s | {horizon.observations:,} | "
            f"{horizon.expected_round_trip_bps:.2f} bps | "
            f"{horizon.stress_round_trip_bps:.2f} bps | "
            f"{horizon.oracle_mfe_p50_bps:.2f} | {horizon.oracle_mfe_p90_bps:.2f} | "
            f"{horizon.oracle_mfe_p99_bps:.2f} | "
            f"{horizon.expected_cost_clear_rate:.1%} | "
            f"{horizon.stress_cost_clear_rate:.1%} | "
            f"{horizon.stress_oracle_p99_net_bps:.2f} bps | "
            f"{'YES' if horizon.feasibility_gate else 'NO'} |"
        )
    lines.extend(
        (
            "",
            "Passing this table only means that enough price movement exists in hindsight. "
            "It does not demonstrate that any causal signal can choose the direction or capture it.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=1.0)
    arguments = parser.parse_args()

    expected = RoundTripCostModel(
        fee_bps_per_side=arguments.fee_bps_per_side,
        slippage_bps_per_side=arguments.slippage_bps_per_side,
    )
    stress = stressed_cost_model(expected)
    market = load_verified_second_bars(arguments.input)
    report = compute_edge_budget(
        market,
        horizons_seconds=(5, 15, 30, 120, 300),
        expected_cost=expected,
        stress_cost=stress,
        source_label="verified Binance Spot BTCUSDT 1-second bars, 2026-09-13 through 2026-09-20",
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(render_markdown(report), encoding="utf-8")
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
