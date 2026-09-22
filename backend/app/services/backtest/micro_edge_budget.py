"""Upper-bound economic feasibility analysis before any signal modelling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

from app.services.backtest.cost_model import RoundTripCostModel


@dataclass(frozen=True)
class HorizonEdgeBudget:
    horizon_seconds: int
    observations: int
    expected_round_trip_bps: float
    stress_round_trip_bps: float
    oracle_mfe_p50_bps: float
    oracle_mfe_p75_bps: float
    oracle_mfe_p90_bps: float
    oracle_mfe_p95_bps: float
    oracle_mfe_p99_bps: float
    terminal_abs_p90_bps: float
    expected_cost_clear_rate: float
    stress_cost_clear_rate: float
    stress_oracle_p90_net_bps: float
    stress_oracle_p99_net_bps: float
    feasibility_gate: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeBudgetReport:
    source_label: str
    decision_stride_seconds: int
    oracle_warning: str
    horizons: tuple[HorizonEdgeBudget, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_label": self.source_label,
            "decision_stride_seconds": self.decision_stride_seconds,
            "oracle_warning": self.oracle_warning,
            "horizons": [horizon.to_dict() for horizon in self.horizons],
        }


def compute_edge_budget(
    market: pd.DataFrame,
    *,
    horizons_seconds: tuple[int, ...],
    expected_cost: RoundTripCostModel,
    stress_cost: RoundTripCostModel,
    decision_stride_seconds: int = 5,
    source_label: str = "event-price series",
) -> EdgeBudgetReport:
    """Measure an optimistic movement ceiling, never a tradeable strategy edge.

    ``market`` needs positive ``close``, ``high``, and ``low`` columns at one
    observation per second. For each decision, the oracle chooses the favorable
    direction after seeing the future. Therefore this function can reject an
    economically impossible horizon but can never approve a signal.
    """
    _validate_market(market)
    if not horizons_seconds or any(horizon <= 0 for horizon in horizons_seconds):
        raise ValueError("horizons_seconds must contain positive values")
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if stress_cost.round_trip_bps < expected_cost.round_trip_bps:
        raise ValueError("Stress cost cannot be lower than expected cost")

    close = market["close"].astype(float)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    reports: list[HorizonEdgeBudget] = []

    for horizon in sorted(set(horizons_seconds)):
        forward_window = FixedForwardWindowIndexer(window_size=horizon)
        future_high = high.shift(-1).rolling(forward_window, min_periods=horizon).max()
        future_low = low.shift(-1).rolling(forward_window, min_periods=horizon).min()
        terminal = close.shift(-horizon)

        long_mfe = np.log(future_high / close) * 10_000
        short_mfe = np.log(close / future_low) * 10_000
        oracle_mfe = pd.concat((long_mfe, short_mfe), axis=1).max(axis=1)
        terminal_abs = np.abs(np.log(terminal / close) * 10_000)

        decision_mask = np.arange(len(market)) % decision_stride_seconds == 0
        valid_mask = (
            decision_mask
            & np.isfinite(oracle_mfe.to_numpy())
            & np.isfinite(terminal_abs.to_numpy())
        )
        selected_mfe = oracle_mfe.to_numpy(dtype=np.float64)[valid_mask]
        selected_terminal = terminal_abs.to_numpy(dtype=np.float64)[valid_mask]
        if not len(selected_mfe):
            raise ValueError(f"No complete paths for {horizon}s horizon")

        percentiles = np.percentile(selected_mfe, [50, 75, 90, 95, 99])
        stress_p90_net = float(percentiles[2] - stress_cost.round_trip_bps)
        stress_p99_net = float(percentiles[4] - stress_cost.round_trip_bps)
        stress_clear_rate = float(np.mean(selected_mfe > stress_cost.round_trip_bps))
        reports.append(
            HorizonEdgeBudget(
                horizon_seconds=horizon,
                observations=len(selected_mfe),
                expected_round_trip_bps=expected_cost.round_trip_bps,
                stress_round_trip_bps=stress_cost.round_trip_bps,
                oracle_mfe_p50_bps=float(percentiles[0]),
                oracle_mfe_p75_bps=float(percentiles[1]),
                oracle_mfe_p90_bps=float(percentiles[2]),
                oracle_mfe_p95_bps=float(percentiles[3]),
                oracle_mfe_p99_bps=float(percentiles[4]),
                terminal_abs_p90_bps=float(np.percentile(selected_terminal, 90)),
                expected_cost_clear_rate=float(
                    np.mean(selected_mfe > expected_cost.round_trip_bps)
                ),
                stress_cost_clear_rate=stress_clear_rate,
                stress_oracle_p90_net_bps=stress_p90_net,
                stress_oracle_p99_net_bps=stress_p99_net,
                feasibility_gate=stress_p99_net > 0 and stress_clear_rate >= 0.01,
            )
        )

    return EdgeBudgetReport(
        source_label=source_label,
        decision_stride_seconds=decision_stride_seconds,
        oracle_warning=(
            "The oracle selects direction after observing the path. It is only an upper-bound "
            "feasibility diagnostic and cannot approve a signal or authorize execution."
        ),
        horizons=tuple(reports),
    )


def _validate_market(market: pd.DataFrame) -> None:
    required_columns = {"close", "high", "low"}
    missing = required_columns - set(market.columns)
    if missing:
        raise ValueError(f"Market frame is missing columns: {sorted(missing)}")
    if len(market) < 2:
        raise ValueError("Market frame needs at least two rows")
    values = market[["close", "high", "low"]].to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    if finite_rows.mean() < 0.90:
        raise ValueError("At least 90% of market rows must be complete")
    finite_values = values[finite_rows]
    if (finite_values <= 0).any():
        raise ValueError("Market prices must be positive")
    if (finite_values[:, 1] < finite_values[:, 2]).any():
        raise ValueError("High price cannot be below low price")
    if not math.isfinite(float(values[finite_rows, 0].mean())):
        raise ValueError("Market prices are not numerically stable")
