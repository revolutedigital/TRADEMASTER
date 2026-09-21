"""Screen frozen executed-flow entries with breakeven and trailing management."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import t

from scripts.research.fx_fast10_tournament import Candidate, TournamentData
from scripts.research.fx_fast11_flow import (
    BASE_COST_BPS,
    BASE_URL,
    DEFAULT_ROOT as FAST11_ROOT,
    STRESS_COST_BPS,
    SYMBOL,
    _download,
    build_features,
    build_horizon_data,
    crypto_masks,
    enumerate_horizon,
    load_market,
    read_day,
)
from scripts.research.fx_fast4_materialize import _sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast12"
AUDIT_DAY = "2026-09-20"
MANAGEMENT_SELECTION_START = pd.Timestamp("2026-09-18T00:00:00Z")
AUDIT_START = pd.Timestamp(f"{AUDIT_DAY}T00:00:00Z")
HOLD_BY_SIGNAL = {5: 30, 30: 120, 120: 300}
COSTS_BPS = np.asarray([BASE_COST_BPS, STRESS_COST_BPS], dtype=np.float64)


@dataclass(frozen=True)
class Management:
    initial_stop_bps: float
    activation_bps: float
    trail_distance_bps: float


MANAGEMENTS = (
    Management(5, 3, 2),
    Management(10, 5, 3),
    Management(10, 5, 5),
    Management(20, 10, 5),
    Management(20, 10, 10),
    Management(40, 20, 10),
    Management(40, 20, 20),
)


def download_audit_day(raw_root: Path) -> dict[str, object]:
    filename = f"{SYMBOL}-1s-{AUDIT_DAY}.zip"
    folder = f"{BASE_URL}/{SYMBOL}/1s"
    archive = raw_root / filename
    checksum_path = raw_root / f"{filename}.CHECKSUM"
    _download(f"{folder}/{filename}", archive)
    _download(f"{folder}/{filename}.CHECKSUM", checksum_path)
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    observed = _sha256(archive)
    if observed != expected:
        raise ValueError(f"checksum mismatch for {filename}: {observed} != {expected}")
    return {
        "archive": filename,
        "bytes": archive.stat().st_size,
        "sha256": observed,
        "official_checksum": expected,
    }


def load_extended_market(raw_root: Path) -> pd.DataFrame:
    original = load_market(raw_root)
    audit = read_day(raw_root / f"{SYMBOL}-1s-{AUDIT_DAY}.zip")
    frame = pd.concat((original, audit)).sort_index()
    full_index = pd.date_range(
        original.index[0],
        pd.Timestamp("2026-09-21T00:00:00Z") - pd.Timedelta(seconds=1),
        freq="s",
    )
    return frame.reindex(full_index)


@njit(cache=True)
def _managed_kernel(  # noqa: PLR0913
    opens,
    highs,
    lows,
    closes,
    decision_positions,
    sides,
    max_hold_seconds,
    initial_stop_bps,
    activation_bps,
    trail_distance_bps,
    costs_bps,
):
    output = np.full((len(decision_positions), len(costs_bps)), np.nan)
    size = len(opens)
    for event in range(len(decision_positions)):
        decision = decision_positions[event]
        entry_position = decision + 1
        final_position = decision + max_hold_seconds
        if final_position >= size:
            continue
        entry = opens[entry_position]
        if not np.isfinite(entry) or entry <= 0:
            continue
        complete = True
        for bar in range(entry_position, final_position + 1):
            if not (
                np.isfinite(opens[bar])
                and np.isfinite(highs[bar])
                and np.isfinite(lows[bar])
                and np.isfinite(closes[bar])
            ):
                complete = False
                break
        if not complete:
            continue
        side = sides[event]
        for scenario in range(len(costs_bps)):
            cost = costs_bps[scenario]
            stop = -initial_stop_bps
            best = 0.0
            exited = False
            for bar in range(entry_position, final_position + 1):
                if side == 1:
                    open_bps = np.log(opens[bar] / entry) * 10_000.0
                    favorable = np.log(highs[bar] / entry) * 10_000.0
                    adverse = np.log(lows[bar] / entry) * 10_000.0
                    close_bps = np.log(closes[bar] / entry) * 10_000.0
                else:
                    open_bps = np.log(entry / opens[bar]) * 10_000.0
                    favorable = np.log(entry / lows[bar]) * 10_000.0
                    adverse = np.log(entry / highs[bar]) * 10_000.0
                    close_bps = np.log(entry / closes[bar]) * 10_000.0

                if open_bps <= stop:
                    output[event, scenario] = open_bps - cost
                    exited = True
                    break
                if adverse <= stop:
                    output[event, scenario] = stop - cost
                    exited = True
                    break

                best = max(best, favorable)
                if best >= max(activation_bps, cost):
                    stop = max(stop, cost, best - trail_distance_bps)
                    if adverse <= stop:
                        output[event, scenario] = stop - cost
                        exited = True
                        break

                if bar == final_position:
                    output[event, scenario] = close_bps - cost
                    exited = True
                    break
            if not exited:
                output[event, scenario] = np.nan
    return output


def simulate_managed_batch(
    market: pd.DataFrame,
    event_index: pd.DatetimeIndex,
    sides: np.ndarray,
    *,
    max_hold_seconds: int,
    management: Management,
) -> np.ndarray:
    positions = market.index.get_indexer(event_index)
    if (positions < 0).any():
        raise ValueError("event timestamps missing from market")
    return _managed_kernel(
        market["open"].to_numpy(dtype=np.float64),
        market["high"].to_numpy(dtype=np.float64),
        market["low"].to_numpy(dtype=np.float64),
        market["close"].to_numpy(dtype=np.float64),
        positions.astype(np.int64),
        np.asarray(sides, dtype=np.int8),
        max_hold_seconds,
        management.initial_stop_bps,
        management.activation_bps,
        management.trail_distance_bps,
        COSTS_BPS,
    )


def half_hour_lower_bound(
    values: np.ndarray, index: pd.DatetimeIndex, alpha: float
) -> tuple[float, int]:
    realized = np.asarray(values, dtype=np.float64)
    if len(realized) == 0:
        return float("nan"), 0
    blocks = index.floor("30min").asi8
    unique_blocks, inverse = np.unique(blocks, return_inverse=True)
    clusters = len(unique_blocks)
    mean = float(realized.mean())
    if clusters < 2:
        return float("nan"), clusters
    scores = np.bincount(inverse, weights=realized - mean)
    variance = clusters / (clusters - 1) * float(np.square(scores).sum()) / len(realized) ** 2
    critical = float(t.ppf(1 - alpha, df=clusters - 1))
    return mean - critical * np.sqrt(max(variance, 0.0)), clusters


def _entry_finalists(features: pd.DataFrame, market: pd.DataFrame):
    values: dict[int, tuple[TournamentData, list[Candidate]]] = {}
    for horizon in HOLD_BY_SIGNAL:
        data = build_horizon_data(market, features, horizon)
        finalists, _, _ = enumerate_horizon(data, crypto_masks(data.index))
        if len(finalists) != 25:
            raise ValueError(f"expected 25 frozen h{horizon} finalists, found {len(finalists)}")
        values[horizon] = (data, finalists)
    return values


def _selection_metrics(mask: np.ndarray, outcomes: np.ndarray) -> dict[str, object]:
    valid = mask & np.isfinite(outcomes).all(axis=1)
    return {
        "rows": int(valid.sum()),
        "mean_base_bps": float(outcomes[valid, 0].mean()) if valid.any() else float("nan"),
        "mean_stress_bps": float(outcomes[valid, 1].mean()) if valid.any() else float("nan"),
    }


def run(root: Path = DEFAULT_ROOT, source_root: Path = FAST11_ROOT) -> dict[str, object]:
    raw_root = source_root / "raw"
    audit_source = download_audit_day(raw_root)
    market = load_extended_market(raw_root)
    features = build_features(market)
    entries = _entry_finalists(features, market)
    candidates: list[dict[str, object]] = []
    outcome_cache: dict[tuple[int, int], tuple[pd.DatetimeIndex, np.ndarray]] = {}

    for signal_horizon, (data, finalists) in entries.items():
        period = np.asarray(data.index >= MANAGEMENT_SELECTION_START)
        period_index = data.index[period]
        period_sides = data.matrix[period, -1].astype(np.int8)
        selection_period = np.asarray(period_index < AUDIT_START)
        hold = HOLD_BY_SIGNAL[signal_horizon]
        for management_index, management in enumerate(MANAGEMENTS):
            outcomes = simulate_managed_batch(
                market,
                period_index,
                period_sides,
                max_hold_seconds=hold,
                management=management,
            )
            outcome_cache[(signal_horizon, management_index)] = (period_index, outcomes)
            for entry_index, entry in enumerate(finalists):
                entry_period = entry.mask[period]
                metrics = _selection_metrics(entry_period & selection_period, outcomes)
                candidates.append(
                    {
                        "signal_horizon_seconds": signal_horizon,
                        "max_hold_seconds": hold,
                        "entry_index": entry_index,
                        "entry_name": entry.name,
                        "entry_family": entry.family,
                        "entry_specification": entry.specification,
                        "management_index": management_index,
                        "management": asdict(management),
                        **metrics,
                    }
                )
        sys.stdout.write(f"h{signal_horizon}: 175 managed hypotheses evaluated\n")
        sys.stdout.flush()

    eligible = [item for item in candidates if item["rows"] >= 100]
    ranked = sorted(
        eligible,
        key=lambda item: (
            item["mean_stress_bps"],
            item["mean_base_bps"],
            item["rows"],
        ),
        reverse=True,
    )
    finalists = ranked[:25]
    alpha = 0.05 / len(finalists) if finalists else 0.05
    audit: list[dict[str, object]] = []
    for rank, item in enumerate(finalists, start=1):
        signal_horizon = int(item["signal_horizon_seconds"])
        management_index = int(item["management_index"])
        entry_index = int(item["entry_index"])
        data, entry_candidates = entries[signal_horizon]
        period = np.asarray(data.index >= MANAGEMENT_SELECTION_START)
        period_index, outcomes = outcome_cache[(signal_horizon, management_index)]
        mask = entry_candidates[entry_index].mask[period] & np.asarray(period_index >= AUDIT_START)
        valid = mask & np.isfinite(outcomes).all(axis=1)
        base = outcomes[valid, 0]
        stress = outcomes[valid, 1]
        base_lcb, blocks = half_hour_lower_bound(base, period_index[valid], alpha)
        stress_lcb, stress_blocks = half_hour_lower_bound(stress, period_index[valid], alpha)
        mean_base = float(base.mean()) if len(base) else float("nan")
        mean_stress = float(stress.mean()) if len(stress) else float("nan")
        gate = (
            len(base) >= 50
            and blocks >= 24
            and stress_blocks >= 24
            and mean_base > 0
            and mean_stress > 0
            and base_lcb > 0
            and stress_lcb > 0
        )
        audit.append(
            {
                "rank": rank,
                **item,
                "audit_rows": len(base),
                "audit_mean_base_bps": mean_base,
                "audit_mean_stress_bps": mean_stress,
                "half_hour_blocks": blocks,
                "bonferroni_alpha": alpha,
                "lower_95_base_bps": base_lcb,
                "lower_95_stress_bps": stress_lcb,
                "pilot_gate": gate,
            }
        )

    passed = [item for item in audit if item["pilot_gate"]]
    winner = max(passed, key=lambda item: item["audit_mean_stress_bps"], default=None)
    report = {
        "kind": "round12_signal_breakeven_trailing_audit",
        "symbol": SYMBOL,
        "audit_source": audit_source,
        "audit_day": AUDIT_DAY,
        "orders_sent": False,
        "costs_bps": COSTS_BPS.tolist(),
        "generated_hypotheses": len(candidates),
        "eligible_hypotheses": len(eligible),
        "audited_hypotheses": len(audit),
        "selection": finalists,
        "audit": audit,
        "winner": winner,
        "status": "p0_pass" if winner else "managed_architecture_rejected",
    }
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "managed-flow-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        f"round12: {len(candidates)} generated, {len(audit)} audited, {len(passed)} passed, "
        f"status={report['status']}, sha256={_sha256(report_path)}\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=FAST11_ROOT)
    arguments = parser.parse_args(argv)
    run(arguments.root, arguments.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
