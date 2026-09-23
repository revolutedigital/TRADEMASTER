"""Fast68 directional liquidity/volatility gate diagnostic.

Fast67 killed the pure high-liquidity/high-volatility gate: the extension block
shorted into a strong recent up-move. Fast68 treats that as a new family, not a
patch to Fast67: require active microstructure plus causal directional
alignment, then audit the family with historical leave-one-block-out and a
frozen extension check.

Because the family is discovered after seeing the Fast67 failure, survivors are
not approved strategies. They are preregistration candidates for a clean future
shadow period.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast65_causal_feature_gate_walkforward import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    one_position_summary,
)
from scripts.research.run_fast66_liquidity_volatility_gate_audit import (  # noqa: E402
    DEFAULT_FAST65_MERGED,
    load_or_build_selected_trades,
)
from scripts.research.run_fast67_frozen_liquidity_volatility_gate_extension import (  # noqa: E402
    DEFAULT_FAST56_REPORT,
    aggregate_block_summaries,
    load_extension_trades,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast68-directional-gate-diagnostic"
DEFAULT_QUOTE_VOLUME_FLOOR_QUANTILES = (0.70, 0.80, 0.90)
DEFAULT_VOLATILITY_FLOOR_QUANTILES = (0.50, 0.67, 0.75, 0.80)
DEFAULT_QUOTE_VOLUME_CAP_QUANTILES: tuple[float | None, ...] = (None, 0.95, 0.99)
DEFAULT_VOLATILITY_CAP_QUANTILES: tuple[float | None, ...] = (None, 0.90, 0.95, 0.99)
DEFAULT_DIRECTIONAL_FEATURES: tuple[str | None, ...] = (
    None,
    "directed_return_5s_bps",
    "directed_return_30s_bps",
    "directed_return_60s_bps",
    "directed_flow_imbalance_30s",
    "directed_flow_imbalance_60s",
)
DEFAULT_DIRECTIONAL_QUANTILES = (0.20, 0.33, 0.50)


@dataclass(frozen=True)
class DirectionalGateSpec:
    quote_volume_floor_quantile: float
    realized_vol_floor_quantile: float
    quote_volume_cap_quantile: float | None
    realized_vol_cap_quantile: float | None
    directional_feature: str | None
    directional_quantile: float | None

    @property
    def name(self) -> str:
        parts = [
            f"quote_volume_30s>=q{self.quote_volume_floor_quantile:.2f}",
            f"realized_vol_30s_bps>=q{self.realized_vol_floor_quantile:.2f}",
        ]
        if self.quote_volume_cap_quantile is not None:
            parts.append(f"quote_volume_30s<=q{self.quote_volume_cap_quantile:.2f}")
        if self.realized_vol_cap_quantile is not None:
            parts.append(f"realized_vol_30s_bps<=q{self.realized_vol_cap_quantile:.2f}")
        if self.directional_feature is not None:
            parts.append(f"{self.directional_feature}>=q{self.directional_quantile:.2f}")
        return " AND ".join(parts)


@dataclass(frozen=True)
class DirectionalGateThresholds:
    spec_name: str
    quote_volume_floor: float
    realized_vol_floor: float
    quote_volume_cap: float | None
    realized_vol_cap: float | None
    directional_feature: str | None
    directional_floor: float | None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fast65-merged", type=Path, default=DEFAULT_FAST65_MERGED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--historical-fast-start", type=int, default=43)
    parser.add_argument("--historical-fast-end", type=int, default=53)
    parser.add_argument("--minimum-block-trades", type=int, default=3)
    parser.add_argument("--minimum-survivor-blocks", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--extension-report", type=Path, action="append", default=None)
    arguments = parser.parse_args()

    extension_reports = (
        tuple(arguments.extension_report)
        if arguments.extension_report is not None
        else ((DEFAULT_FAST56_REPORT,) if DEFAULT_FAST56_REPORT.exists() else tuple())
    )
    report = run_fast68_directional_gate_diagnostic(
        data_root=arguments.data_root,
        fast65_merged_path=arguments.fast65_merged,
        output_root=arguments.output_root,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        minimum_block_trades=arguments.minimum_block_trades,
        minimum_survivor_blocks=arguments.minimum_survivor_blocks,
        top_n=arguments.top_n,
        extension_report_paths=extension_reports,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast68_directional_gate_diagnostic(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    output_root: Path,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    minimum_block_trades: int = 3,
    minimum_survivor_blocks: int = 5,
    top_n: int = 25,
    extension_report_paths: tuple[Path, ...] = tuple(),
) -> dict[str, Any]:
    _validate_parameters(
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        minimum_block_trades=minimum_block_trades,
        minimum_survivor_blocks=minimum_survivor_blocks,
        top_n=top_n,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    historical_trades = load_or_build_selected_trades(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
    )
    extension_trades = load_extension_trades(extension_report_paths, data_root=data_root)
    if historical_trades.empty:
        return _write_report(
            output_root,
            _base_report(
                data_root=data_root,
                fast65_merged_path=fast65_merged_path,
                historical_fast_start=historical_fast_start,
                historical_fast_end=historical_fast_end,
                minimum_block_trades=minimum_block_trades,
                minimum_survivor_blocks=minimum_survivor_blocks,
                extension_report_paths=extension_report_paths,
            )
            | {
                "status": "FAST68_NO_HISTORICAL_TRADES",
                "status_reasons": ["fast65_selected_trades_unavailable"],
            },
        )

    candidates = [
        evaluate_directional_spec(
            historical_trades,
            extension_trades,
            spec=spec,
            minimum_block_trades=minimum_block_trades,
            extension_block_names=tuple(
                path.parent.name for path in extension_report_paths if path.exists()
            ),
        )
        for spec in generate_specs(historical_trades)
    ]
    candidates.sort(key=_candidate_sort_key, reverse=True)
    survivors = [
        candidate
        for candidate in candidates
        if is_survivor(
            candidate,
            minimum_block_trades=minimum_block_trades,
            minimum_survivor_blocks=minimum_survivor_blocks,
        )
    ]
    top_candidates = candidates[:top_n]
    status, status_reasons = fast68_status(
        survivor_count=len(survivors),
        extension_available=not extension_trades.empty,
        minimum_survivor_blocks=minimum_survivor_blocks,
    )

    candidates_path = output_root / "top_candidates.json"
    candidates_path.write_text(
        json.dumps(top_candidates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    survivors_path = output_root / "survivors.json"
    survivors_path.write_text(
        json.dumps(survivors, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = _base_report(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        minimum_block_trades=minimum_block_trades,
        minimum_survivor_blocks=minimum_survivor_blocks,
        extension_report_paths=extension_report_paths,
    ) | {
        "status": status,
        "status_reasons": status_reasons,
        "candidate_count": len(candidates),
        "survivor_count": len(survivors),
        "top_candidate": top_candidates[0] if top_candidates else None,
        "top_candidates_path": str(candidates_path),
        "top_candidates_sha256": _sha256_file(candidates_path),
        "survivors_path": str(survivors_path),
        "survivors_sha256": _sha256_file(survivors_path),
        "top_candidates": top_candidates,
        "survivors": survivors,
    }
    return _write_report(output_root, report)


def generate_specs(historical_trades: pd.DataFrame) -> list[DirectionalGateSpec]:
    specs = []
    available_features = set(historical_trades.columns)
    for (
        quote_volume_floor_quantile,
        realized_vol_floor_quantile,
        quote_volume_cap_quantile,
        realized_vol_cap_quantile,
        directional_feature,
    ) in itertools.product(
        DEFAULT_QUOTE_VOLUME_FLOOR_QUANTILES,
        DEFAULT_VOLATILITY_FLOOR_QUANTILES,
        DEFAULT_QUOTE_VOLUME_CAP_QUANTILES,
        DEFAULT_VOLATILITY_CAP_QUANTILES,
        DEFAULT_DIRECTIONAL_FEATURES,
    ):
        if (
            quote_volume_cap_quantile is not None
            and quote_volume_cap_quantile <= quote_volume_floor_quantile
        ):
            continue
        if (
            realized_vol_cap_quantile is not None
            and realized_vol_cap_quantile <= realized_vol_floor_quantile
        ):
            continue
        if directional_feature is None:
            specs.append(
                DirectionalGateSpec(
                    quote_volume_floor_quantile=quote_volume_floor_quantile,
                    realized_vol_floor_quantile=realized_vol_floor_quantile,
                    quote_volume_cap_quantile=quote_volume_cap_quantile,
                    realized_vol_cap_quantile=realized_vol_cap_quantile,
                    directional_feature=None,
                    directional_quantile=None,
                )
            )
            continue
        if directional_feature not in available_features:
            continue
        for directional_quantile in DEFAULT_DIRECTIONAL_QUANTILES:
            specs.append(
                DirectionalGateSpec(
                    quote_volume_floor_quantile=quote_volume_floor_quantile,
                    realized_vol_floor_quantile=realized_vol_floor_quantile,
                    quote_volume_cap_quantile=quote_volume_cap_quantile,
                    realized_vol_cap_quantile=realized_vol_cap_quantile,
                    directional_feature=directional_feature,
                    directional_quantile=directional_quantile,
                )
            )
    return specs


def evaluate_directional_spec(
    historical_trades: pd.DataFrame,
    extension_trades: pd.DataFrame,
    *,
    spec: DirectionalGateSpec,
    minimum_block_trades: int,
    extension_block_names: tuple[str, ...],
) -> dict[str, Any]:
    historical_blocks = []
    for holdout_block in sorted(str(value) for value in historical_trades["block"].unique()):
        training = historical_trades[historical_trades["block"] != holdout_block]
        holdout = historical_trades[historical_trades["block"] == holdout_block]
        thresholds = thresholds_from_training(training, spec)
        summary = one_position_summary(apply_directional_gate(holdout, thresholds))
        historical_blocks.append(
            {
                "block": holdout_block,
                "minimum_trade_sample": int(summary["trade_count"]) >= minimum_block_trades,
                "passed_stress": (
                    int(summary["trade_count"]) >= minimum_block_trades
                    and float(summary["stress_total_log_return_bps"]) > 0
                    and float(summary["stress_mean_bps"]) > 0
                ),
                **summary,
            }
        )
    historical_aggregate = aggregate_block_summaries(
        historical_blocks,
        minimum_block_trades=minimum_block_trades,
    )
    frozen_thresholds = thresholds_from_training(historical_trades, spec)
    extension_blocks = summarize_extension_blocks(
        extension_trades,
        thresholds=frozen_thresholds,
        block_names=extension_block_names,
        minimum_block_trades=minimum_block_trades,
    )
    extension_aggregate = aggregate_block_summaries(
        extension_blocks,
        minimum_block_trades=minimum_block_trades,
    )
    return {
        "spec": asdict(spec) | {"name": spec.name},
        "frozen_thresholds": asdict(frozen_thresholds),
        "historical_aggregate": historical_aggregate,
        "extension_aggregate": extension_aggregate,
        "historical_blocks": historical_blocks,
        "extension_blocks": extension_blocks,
    }


def thresholds_from_training(
    training: pd.DataFrame,
    spec: DirectionalGateSpec,
) -> DirectionalGateThresholds:
    quote_volume_cap = (
        None
        if spec.quote_volume_cap_quantile is None
        else float(training["quote_volume_30s"].quantile(spec.quote_volume_cap_quantile))
    )
    realized_vol_cap = (
        None
        if spec.realized_vol_cap_quantile is None
        else float(training["realized_vol_30s_bps"].quantile(spec.realized_vol_cap_quantile))
    )
    directional_floor = (
        None
        if spec.directional_feature is None
        else float(training[spec.directional_feature].quantile(spec.directional_quantile))
    )
    return DirectionalGateThresholds(
        spec_name=spec.name,
        quote_volume_floor=float(
            training["quote_volume_30s"].quantile(spec.quote_volume_floor_quantile)
        ),
        realized_vol_floor=float(
            training["realized_vol_30s_bps"].quantile(spec.realized_vol_floor_quantile)
        ),
        quote_volume_cap=quote_volume_cap,
        realized_vol_cap=realized_vol_cap,
        directional_feature=spec.directional_feature,
        directional_floor=directional_floor,
    )


def apply_directional_gate(
    frame: pd.DataFrame,
    thresholds: DirectionalGateThresholds,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = (
        (frame["quote_volume_30s"] >= thresholds.quote_volume_floor)
        & (frame["realized_vol_30s_bps"] >= thresholds.realized_vol_floor)
    )
    if thresholds.quote_volume_cap is not None:
        mask &= frame["quote_volume_30s"] <= thresholds.quote_volume_cap
    if thresholds.realized_vol_cap is not None:
        mask &= frame["realized_vol_30s_bps"] <= thresholds.realized_vol_cap
    if thresholds.directional_feature is not None:
        mask &= frame[thresholds.directional_feature] >= float(thresholds.directional_floor)
    return frame[mask].copy()


def summarize_extension_blocks(
    extension_trades: pd.DataFrame,
    *,
    thresholds: DirectionalGateThresholds,
    block_names: tuple[str, ...],
    minimum_block_trades: int,
) -> list[dict[str, Any]]:
    filtered = apply_directional_gate(extension_trades, thresholds)
    grouped = {
        str(block): block_frame
        for block, block_frame in filtered.groupby("block", sort=True)
    } if not filtered.empty else {}
    blocks = []
    for block_name in block_names:
        summary = one_position_summary(grouped.get(block_name, pd.DataFrame()))
        blocks.append(
            {
                "block": block_name,
                "minimum_trade_sample": int(summary["trade_count"]) >= minimum_block_trades,
                "passed_stress": (
                    int(summary["trade_count"]) >= minimum_block_trades
                    and float(summary["stress_total_log_return_bps"]) > 0
                    and float(summary["stress_mean_bps"]) > 0
                ),
                **summary,
            }
        )
    return blocks


def is_survivor(
    candidate: dict[str, Any],
    *,
    minimum_block_trades: int,
    minimum_survivor_blocks: int,
) -> bool:
    historical = candidate["historical_aggregate"]
    extension = candidate["extension_aggregate"]
    return (
        int(historical["minimum_trade_block_count"]) >= minimum_survivor_blocks
        and not historical["negative_sufficient_blocks"]
        and float(historical["stress_total_log_return_bps"]) > 0
        and int(extension["minimum_trade_block_count"]) > 0
        and not extension["negative_sufficient_blocks"]
        and int(extension["trade_count"]) >= minimum_block_trades
        and float(extension["stress_total_log_return_bps"]) > 0
    )


def fast68_status(
    *,
    survivor_count: int,
    extension_available: bool,
    minimum_survivor_blocks: int,
) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "diagnostic_after_fast67_failure",
        "no_order_execution",
        f"minimum_survivor_blocks:{minimum_survivor_blocks}",
    ]
    if not extension_available:
        return "FAST68_DIRECTIONAL_GATE_PENDING_EXTENSION", reasons + [
            "no_extension_block_available"
        ]
    if survivor_count <= 0:
        return "FAST68_DIRECTIONAL_GATE_NO_SURVIVORS", reasons + [
            "no_candidate_passed_historical_and_extension_filters"
        ]
    return "FAST68_DIRECTIONAL_GATE_DIAGNOSTIC_SURVIVORS_FOUND_REQUIRES_SHADOW", reasons + [
        f"survivor_count:{survivor_count}",
        "requires_clean_future_prospective_shadow_before_testnet",
    ]


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[bool, bool, int, int, float, float, int]:
    historical = candidate["historical_aggregate"]
    extension = candidate["extension_aggregate"]
    return (
        not historical["negative_sufficient_blocks"],
        not extension["negative_sufficient_blocks"],
        int(historical["minimum_trade_block_count"]),
        int(historical["passed_stress_count"]),
        float(historical["stress_total_log_return_bps"]),
        float(extension["stress_total_log_return_bps"]),
        int(extension["trade_count"]),
    )


def _base_report(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    historical_fast_start: int,
    historical_fast_end: int,
    minimum_block_trades: int,
    minimum_survivor_blocks: int,
    extension_report_paths: tuple[Path, ...],
) -> dict[str, Any]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast68_is_diagnostic_search_after_fast67_extension_failure",
            "survivors_are_preregistration_candidates_only",
            "extension_block_participated_in_family_discovery",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast68_directional_gate_diagnostic",
        "data_root": str(data_root),
        "fast65_merged_path": str(fast65_merged_path),
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "minimum_block_trades": minimum_block_trades,
        "minimum_survivor_blocks": minimum_survivor_blocks,
        "extension_report_paths": [str(path) for path in extension_report_paths],
    }


def _validate_parameters(
    *,
    historical_fast_start: int,
    historical_fast_end: int,
    minimum_block_trades: int,
    minimum_survivor_blocks: int,
    top_n: int,
) -> None:
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")
    if minimum_block_trades < 1:
        raise ValueError("minimum_block_trades must be positive")
    if minimum_survivor_blocks < 1:
        raise ValueError("minimum_survivor_blocks must be positive")
    if top_n < 1:
        raise ValueError("top_n must be positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate_count": report.get("candidate_count"),
        "survivor_count": report.get("survivor_count"),
        "top_candidate": report.get("top_candidate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
