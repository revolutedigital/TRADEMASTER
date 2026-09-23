"""Fast79 staged rare pair-setup audit for frozen top-p trades.

Fast73-Fast74 showed that the current top-p stream often predicts "touch before
stop" but not trailing payoff. Fast79 asks a narrower diagnostic question:
among simple causal pre-trade conditions, can a rare two-condition setup keep
all historical sufficient blocks positive and survive the later Fast56
extension?

This is a staged scout, not an exhaustive proof and not a strategy. Candidate
pairs are selected from the best single-condition historical gates, then tested
on Fast56 without tuning on that extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast65_causal_feature_gate_walkforward import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    causal_feature_columns,
    load_selected_trades_with_causal_features,
    one_position_summary,
)


DEFAULT_HISTORICAL_MERGED_PATH = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "fast65-causal-feature-gate-walkforward"
    / "selected_trades_with_causal_features.parquet"
)
DEFAULT_FAST56_REPORT = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms"
    / "report.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast79-rare-pair-setup-audit"
DEFAULT_QUANTILES = (0.20, 0.33, 0.50, 0.67, 0.80)
DEFAULT_ATOM_SEED_COUNT = 60
DEFAULT_EXTENSION_TOP_COUNT = 25


@dataclass(frozen=True)
class Condition:
    feature: str
    direction: str
    threshold: float
    quantile: float

    @property
    def name(self) -> str:
        operator = ">=" if self.direction == "high" else "<="
        return f"{self.feature}{operator}q{self.quantile:.2f}:{self.threshold:.10g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--historical-merged-path", type=Path, default=DEFAULT_HISTORICAL_MERGED_PATH)
    parser.add_argument("--extension-report", type=Path, default=DEFAULT_FAST56_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-total-trades", type=int, default=20)
    parser.add_argument("--minimum-block-trades", type=int, default=3)
    parser.add_argument("--minimum-sufficient-blocks", type=int, default=5)
    parser.add_argument("--atom-seed-count", type=int, default=DEFAULT_ATOM_SEED_COUNT)
    parser.add_argument("--extension-top-count", type=int, default=DEFAULT_EXTENSION_TOP_COUNT)
    arguments = parser.parse_args()
    report = run_fast79_rare_pair_setup_audit(
        data_root=arguments.data_root,
        historical_merged_path=arguments.historical_merged_path,
        extension_report=arguments.extension_report,
        output_root=arguments.output_root,
        minimum_total_trades=arguments.minimum_total_trades,
        minimum_block_trades=arguments.minimum_block_trades,
        minimum_sufficient_blocks=arguments.minimum_sufficient_blocks,
        atom_seed_count=arguments.atom_seed_count,
        extension_top_count=arguments.extension_top_count,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast79_rare_pair_setup_audit(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    historical_merged_path: Path = DEFAULT_HISTORICAL_MERGED_PATH,
    extension_report: Path = DEFAULT_FAST56_REPORT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    minimum_total_trades: int = 20,
    minimum_block_trades: int = 3,
    minimum_sufficient_blocks: int = 5,
    atom_seed_count: int = DEFAULT_ATOM_SEED_COUNT,
    extension_top_count: int = DEFAULT_EXTENSION_TOP_COUNT,
) -> dict[str, Any]:
    validate_parameters(
        minimum_total_trades=minimum_total_trades,
        minimum_block_trades=minimum_block_trades,
        minimum_sufficient_blocks=minimum_sufficient_blocks,
        atom_seed_count=atom_seed_count,
        extension_top_count=extension_top_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    historical = load_historical(historical_merged_path)
    extension = load_selected_trades_with_causal_features((extension_report,), data_root=data_root)
    feature_columns = causal_feature_columns(historical)
    atoms = condition_candidates(historical, feature_columns=feature_columns)
    atom_evaluations = [
        evaluate_candidate(
            historical,
            conditions=(condition,),
            minimum_block_trades=minimum_block_trades,
        )
        for condition in atoms
    ]
    eligible_atoms = [
        item
        for item in atom_evaluations
        if item["total"]["trade_count"] >= minimum_total_trades
        and item["sufficient_block_count"] >= minimum_sufficient_blocks
    ]
    eligible_atoms.sort(key=historical_sort_key)
    seed_conditions = [
        condition_from_dict(item["conditions"][0])
        for item in eligible_atoms[:atom_seed_count]
    ]

    pair_evaluations = [
        evaluate_candidate(
            historical,
            conditions=pair,
            minimum_block_trades=minimum_block_trades,
        )
        for pair in combinations(seed_conditions, 2)
    ]
    eligible_pairs = [
        item
        for item in pair_evaluations
        if item["total"]["trade_count"] >= minimum_total_trades
        and item["sufficient_block_count"] >= minimum_sufficient_blocks
    ]
    eligible_pairs.sort(key=historical_sort_key)
    historical_survivors = [
        item for item in eligible_pairs if item["negative_sufficient_block_count"] == 0
    ]
    extension_candidates = historical_survivors[:extension_top_count]
    extension_audits = [
        {
            "historical": candidate,
            "extension": evaluate_candidate(
                extension,
                conditions=tuple(condition_from_dict(item) for item in candidate["conditions"]),
                minimum_block_trades=minimum_block_trades,
            ),
        }
        for candidate in extension_candidates
    ]
    extension_survivors = [
        item
        for item in extension_audits
        if item["extension"]["total"]["trade_count"] >= minimum_block_trades
        and item["extension"]["total"]["stress_total_log_return_bps"] > 0
        and item["extension"]["total"]["stress_mean_bps"] > 0
    ]
    extension_survivors.sort(key=extension_sort_key, reverse=True)
    status, status_reasons = fast79_status(
        historical_survivor_count=len(historical_survivors),
        extension_survivor_count=len(extension_survivors),
    )

    atom_path = output_root / "atom_evaluations.json"
    atom_path.write_text(
        json.dumps(eligible_atoms[:100], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pair_path = output_root / "pair_evaluations.json"
    pair_path.write_text(
        json.dumps(eligible_pairs[:250], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    extension_path = output_root / "extension_audits.json"
    extension_path.write_text(
        json.dumps(extension_audits, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast79_is_staged_diagnostic_scout",
            "candidate_pairs_selected_from_historical_results",
            "future_preregistered_shadow_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast79_rare_pair_setup_audit",
        "status": status,
        "status_reasons": status_reasons,
        "historical_merged_path": str(historical_merged_path),
        "historical_merged_sha256": _sha256_file(historical_merged_path),
        "extension_report": str(extension_report),
        "extension_report_sha256": _sha256_file(extension_report),
        "feature_count": len(feature_columns),
        "atom_candidate_count": len(atoms),
        "eligible_atom_count": len(eligible_atoms),
        "seed_atom_count": len(seed_conditions),
        "pair_candidate_count": len(pair_evaluations),
        "eligible_pair_count": len(eligible_pairs),
        "historical_survivor_count": len(historical_survivors),
        "extension_audited_count": len(extension_audits),
        "extension_survivor_count": len(extension_survivors),
        "minimum_total_trades": minimum_total_trades,
        "minimum_block_trades": minimum_block_trades,
        "minimum_sufficient_blocks": minimum_sufficient_blocks,
        "atom_evaluations_path": str(atom_path),
        "atom_evaluations_sha256": _sha256_file(atom_path),
        "pair_evaluations_path": str(pair_path),
        "pair_evaluations_sha256": _sha256_file(pair_path),
        "extension_audits_path": str(extension_path),
        "extension_audits_sha256": _sha256_file(extension_path),
        "top_historical_survivors": historical_survivors[:10],
        "top_extension_survivors": extension_survivors[:10],
    }
    return _write_report(output_root, report)


def validate_parameters(
    *,
    minimum_total_trades: int,
    minimum_block_trades: int,
    minimum_sufficient_blocks: int,
    atom_seed_count: int,
    extension_top_count: int,
) -> None:
    if minimum_total_trades < 1:
        raise ValueError("minimum_total_trades must be positive")
    if minimum_block_trades < 1:
        raise ValueError("minimum_block_trades must be positive")
    if minimum_sufficient_blocks < 1:
        raise ValueError("minimum_sufficient_blocks must be positive")
    if atom_seed_count < 2:
        raise ValueError("atom_seed_count must be at least 2")
    if extension_top_count < 1:
        raise ValueError("extension_top_count must be positive")


def load_historical(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError("historical merged selected trades is empty")
    return frame


def condition_candidates(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> list[Condition]:
    candidates = []
    for feature in feature_columns:
        values = clean_numeric(frame[feature]).dropna()
        if values.nunique() < 5:
            continue
        for quantile in quantiles:
            threshold = float(values.quantile(quantile))
            candidates.append(
                Condition(
                    feature=feature,
                    direction="high",
                    threshold=threshold,
                    quantile=quantile,
                )
            )
            candidates.append(
                Condition(
                    feature=feature,
                    direction="low",
                    threshold=threshold,
                    quantile=quantile,
                )
            )
    return candidates


def evaluate_candidate(
    frame: pd.DataFrame,
    *,
    conditions: tuple[Condition, ...],
    minimum_block_trades: int,
) -> dict[str, Any]:
    filtered = apply_conditions(frame, conditions)
    blocks = []
    for block in sorted(str(value) for value in frame["block"].unique()):
        block_summary = one_position_summary(filtered[filtered["block"].eq(block)])
        blocks.append({"block": block, **block_summary})
    sufficient_blocks = [block for block in blocks if block["trade_count"] >= minimum_block_trades]
    negative_sufficient_blocks = [
        block
        for block in sufficient_blocks
        if block["stress_total_log_return_bps"] <= 0 or block["stress_mean_bps"] <= 0
    ]
    total = one_position_summary(filtered)
    return {
        "name": " AND ".join(condition.name for condition in conditions),
        "conditions": [asdict(condition) | {"name": condition.name} for condition in conditions],
        "total": total,
        "sufficient_block_count": len(sufficient_blocks),
        "positive_sufficient_block_count": len(sufficient_blocks) - len(negative_sufficient_blocks),
        "negative_sufficient_block_count": len(negative_sufficient_blocks),
        "min_sufficient_block_stress_total_bps": min(
            (block["stress_total_log_return_bps"] for block in sufficient_blocks),
            default=0.0,
        ),
        "blocks": blocks,
    }


def apply_conditions(frame: pd.DataFrame, conditions: tuple[Condition, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        values = clean_numeric(frame[condition.feature])
        if condition.direction == "high":
            mask &= values.ge(condition.threshold).to_numpy()
        elif condition.direction == "low":
            mask &= values.le(condition.threshold).to_numpy()
        else:
            raise ValueError(f"unknown condition direction: {condition.direction}")
    return frame.loc[mask].copy()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def condition_from_dict(payload: dict[str, Any]) -> Condition:
    return Condition(
        feature=str(payload["feature"]),
        direction=str(payload["direction"]),
        threshold=float(payload["threshold"]),
        quantile=float(payload["quantile"]),
    )


def historical_sort_key(candidate: dict[str, Any]) -> tuple[int, int, float, float, float]:
    return (
        candidate["negative_sufficient_block_count"],
        -candidate["sufficient_block_count"],
        -candidate["min_sufficient_block_stress_total_bps"],
        -candidate["total"]["stress_total_log_return_bps"],
        -candidate["total"]["stress_mean_bps"],
    )


def extension_sort_key(item: dict[str, Any]) -> tuple[float, int, float, float]:
    extension_total = item["extension"]["total"]
    historical_total = item["historical"]["total"]
    return (
        extension_total["stress_total_log_return_bps"],
        extension_total["trade_count"],
        historical_total["stress_total_log_return_bps"],
        item["historical"]["min_sufficient_block_stress_total_bps"],
    )


def fast79_status(
    *,
    historical_survivor_count: int,
    extension_survivor_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if historical_survivor_count <= 0:
        return "FAST79_RARE_PAIR_SETUP_FAILED_HISTORICAL_SCREEN", reasons + [
            "no_historical_survivor",
        ]
    if extension_survivor_count <= 0:
        return "FAST79_RARE_PAIR_SETUP_FAILED_EXTENSION", reasons + [
            f"historical_survivors:{historical_survivor_count}",
            "no_extension_survivor",
        ]
    return "FAST79_RARE_PAIR_SETUP_WEAK_SURVIVORS_REQUIRE_PREREGISTRATION", reasons + [
        f"historical_survivors:{historical_survivor_count}",
        f"extension_survivors:{extension_survivor_count}",
        "staged_search_multiple_testing_not_strategy",
        "future_preregistered_shadow_required",
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
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
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "historical_survivor_count": report["historical_survivor_count"],
        "extension_survivor_count": report["extension_survivor_count"],
        "top_extension_survivors": report["top_extension_survivors"][:5],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
