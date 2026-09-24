"""Fast175: train-selected regime gate audit for Fast174's lead-lag clue.

Fast174 found one cross-asset lead-lag spec that passed train and validation:

    target=ETHUSDT, horizon=900s, top_fraction=0.02

but it failed the extension block. Fast175 asks whether a simple pre-entry
regime gate, selected without looking at validation/extension outcomes, can
stabilize that clue.

The audit also reports post-hoc strict survivors as diagnostics only. Those are
not approvals because the family and thresholds are being inspected after the
historical extension is known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import (  # noqa: E402
    DECISION_STRIDE_MINUTES,
    DEFAULT_BTC_TRADE_ROOT,
    DEFAULT_ETH_TRADE_ROOT,
    DEFAULT_SOL_TRADE_ROOT,
    DEFAULT_OUTPUT_ROOT as FAST174_OUTPUT_ROOT,
    EXPLICIT_COST_BPS,
    EXTENSION_BLOCK,
    TRAIN_BLOCKS,
    VALIDATION_BLOCK,
    build_all_feature_rows,
    leadlag_model,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast175-fast174-regime-gate-audit"
)
TARGET_SYMBOL = "ETHUSDT"
HORIZON_SECONDS = 900
TOP_FRACTION = 0.02
GATE_QUANTILES = (0.2, 0.33, 0.5, 0.67, 0.8)
MIN_TRAIN_TRADES = 8
MIN_HOLDOUT_TRADES = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    report = run_fast175_fast174_regime_gate_audit(
        trade_roots={
            "BTCUSDT": arguments.btc_trade_root,
            "ETHUSDT": arguments.eth_trade_root,
            "SOLUSDT": arguments.sol_trade_root,
        },
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast175_fast174_regime_gate_audit(
    *,
    trade_roots: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = build_all_feature_rows(trade_roots)
    feature_columns = [
        column
        for column in rows.columns
        if any(token in column for token in ("_ret_", "_vol_", "_flow_", "_qv_"))
    ] + ["hour_sin", "hour_cos"]
    samples = build_fast174_validation_survivor_samples(rows, feature_columns)
    base_summary = {name: summarize_selected(sample) for name, sample in samples.items()}
    gate_evaluations = evaluate_gates(samples, feature_columns)
    preselected_gate = select_train_preselected_gate(gate_evaluations)
    validation_survivors = [
        gate
        for gate in gate_evaluations
        if gate["train"]["passed"] and gate["validation"]["passed"]
    ]
    strict_survivors = [
        gate
        for gate in validation_survivors
        if gate["extension"]["passed"]
    ]
    all_train_block_strict_survivors = [
        gate for gate in strict_survivors if gate["train"]["all_train_blocks_represented"]
    ]
    status, status_reasons = fast175_status(
        preselected_gate=preselected_gate,
        strict_survivor_count=len(strict_survivors),
    )

    gates_path = output_root / "gate-evaluations.json"
    write_json(gates_path, gate_evaluations)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "source_fast174_report_path": str(FAST174_OUTPUT_ROOT / "report.json"),
        "source_fast174_report_sha256": sha256_file(FAST174_OUTPUT_ROOT / "report.json")
        if (FAST174_OUTPUT_ROOT / "report.json").exists()
        else None,
        "target_symbol": TARGET_SYMBOL,
        "horizon_seconds": HORIZON_SECONDS,
        "top_fraction": TOP_FRACTION,
        "gate_quantiles": list(GATE_QUANTILES),
        "min_train_trades": MIN_TRAIN_TRADES,
        "min_holdout_trades": MIN_HOLDOUT_TRADES,
        "base_summary": base_summary,
        "gate_count": len(gate_evaluations),
        "train_gate_count": sum(1 for gate in gate_evaluations if gate["train"]["passed"]),
        "all_train_block_gate_count": sum(
            1 for gate in gate_evaluations if gate["train"]["all_train_blocks_represented"]
        ),
        "validation_survivor_count": len(validation_survivors),
        "strict_survivor_count": len(strict_survivors),
        "all_train_block_strict_survivor_count": len(all_train_block_strict_survivors),
        "preselected_gate": compact_gate(preselected_gate),
        "top_diagnostic_strict_survivors": [
            compact_gate(gate)
            for gate in sorted(
                strict_survivors,
                key=lambda item: (
                    item["extension"]["net_total_bps"],
                    item["validation"]["net_total_bps"],
                    item["train"]["net_total_bps"],
                ),
                reverse=True,
            )[:10]
        ],
        "top_validation_survivors": [
            compact_gate(gate)
            for gate in sorted(
                validation_survivors,
                key=lambda item: (
                    item["validation"]["net_total_bps"],
                    item["train"]["net_total_bps"],
                    item["extension"]["net_total_bps"],
                ),
                reverse=True,
            )[:10]
        ],
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)

    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast175_is_regime_gate_audit_not_strategy_approval",
            "preselected_gate_must_be_chosen_without_validation_or_extension",
            "posthoc_strict_survivors_are_diagnostic_only",
            "fresh_preregistered_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast175_fast174_regime_gate_audit",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "gate_evaluations": {"path": str(gates_path), "sha256": sha256_file(gates_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def build_fast174_validation_survivor_samples(
    rows: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    shift_rows = int(HORIZON_SECONDS / 60 / DECISION_STRIDE_MINUTES)
    spec_rows = rows.copy()
    spec_rows["future_price"] = spec_rows.groupby("date")[f"{TARGET_SYMBOL}_price"].shift(-shift_rows)
    spec_rows["actual_bps"] = (
        spec_rows["future_price"] / spec_rows[f"{TARGET_SYMBOL}_price"] - 1.0
    ) * 10000.0
    spec_rows = spec_rows.dropna(subset=["actual_bps"])
    train_rows = spec_rows.loc[spec_rows["block"].isin(TRAIN_BLOCKS)]
    model = leadlag_model()
    model.fit(train_rows[feature_columns], train_rows["actual_bps"])
    return {
        "train": select_top_predictions(model, train_rows, feature_columns),
        "validation": select_top_predictions(
            model,
            spec_rows.loc[spec_rows["block"].eq(VALIDATION_BLOCK)],
            feature_columns,
        ),
        "extension": select_top_predictions(
            model,
            spec_rows.loc[spec_rows["block"].eq(EXTENSION_BLOCK)],
            feature_columns,
        ),
    }


def select_top_predictions(
    model: Any,
    sample: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    predictions = model.predict(sample[feature_columns])
    ranked = sample.assign(prediction=predictions, edge=np.abs(predictions), side=np.sign(predictions))
    ranked = ranked.loc[ranked["side"].ne(0)]
    selected = ranked.nlargest(max(1, int(len(ranked) * TOP_FRACTION)), "edge").copy()
    selected["net_bps"] = selected["actual_bps"] * selected["side"] - EXPLICIT_COST_BPS
    return selected


def evaluate_gates(samples: dict[str, pd.DataFrame], feature_columns: list[str]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for feature in [*feature_columns, "prediction", "edge"]:
        train_values = clean_numeric(samples["train"][feature])
        if train_values.nunique() < 3:
            continue
        for quantile in GATE_QUANTILES:
            threshold = float(train_values.quantile(quantile))
            for operator in ("lte", "gte"):
                gate = evaluate_gate(samples, feature=feature, operator=operator, threshold=threshold, quantile=quantile)
                if gate["train"]["passed"]:
                    gates.append(gate)
    return gates


def evaluate_gate(
    samples: dict[str, pd.DataFrame],
    *,
    feature: str,
    operator: str,
    threshold: float,
    quantile: float,
) -> dict[str, Any]:
    result = {
        "feature": feature,
        "operator": operator,
        "threshold": threshold,
        "quantile": quantile,
    }
    for sample_name, sample in samples.items():
        selected = sample.loc[gate_mask(sample, feature=feature, operator=operator, threshold=threshold)]
        result[sample_name] = summarize_selected(selected, sample_name=sample_name)
    result["strict_passed"] = (
        result["train"]["passed"]
        and result["validation"]["passed"]
        and result["extension"]["passed"]
    )
    return result


def gate_mask(sample: pd.DataFrame, *, feature: str, operator: str, threshold: float) -> pd.Series:
    values = clean_numeric(sample[feature])
    if operator == "lte":
        return values <= threshold
    if operator == "gte":
        return values >= threshold
    raise ValueError(f"unsupported operator {operator}")


def summarize_selected(selected: pd.DataFrame, *, sample_name: str = "train") -> dict[str, Any]:
    net = selected["net_bps"]
    block_summaries = []
    for block, block_rows in selected.groupby("block", sort=True):
        block_net = block_rows["net_bps"]
        block_summaries.append(
            {
                "block": block,
                "trade_count": int(len(block_rows)),
                "net_total_bps": float(block_net.sum()),
                "net_mean_bps": float(block_net.mean()) if len(block_net) else 0.0,
                "win_rate": float((block_net > 0).mean()) if len(block_net) else 0.0,
            }
        )
    minimum_trades = MIN_TRAIN_TRADES if sample_name == "train" else MIN_HOLDOUT_TRADES
    represented_train_blocks = {
        summary["block"] for summary in block_summaries if summary["block"] in TRAIN_BLOCKS
    }
    return {
        "passed": bool(len(net) >= minimum_trades and net.sum() > 0),
        "trade_count": int(len(selected)),
        "net_total_bps": float(net.sum()) if len(net) else 0.0,
        "net_mean_bps": float(net.mean()) if len(net) else 0.0,
        "win_rate": float((net > 0).mean()) if len(net) else 0.0,
        "blocks": block_summaries,
        "train_block_count": len(represented_train_blocks),
        "all_train_blocks_represented": len(represented_train_blocks) == len(TRAIN_BLOCKS),
        "min_train_block_net_bps": min(
            (
                summary["net_total_bps"]
                for summary in block_summaries
                if summary["block"] in TRAIN_BLOCKS
            ),
            default=0.0,
        ),
    }


def select_train_preselected_gate(gates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        gate
        for gate in gates
        if gate["train"]["passed"] and gate["train"]["all_train_blocks_represented"]
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda gate: (
            gate["train"]["min_train_block_net_bps"],
            gate["train"]["net_total_bps"],
            gate["train"]["trade_count"],
            gate["feature"],
            gate["operator"],
            gate["quantile"],
        ),
        reverse=True,
    )[0]


def fast175_status(
    *,
    preselected_gate: dict[str, Any] | None,
    strict_survivor_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "regime_gate_audit"]
    if preselected_gate is not None and preselected_gate["strict_passed"]:
        return (
            "FAST175_PRESELECTED_REGIME_GATE_REQUIRES_PREREGISTRATION",
            reasons + ["preselected_gate_passed_validation_and_extension", "fresh_observation_required"],
        )
    if strict_survivor_count:
        return (
            "FAST175_POSTHOC_REGIME_GATES_FOUND_BUT_PRESELECTED_GATE_FAILED",
            reasons
            + [
                f"posthoc_strict_survivors:{strict_survivor_count}",
                "preselected_gate_failed_validation_or_extension",
                "posthoc_survivors_are_diagnostic_only",
            ],
        )
    if preselected_gate is not None and preselected_gate["validation"]["passed"]:
        return (
            "FAST175_PRESELECTED_REGIME_GATE_FAILED_EXTENSION",
            reasons + ["preselected_gate_passed_validation", "preselected_gate_failed_extension"],
        )
    return (
        "FAST175_REGIME_GATE_FAILED_VALIDATION",
        reasons + ["no_preselected_gate_survived_validation"],
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def compact_gate(gate: dict[str, Any] | None) -> dict[str, Any] | None:
    if gate is None:
        return None
    return {
        "feature": gate["feature"],
        "operator": gate["operator"],
        "threshold": gate["threshold"],
        "quantile": gate["quantile"],
        "strict_passed": gate["strict_passed"],
        "train": gate["train"],
        "validation": gate["validation"],
        "extension": gate["extension"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "research_only": report["research_only"],
        "approved_strategy": report["approved_strategy"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "gate_count": summary["gate_count"],
        "validation_survivor_count": summary["validation_survivor_count"],
        "strict_survivor_count": summary["strict_survivor_count"],
        "all_train_block_strict_survivor_count": summary["all_train_block_strict_survivor_count"],
        "preselected_gate": summary["preselected_gate"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
