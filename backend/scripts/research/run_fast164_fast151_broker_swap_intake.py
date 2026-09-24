"""Fast164: broker-swap intake pack for the Fast151 operating candidate.

Fast163 says the candidate is blocked by a broker-swap table. Fast159 already
knows how to score a signed pips-per-rollover-day table, but a broker UI often
shows swaps in points or another human-facing format. Fast164 creates the intake
package that bridges that gap:

* a fillable raw intake template;
* pair-level tolerance limits from Fast160;
* a normalized Fast159-compatible CSV when a raw table is supplied;
* an optional offline Fast159 broker-swap gate report.

No broker is contacted. Historical/research coordination only. No orders, no
ledger, no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)
from scripts.research.run_fast159_fast151_exact_swap_rollover_audit import (  # noqa: E402
    run as run_fast159,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast164" / "fast151-broker-swap-intake"
DEFAULT_FAST160_REPORT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast160"
    / "fast151-pair-swap-sensitivity-audit"
    / "report.json"
)
TARGET_PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "NZDUSD")
RAW_REQUIRED_COLUMNS = ("pair", "long_swap_value", "short_swap_value", "value_unit")
NORMALIZED_COLUMNS = ("pair", "long_pips_per_rollover_day", "short_pips_per_rollover_day")
SUPPORTED_UNITS = ("pips_per_rollover_day", "points_per_rollover_day")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-swap-path", type=Path, default=None)
    parser.add_argument("--points-per-pip", type=float, default=10.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fast160-report-path", type=Path, default=DEFAULT_FAST160_REPORT_PATH)
    parser.add_argument(
        "--skip-fast159",
        action="store_true",
        help="normalize only; do not run the Fast159 broker-swap gate even when raw swap is supplied",
    )
    arguments = parser.parse_args(argv)
    report = run(
        raw_swap_path=arguments.raw_swap_path,
        points_per_pip=arguments.points_per_pip,
        output_root=arguments.output_root,
        fast160_report_path=arguments.fast160_report_path,
        run_broker_gate=not arguments.skip_fast159,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    raw_swap_path: Path | None = None,
    points_per_pip: float = 10.0,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fast160_report_path: Path = DEFAULT_FAST160_REPORT_PATH,
    run_broker_gate: bool = True,
) -> dict[str, Any]:
    if points_per_pip <= 0:
        raise ValueError("points_per_pip must be positive")
    pair_limits = load_pair_limits(fast160_report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_template_path = output_root / "broker-swap-raw-intake-template.csv"
    pair_limits_path = output_root / "broker-swap-pair-limits.csv"
    normalized_path = output_root / "fast159-broker-swap-input.csv"
    summary_path = output_root / "broker-swap-intake-summary.json"

    build_raw_template(pair_limits).to_csv(raw_template_path, index=False)
    pd.DataFrame(pair_limits).to_csv(pair_limits_path, index=False)

    normalized_rows: list[dict[str, Any]] = []
    broker_gate_report: dict[str, Any] | None = None
    if raw_swap_path is not None:
        normalized = normalize_raw_swap_table(raw_swap_path, points_per_pip=points_per_pip)
        normalized.to_csv(normalized_path, index=False)
        normalized_rows = normalized.to_dict("records")
        if run_broker_gate:
            broker_gate_report = run_fast159(
                broker_swap_path=normalized_path,
                output_root=output_root / "fast159-broker-swap-gate",
            )
    else:
        pd.DataFrame(columns=list(NORMALIZED_COLUMNS)).to_csv(normalized_path, index=False)

    status, status_reasons = fast164_status(
        raw_swap_supplied=raw_swap_path is not None,
        broker_gate_report=broker_gate_report,
    )
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "raw_swap_path": None if raw_swap_path is None else str(raw_swap_path),
        "raw_swap_sha256": None if raw_swap_path is None else sha256_file(raw_swap_path),
        "points_per_pip": points_per_pip,
        "supported_units": list(SUPPORTED_UNITS),
        "sign_convention": "positive values are credits; negative values are debits; strategy uses SHORT side",
        "pair_limits": pair_limits,
        "normalized_rows": normalized_rows,
        "broker_gate_status": None if broker_gate_report is None else broker_gate_report["status"],
        "broker_gate_report_path": None if broker_gate_report is None else broker_gate_report["report_path"],
        "broker_gate_report_sha256": None if broker_gate_report is None else broker_gate_report["report_sha256"],
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast164_is_broker_swap_input_pack_not_strategy_approval",
            "no_broker_contact",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast164_fast151_broker_swap_intake",
        "status": status,
        "status_reasons": status_reasons,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "summary": summary,
        "artifacts": {
            "raw_intake_template": {"path": str(raw_template_path), "sha256": sha256_file(raw_template_path)},
            "pair_limits": {"path": str(pair_limits_path), "sha256": sha256_file(pair_limits_path)},
            "normalized_fast159_input": {"path": str(normalized_path), "sha256": sha256_file(normalized_path)},
            "broker_swap_intake_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def load_pair_limits(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Fast160 report not found: {path}")
    report = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for evaluation in report["summary"]["pair_evaluations"]:
        pair = evaluation["pair"]
        rows.append(
            {
                "pair": pair,
                "current_pair_only_break_even_pips_per_rollover_day": evaluation["current"][
                    "pair_only_break_even_lcb_zero_pips_per_rollover_day"
                ],
                "pre_pair_only_break_even_pips_per_rollover_day": evaluation["pre"][
                    "pair_only_break_even_lcb_zero_pips_per_rollover_day"
                ],
                "current_pair_total_rollover_days": evaluation["current"]["pair_total_rollover_days"],
                "current_pair_baseline_mean_stress_pips": evaluation["current"]["pair_baseline_mean_stress_pips"],
            }
        )
    return rows


def build_raw_template(pair_limits: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in pair_limits:
        rows.append(
            {
                "pair": row["pair"],
                "broker_symbol": row["pair"],
                "long_swap_value": "",
                "short_swap_value": "",
                "value_unit": "pips_per_rollover_day",
                "points_per_pip": 10.0,
                "notes": "positive=credit, negative=debit; fill from broker contract/spec/symbol info",
                "current_pair_only_break_even_pips_per_rollover_day": row[
                    "current_pair_only_break_even_pips_per_rollover_day"
                ],
            }
        )
    return pd.DataFrame(rows)


def normalize_raw_swap_table(path: Path, *, points_per_pip: float) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if all(column in raw.columns for column in NORMALIZED_COLUMNS):
        normalized = raw[list(NORMALIZED_COLUMNS)].copy()
    else:
        missing = [column for column in RAW_REQUIRED_COLUMNS if column not in raw.columns]
        if missing:
            raise ValueError(f"raw swap table missing columns: {missing}")
        normalized = raw_to_normalized(raw, default_points_per_pip=points_per_pip)
    normalized["pair"] = normalized["pair"].astype("string").str.upper()
    normalized = normalized.loc[normalized["pair"].isin(TARGET_PAIRS)].copy()
    missing_pairs = sorted(set(TARGET_PAIRS) - set(normalized["pair"].astype(str).tolist()))
    if missing_pairs:
        raise ValueError(f"raw swap table missing target pairs: {missing_pairs}")
    if normalized["pair"].duplicated().any():
        duplicates = sorted(normalized.loc[normalized["pair"].duplicated(), "pair"].astype(str).unique().tolist())
        raise ValueError(f"raw swap table has duplicate pairs: {duplicates}")
    for column in ("long_pips_per_rollover_day", "short_pips_per_rollover_day"):
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("float64")
    return normalized.sort_values("pair", ignore_index=True)


def raw_to_normalized(raw: pd.DataFrame, *, default_points_per_pip: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in raw.to_dict("records"):
        unit = str(row["value_unit"]).strip()
        if unit not in SUPPORTED_UNITS:
            raise ValueError(f"unsupported swap unit {unit!r}; expected one of {SUPPORTED_UNITS}")
        row_points_per_pip = row.get("points_per_pip")
        points = (
            default_points_per_pip
            if row_points_per_pip is None or pd.isna(row_points_per_pip) or row_points_per_pip == ""
            else float(row_points_per_pip)
        )
        if points <= 0:
            raise ValueError("points_per_pip must be positive")
        long_value = float(row["long_swap_value"])
        short_value = float(row["short_swap_value"])
        if unit == "points_per_rollover_day":
            long_value /= points
            short_value /= points
        rows.append(
            {
                "pair": str(row["pair"]).upper(),
                "long_pips_per_rollover_day": long_value,
                "short_pips_per_rollover_day": short_value,
            }
        )
    return pd.DataFrame(rows)


def fast164_status(
    *,
    raw_swap_supplied: bool,
    broker_gate_report: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "broker_swap_intake"]
    if not raw_swap_supplied:
        return "FAST164_BROKER_SWAP_INPUT_PENDING", reasons + ["raw_broker_swap_table_not_supplied"]
    if broker_gate_report is None:
        return "FAST164_BROKER_SWAP_NORMALIZED_NOT_GATED", reasons + ["fast159_gate_skipped"]
    if broker_gate_report["status"] == "FAST159_FAST151_BROKER_SWAP_TABLE_PASSES_REQUIRES_PROSPECTIVE_OBSERVATION":
        return (
            "FAST164_BROKER_SWAP_INPUT_PASSES_FAST159_GATE",
            reasons + ["broker_swap_lcb_gate_passed", "fresh_preregistered_prospective_observation_still_required"],
        )
    return (
        "FAST164_BROKER_SWAP_INPUT_FAILS_FAST159_GATE",
        reasons + ["broker_swap_lcb_gate_failed", broker_gate_report["status"]],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "raw_swap_path": report["summary"]["raw_swap_path"],
        "broker_gate_status": report["summary"]["broker_gate_status"],
        "raw_intake_template": report["artifacts"]["raw_intake_template"]["path"],
        "normalized_fast159_input": report["artifacts"]["normalized_fast159_input"]["path"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
