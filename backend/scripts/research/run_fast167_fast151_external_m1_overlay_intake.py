"""Fast167: external M1 overlay intake for the Fast151 forward blocker.

Fast165/166 showed that the public sources tested do not yet provide M1 bars
after the Fast151 preregistration timestamp. Fast167 prepares the safe intake
path for a broker/platform/exported M1 file:

* create a fillable normalized CSV template;
* validate bid/ask M1 OHLC rows for the Fast151 target pairs;
* build an isolated overlay directory that keeps canonical M1 untouched;
* optionally run Fast152 on that overlay when every target pair has
  post-preregistration bars.

No canonical data is overwritten. No broker is contacted. No orders, no ledger,
no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    DEFAULT_M1_DIR,
    MAJOR_RATE_PAIRS,
    sha256_file,
)
from scripts.research.run_fast124_usdjpy_compression_observation import absolute_path  # noqa: E402
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    PREREGISTRATION_SHA256,
    PREREGISTRATION_TIMESTAMP,
    SAFETY_FLAGS,
    TARGET_PAIRS,
)
from scripts.research.run_fast152_fast151_readiness_monitor import (  # noqa: E402
    run_fast152_fast151_readiness_monitor,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast167" / "fast151-external-m1-overlay-intake"
)
EXTERNAL_M1_COLUMNS = (
    "pair",
    "timestamp",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)
PRICE_COLUMNS = EXTERNAL_M1_COLUMNS[2:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-m1-path", type=Path, default=None)
    parser.add_argument("--canonical-m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--skip-fast152",
        action="store_true",
        help="normalize/build overlay only; do not run Fast152 even if external post-prereg data is complete",
    )
    arguments = parser.parse_args(argv)
    report = run(
        external_m1_path=arguments.external_m1_path,
        canonical_m1_dir=arguments.canonical_m1_dir,
        output_root=arguments.output_root,
        run_readiness=not arguments.skip_fast152,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    external_m1_path: Path | None = None,
    canonical_m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_readiness: bool = True,
) -> dict[str, Any]:
    canonical_m1_dir = absolute_path(canonical_m1_dir)
    output_root = absolute_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    template_path = output_root / "external-m1-normalized-template.csv"
    normalized_dir = output_root / "normalized_external_m1"
    overlay_dir = output_root / "overlay_m1"
    summary_path = output_root / "external-m1-overlay-intake-summary.json"
    build_external_template().to_csv(template_path, index=False)

    normalized_summary: dict[str, Any] | None = None
    overlay_summary: dict[str, Any] | None = None
    readiness_report: dict[str, Any] | None = None
    if external_m1_path is not None:
        normalized = normalize_external_m1_csv(external_m1_path)
        normalized_summary = write_normalized_pair_files(normalized, normalized_dir)
        overlay_summary = build_overlay_dir(
            canonical_m1_dir=canonical_m1_dir,
            normalized_dir=normalized_dir,
            overlay_dir=overlay_dir,
        )
        if all_target_pairs_have_post_preregistration(normalized_summary) and run_readiness:
            readiness_report = run_fast152_fast151_readiness_monitor(
                m1_dir=overlay_dir,
                output_root=output_root / "fast152-overlay-readiness",
                fast151_output_root=output_root / "fast151-overlay-observation",
            )

    status, status_reasons = fast167_status(
        external_m1_supplied=external_m1_path is not None,
        normalized_summary=normalized_summary,
        readiness_report=readiness_report,
        run_readiness=run_readiness,
    )
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "external_m1_path": None if external_m1_path is None else str(absolute_path(external_m1_path)),
        "external_m1_sha256": None if external_m1_path is None else sha256_file(absolute_path(external_m1_path)),
        "canonical_m1_dir": str(canonical_m1_dir),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "normalized_dir": str(normalized_dir),
        "overlay_dir": str(overlay_dir),
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "target_pairs": list(TARGET_PAIRS),
        "required_rate_pairs_in_overlay": list(MAJOR_RATE_PAIRS),
        "normalized_summary": normalized_summary,
        "overlay_summary": overlay_summary,
        "fast152_overlay_status": None if readiness_report is None else readiness_report["status"],
        "fast152_overlay_report_path": None if readiness_report is None else readiness_report["report_path"],
        "fast152_overlay_report_sha256": None if readiness_report is None else readiness_report["report_sha256"],
        "next_action": next_action_for_status(status),
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast167_is_external_m1_intake_not_strategy_approval",
            "canonical_m1_not_overwritten",
            "overlay_m1_isolated_under_lab_output",
            "no_broker_contact",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast167_fast151_external_m1_overlay_intake",
        "status": status,
        "status_reasons": status_reasons,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "summary": summary,
        "artifacts": {
            "external_m1_template": {"path": str(template_path), "sha256": sha256_file(template_path)},
            "external_m1_overlay_intake_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def build_external_template() -> pd.DataFrame:
    rows = []
    example_timestamp = PREREGISTRATION_TIMESTAMP.tz_convert("UTC").ceil("1min").isoformat()
    for pair in TARGET_PAIRS:
        rows.append(
            {
                "pair": pair,
                "timestamp": example_timestamp,
                "bid_open": "",
                "bid_high": "",
                "bid_low": "",
                "bid_close": "",
                "ask_open": "",
                "ask_high": "",
                "ask_low": "",
                "ask_close": "",
                "notes": "UTC timestamp; bid/ask M1 OHLC from broker/platform export; ask must be >= bid",
            }
        )
    return pd.DataFrame(rows)


def normalize_external_m1_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = [column for column in EXTERNAL_M1_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"external M1 CSV missing columns: {missing}")
    normalized = raw[list(EXTERNAL_M1_COLUMNS)].copy()
    normalized["pair"] = normalized["pair"].astype("string").str.upper()
    normalized = normalized.loc[normalized["pair"].isin(TARGET_PAIRS)].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="raise")
    for column in PRICE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("float64")
    validate_external_prices(normalized)
    normalized = normalized.sort_values(["pair", "timestamp"])
    normalized = normalized.drop_duplicates(["pair", "timestamp"], keep="last")
    return normalized.reset_index(drop=True)


def validate_external_prices(rows: pd.DataFrame) -> None:
    crossed_checks = (
        ("ask_open", "bid_open"),
        ("ask_high", "bid_high"),
        ("ask_low", "bid_low"),
        ("ask_close", "bid_close"),
    )
    crossed = sum(int((rows[ask] < rows[bid]).sum()) for ask, bid in crossed_checks)
    if crossed:
        raise ValueError(f"external M1 CSV has {crossed} crossed bid/ask OHLC values")
    invalid_ohlc = 0
    for side in ("bid", "ask"):
        low = rows[f"{side}_low"]
        high = rows[f"{side}_high"]
        invalid_ohlc += int((low > rows[[f"{side}_open", f"{side}_close"]].min(axis=1)).sum())
        invalid_ohlc += int((high < rows[[f"{side}_open", f"{side}_close"]].max(axis=1)).sum())
    if invalid_ohlc:
        raise ValueError(f"external M1 CSV has {invalid_ohlc} invalid OHLC relationships")


def write_normalized_pair_files(rows: pd.DataFrame, normalized_dir: Path) -> dict[str, Any]:
    normalized_dir.mkdir(parents=True, exist_ok=True)
    pair_summaries: dict[str, Any] = {}
    for pair in TARGET_PAIRS:
        pair_rows = rows.loc[rows["pair"].eq(pair)].copy()
        path = normalized_dir / f"{pair}_M1_external_overlay.parquet"
        if pair_rows.empty:
            empty_external_frame().to_parquet(path, compression="zstd")
            pair_summaries[pair] = pair_summary(pair_rows, path)
            continue
        pair_rows = pair_rows.drop(columns=["pair"]).set_index("timestamp").sort_index()
        pair_rows.index.name = "timestamp"
        pair_rows.to_parquet(path, compression="zstd")
        pair_summaries[pair] = pair_summary(pair_rows, path)
    return {
        "pair_summaries": pair_summaries,
        "pairs_present": [pair for pair, summary in pair_summaries.items() if summary["row_count"] > 0],
        "pairs_with_post_preregistration_rows": [
            pair for pair, summary in pair_summaries.items() if summary["post_preregistration_row_count"] > 0
        ],
        "all_target_pairs_present": all(summary["row_count"] > 0 for summary in pair_summaries.values()),
        "all_target_pairs_have_post_preregistration_rows": all_target_pairs_have_post_preregistration(
            {"pair_summaries": pair_summaries}
        ),
    }


def empty_external_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(PRICE_COLUMNS), dtype="float64")
    frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return frame


def pair_summary(pair_rows: pd.DataFrame, path: Path) -> dict[str, Any]:
    if pair_rows.empty:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": 0,
            "first": None,
            "last": None,
            "post_preregistration_row_count": 0,
        }
    timestamps = pd.to_datetime(pair_rows.index if pair_rows.index.name == "timestamp" else pair_rows["timestamp"], utc=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": int(len(pair_rows)),
        "first": timestamps.min().isoformat(),
        "last": timestamps.max().isoformat(),
        "post_preregistration_row_count": int((timestamps > PREREGISTRATION_TIMESTAMP).sum()),
    }


def build_overlay_dir(*, canonical_m1_dir: Path, normalized_dir: Path, overlay_dir: Path) -> dict[str, Any]:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for old_path in overlay_dir.glob("*_M1_*.parquet"):
        old_path.unlink()

    linked_canonical: list[str] = []
    for path in sorted(canonical_m1_dir.glob("*_M1_*.parquet")):
        target = overlay_dir / path.name
        safe_link_or_copy(path, target)
        linked_canonical.append(str(target))

    linked_external: list[str] = []
    for path in sorted(normalized_dir.glob("*_M1_external_overlay.parquet")):
        target = overlay_dir / path.name
        safe_link_or_copy(path, target)
        linked_external.append(str(target))

    return {
        "overlay_dir": str(overlay_dir),
        "canonical_file_count": len(linked_canonical),
        "external_file_count": len(linked_external),
        "required_rate_pairs": list(MAJOR_RATE_PAIRS),
        "target_pairs": list(TARGET_PAIRS),
    }


def safe_link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def all_target_pairs_have_post_preregistration(normalized_summary: dict[str, Any] | None) -> bool:
    if normalized_summary is None:
        return False
    pair_summaries = normalized_summary["pair_summaries"]
    return all(pair_summaries[pair]["post_preregistration_row_count"] > 0 for pair in TARGET_PAIRS)


def fast167_status(
    *,
    external_m1_supplied: bool,
    normalized_summary: dict[str, Any] | None,
    readiness_report: dict[str, Any] | None,
    run_readiness: bool,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "external_m1_overlay_intake"]
    if not external_m1_supplied:
        return "FAST167_EXTERNAL_M1_INPUT_PENDING", reasons + ["external_m1_csv_not_supplied"]
    if normalized_summary is None:
        return "FAST167_EXTERNAL_M1_NORMALIZATION_MISSING", reasons + ["normalization_not_completed"]
    if not normalized_summary["all_target_pairs_present"]:
        return (
            "FAST167_EXTERNAL_M1_NORMALIZED_TARGET_PAIRS_INCOMPLETE",
            reasons + ["not_all_target_pairs_present_in_external_m1"],
        )
    if not normalized_summary["all_target_pairs_have_post_preregistration_rows"]:
        return (
            "FAST167_EXTERNAL_M1_NORMALIZED_POST_PREREGISTRATION_INCOMPLETE",
            reasons + ["not_all_target_pairs_have_post_preregistration_rows"],
        )
    if not run_readiness:
        return (
            "FAST167_EXTERNAL_M1_OVERLAY_READY_NOT_GATED",
            reasons + ["fast152_overlay_gate_skipped"],
        )
    if readiness_report is None:
        return "FAST167_EXTERNAL_M1_OVERLAY_READY_BUT_FAST152_NOT_RUN", reasons + ["fast152_overlay_gate_missing"]
    return (
        "FAST167_EXTERNAL_M1_OVERLAY_RAN_FAST152",
        reasons + [f"fast152_status:{readiness_report['status']}"],
    )


def next_action_for_status(status: str) -> str:
    if status == "FAST167_EXTERNAL_M1_INPUT_PENDING":
        return "fill_external_m1_normalized_template_with_broker_or_platform_export_then_rerun_fast167"
    if status.endswith("INCOMPLETE"):
        return "supply_all_fast151_target_pairs_with_rows_after_preregistration_timestamp"
    if status == "FAST167_EXTERNAL_M1_OVERLAY_READY_NOT_GATED":
        return "rerun_fast167_without_skip_fast152_or_run_fast152_against_overlay_m1_dir"
    if status == "FAST167_EXTERNAL_M1_OVERLAY_RAN_FAST152":
        return "review_fast152_overlay_status_before_any_shadow_or_testnet_decision"
    return "fix_external_m1_overlay_intake_before_forward_readiness_review"


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "external_m1_path": summary["external_m1_path"],
        "template_path": summary["template_path"],
        "overlay_dir": summary["overlay_dir"],
        "fast152_overlay_status": summary["fast152_overlay_status"],
        "next_action": summary["next_action"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
