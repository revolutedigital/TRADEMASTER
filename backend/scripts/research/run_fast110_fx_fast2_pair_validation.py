"""Fast110 out-of-sample validation of Fast109 pair-specific Fast2 weak candidates.

Fast109 found post-hoc weak config/pair pockets inside the rejected Fast2 grid.
Fast110 freezes that weak list and validates only those config/pair identities
on Fast2's later S1/S2 samples. This is still research-only: even a survivor
would require a fresh preregistered prospective observation before execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research import fx_fast2_lab as fast2  # noqa: E402
from scripts.research.run_fast109_fx_fast2_pair_specific_audit import (  # noqa: E402
    plain_json,
    sha256_file,
    simulate_pair_config_accumulators,
    summarize_pair_config_accumulator,
    working_directory,
)


DEFAULT_CANDIDATES_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast109"
    / "fx-fast2-pair-specific-audit"
    / "weak_pair_config_candidates.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast110" / "fx-fast2-pair-validation"
)
DEFAULT_REPLICATION_MIN_TRADES = 100
DEFAULT_CONFIRMATION_MIN_TRADES = 30


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replication-min-trades", type=int, default=DEFAULT_REPLICATION_MIN_TRADES)
    parser.add_argument("--confirmation-min-trades", type=int, default=DEFAULT_CONFIRMATION_MIN_TRADES)
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()
    report = run_fast110_fx_fast2_pair_validation(
        candidates_path=arguments.candidates_path,
        output_root=arguments.output_root,
        replication_min_trades=arguments.replication_min_trades,
        confirmation_min_trades=arguments.confirmation_min_trades,
        progress=arguments.progress,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast110_fx_fast2_pair_validation(
    *,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    replication_min_trades: int = DEFAULT_REPLICATION_MIN_TRADES,
    confirmation_min_trades: int = DEFAULT_CONFIRMATION_MIN_TRADES,
    progress: bool = False,
) -> dict[str, Any]:
    validate_parameters(
        replication_min_trades=replication_min_trades,
        confirmation_min_trades=confirmation_min_trades,
    )
    output_root = output_root if output_root.is_absolute() else (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_path = (
        candidates_path if candidates_path.is_absolute() else (Path.cwd() / candidates_path).resolve()
    )
    candidates = load_candidate_identities(candidates_path)
    candidate_signatures = {candidate_signature(candidate) for candidate in candidates}

    replication = evaluate_stage_candidates(
        stage="replication",
        candidates=candidates,
        candidate_signatures=candidate_signatures,
        progress=progress,
    )
    confirmation = evaluate_stage_candidates(
        stage="confirmation",
        candidates=candidates,
        candidate_signatures=candidate_signatures,
        progress=progress,
    )
    validation_results = [
        validate_candidate(
            candidate,
            replication.get(candidate_signature(candidate)),
            confirmation.get(candidate_signature(candidate)),
            replication_min_trades=replication_min_trades,
            confirmation_min_trades=confirmation_min_trades,
        )
        for candidate in candidates
    ]
    strict_survivors = [row for row in validation_results if row["strict_validated"]]
    weak_survivors = [row for row in validation_results if row["weak_validated"]]
    replication_strict = [row for row in validation_results if row["replication"]["strict_pass"]]
    confirmation_strict = [row for row in validation_results if row["confirmation"]["strict_pass"]]

    validation_path = output_root / "validation_results.json"
    strict_path = output_root / "strict_validation_survivors.json"
    weak_path = output_root / "weak_validation_survivors.json"
    replication_path = output_root / "replication_candidate_evaluations.json"
    confirmation_path = output_root / "confirmation_candidate_evaluations.json"
    write_json(validation_path, validation_results)
    write_json(strict_path, strict_survivors)
    write_json(weak_path, weak_survivors)
    write_json(replication_path, sorted(replication.values(), key=lambda row: (row["config"], row["pair"])))
    write_json(confirmation_path, sorted(confirmation.values(), key=lambda row: (row["config"], row["pair"])))

    status, status_reasons = fast110_status(
        strict_survivor_count=len(strict_survivors),
        weak_survivor_count=len(weak_survivors),
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
            "fast110_validates_post_hoc_fast109_weak_candidates_only",
            "historical_out_of_sample_validation_is_not_execution_authorization",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast110_fx_fast2_pair_validation",
        "status": status,
        "status_reasons": status_reasons,
        "source_candidates_path": str(candidates_path),
        "source_candidates_sha256": sha256_file(candidates_path),
        "candidate_count": len(candidates),
        "unique_config_count": len({candidate["config"] for candidate in candidates}),
        "unique_pair_count": len({candidate["pair"] for candidate in candidates}),
        "replication_min_trades": replication_min_trades,
        "confirmation_min_trades": confirmation_min_trades,
        "replication_strict_pass_count": len(replication_strict),
        "confirmation_strict_pass_count": len(confirmation_strict),
        "weak_survivor_count": len(weak_survivors),
        "strict_survivor_count": len(strict_survivors),
        "validation_results_path": str(validation_path),
        "validation_results_sha256": sha256_file(validation_path),
        "strict_survivors_path": str(strict_path),
        "strict_survivors_sha256": sha256_file(strict_path),
        "weak_survivors_path": str(weak_path),
        "weak_survivors_sha256": sha256_file(weak_path),
        "replication_evaluations_path": str(replication_path),
        "replication_evaluations_sha256": sha256_file(replication_path),
        "confirmation_evaluations_path": str(confirmation_path),
        "confirmation_evaluations_sha256": sha256_file(confirmation_path),
        "top_validation_results": top_validation_results(validation_results),
    }
    return write_report(output_root, report)


def validate_parameters(*, replication_min_trades: int, confirmation_min_trades: int) -> None:
    if replication_min_trades <= 0:
        raise ValueError("replication_min_trades must be positive")
    if confirmation_min_trades <= 0:
        raise ValueError("confirmation_min_trades must be positive")


def load_candidate_identities(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    seen = set()
    for row in rows:
        candidate = {
            "config": str(row["config"]),
            "pair": str(row["pair"]),
            "family": str(row.get("family", row["config"].split(".", 1)[0])),
            "fast109_base_mean_r": row.get("base_mean_r"),
            "fast109_stress_mean_r": row.get("stress_mean_r"),
            "fast109_stress_t": row.get("stress_t"),
            "fast109_positive_stress_year_count": row.get("positive_stress_year_count"),
            "fast109_year_count": row.get("year_count"),
        }
        signature = candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: (candidate["config"], candidate["pair"]))


def candidate_signature(candidate: dict[str, Any]) -> str:
    return f"{candidate['config']}|{candidate['pair']}"


def evaluate_stage_candidates(
    *,
    stage: str,
    candidates: list[dict[str, Any]],
    candidate_signatures: set[str],
    progress: bool,
) -> dict[str, dict[str, Any]]:
    configs_by_key = fast2.configurations()
    needed_configs = {candidate["config"] for candidate in candidates}
    missing_configs = sorted(needed_configs.difference(configs_by_key))
    if missing_configs:
        raise ValueError(f"missing Fast2 configurations: {missing_configs}")
    configs = {key: configs_by_key[key] for key in sorted(needed_configs)}
    needed_pairs = {candidate["pair"] for candidate in candidates}

    with working_directory(BACKEND_ROOT):
        matrices, windows, _universe, rates = fast2.setup(stage)
        filtered_windows = {
            pair: pair_windows
            for pair, pair_windows in windows.items()
            if pair in needed_pairs
        }
        accumulators = simulate_pair_config_accumulators(
            matrices=matrices,
            windows=filtered_windows,
            rates=rates,
            configs=configs,
            progress=progress,
        )

    rows = {}
    for accumulator in accumulators.values():
        row = summarize_pair_config_accumulator(accumulator)
        signature = candidate_signature(row)
        if signature in candidate_signatures:
            rows[signature] = row | {"stage": stage}
    return rows


def validate_candidate(
    candidate: dict[str, Any],
    replication: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
    *,
    replication_min_trades: int,
    confirmation_min_trades: int,
) -> dict[str, Any]:
    replication_gate = stage_gate(replication, min_trades=replication_min_trades)
    confirmation_gate = stage_gate(confirmation, min_trades=confirmation_min_trades)
    weak_validated = replication_gate["weak_pass"] and confirmation_gate["weak_pass"]
    strict_validated = replication_gate["strict_pass"] and confirmation_gate["strict_pass"]
    return plain_json(
        {
            "config": candidate["config"],
            "pair": candidate["pair"],
            "family": candidate["family"],
            "fast109": candidate,
            "replication": replication_gate,
            "confirmation": confirmation_gate,
            "weak_validated": weak_validated,
            "strict_validated": strict_validated,
        }
    )


def stage_gate(row: dict[str, Any] | None, *, min_trades: int) -> dict[str, Any]:
    if row is None:
        return {
            "weak_pass": False,
            "strict_pass": False,
            "failed": ["candidate_not_generated_in_stage"],
            "summary": None,
        }
    failed = []
    if row["base_trade_count"] < min_trades:
        failed.append(f"base_trades_below_{min_trades}")
    if row["stress_trade_count"] < min_trades:
        failed.append(f"stress_trades_below_{min_trades}")
    if row["base_mean_r"] is None or row["base_mean_r"] <= 0:
        failed.append("base_mean_not_positive")
    if row["stress_mean_r"] is None or row["stress_mean_r"] <= 0:
        failed.append("stress_mean_not_positive")

    weak_pass = not failed
    strict_failed = list(failed)
    if row["year_count"] <= 0:
        strict_failed.append("no_years")
    if row["positive_base_year_count"] != row["year_count"]:
        strict_failed.append("base_not_positive_every_year")
    if row["positive_stress_year_count"] != row["year_count"]:
        strict_failed.append("stress_not_positive_every_year")
    strict_pass = not strict_failed
    return {
        "weak_pass": weak_pass,
        "strict_pass": strict_pass,
        "failed": strict_failed if strict_failed else [],
        "summary": compact_stage_row(row),
    }


def compact_stage_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": row.get("stage"),
        "base_trade_count": row["base_trade_count"],
        "stress_trade_count": row["stress_trade_count"],
        "base_mean_r": row["base_mean_r"],
        "stress_mean_r": row["stress_mean_r"],
        "base_t": row["base_t"],
        "stress_t": row["stress_t"],
        "year_count": row["year_count"],
        "positive_base_year_count": row["positive_base_year_count"],
        "positive_stress_year_count": row["positive_stress_year_count"],
        "min_year_stress_mean_r": row["min_year_stress_mean_r"],
    }


def top_validation_results(rows: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row["strict_validated"],
            row["weak_validated"],
            gate_stress_mean(row["confirmation"]),
            gate_stress_mean(row["replication"]),
        ),
        reverse=True,
    )[:limit]


def gate_stress_mean(gate: dict[str, Any]) -> float:
    summary = gate.get("summary")
    if not summary or summary["stress_mean_r"] is None:
        return float("-inf")
    return float(summary["stress_mean_r"])


def fast110_status(
    *,
    strict_survivor_count: int,
    weak_survivor_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if strict_survivor_count:
        return (
            "FAST110_FX_FAST2_PAIR_VALIDATION_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"strict_validation_survivors:{strict_survivor_count}",
                "fresh_preregistered_prospective_observation_required",
            ],
        )
    if weak_survivor_count:
        return (
            "FAST110_FX_FAST2_PAIR_VALIDATION_WEAK_ONLY_FAILED_STABILITY",
            reasons
            + [
                f"weak_validation_survivors:{weak_survivor_count}",
                "out_of_sample_positive_but_failed_strict_year_stability",
            ],
        )
    return (
        "FAST110_FX_FAST2_PAIR_VALIDATION_FAILED_OUT_OF_SAMPLE",
        reasons + ["no_fast109_weak_candidate_survived_replication_and_confirmation"],
    )


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate_count": report["candidate_count"],
        "replication_strict_pass_count": report["replication_strict_pass_count"],
        "confirmation_strict_pass_count": report["confirmation_strict_pass_count"],
        "weak_survivor_count": report["weak_survivor_count"],
        "strict_survivor_count": report["strict_survivor_count"],
        "top_validation_results": report["top_validation_results"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
