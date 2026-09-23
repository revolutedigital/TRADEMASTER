"""Second-filter scout is diagnostic and fail-closed."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from scripts.research.run_book_fixed_gate_second_filter_scout import (
    Candidate,
    SettlementReportRef,
    run_book_fixed_gate_second_filter_scout,
)


BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def test_second_filter_scout_finds_candidate_without_approval(tmp_path: Path) -> None:
    first = _write_window(tmp_path, "first", feature_values=(1.0, 2.0), payoffs=(4.0, -4.0))
    second = _write_window(tmp_path, "second", feature_values=(1.0, 2.0), payoffs=(3.0, -3.0))

    report = run_book_fixed_gate_second_filter_scout(
        settlement_report_refs=(
            SettlementReportRef("first", first),
            SettlementReportRef("second", second),
        ),
        output_root=tmp_path / "out",
        features=("test_feature",),
        quantiles=(0.5,),
        minimum_one_position_count_per_window=1,
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
        cost_levels_bps=(0.0, 1.0, 2.0),
        top_n=3,
        fixed_candidate=Candidate("test_feature", "low", 1.5),
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["candidate_count"] >= 1
    assert report["top_candidates"][0]["candidate"]["feature"] == "test_feature"
    assert report["fixed_candidate_result"]["one_position_positive_window_count_at_profit_cost"] == 2


def test_second_filter_scout_rejects_authorized_report(tmp_path: Path) -> None:
    report_path = _write_window(tmp_path, "bad", feature_values=(1.0,), payoffs=(3.0,))
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["execution_authorization"] = "testnet"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        run_book_fixed_gate_second_filter_scout(
            settlement_report_refs=(SettlementReportRef("bad", report_path),),
            output_root=tmp_path / "out",
        )
    except ValueError as error:
        assert "execution_authorization=none" in str(error)
    else:
        raise AssertionError("expected execution boundary validation to fail")


def _write_window(
    root: Path,
    label: str,
    *,
    feature_values: tuple[float, ...],
    payoffs: tuple[float, ...],
) -> Path:
    window_root = root / label
    decisions_path = window_root / "shadow_decisions.jsonl"
    settlements_path = window_root / "shadow_settlements.parquet"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decision_rows = []
    settlement_rows = []
    for index, (feature_value, payoff) in enumerate(zip(feature_values, payoffs, strict=True)):
        decision_time = BASE_TIME + timedelta(seconds=index * 130)
        decision_time_ms = int(decision_time.timestamp() * 1000)
        feature_hash = f"{index + 1:064x}"
        decision_rows.append(
            {
                "decision_time_ms": decision_time_ms,
                "feature_vector_sha256": feature_hash,
                "feature_vector": {
                    "book_available": 1.0,
                    "spot_available": 1.0,
                    "test_feature": feature_value,
                },
            }
        )
        settlement_rows.append(
            {
                "would_enter": True,
                "skip_reason": None,
                "decision_time_ms": decision_time_ms,
                "entry_time_ms": decision_time_ms,
                "entry_book_time_ms": decision_time_ms,
                "entry_book_age_ms": 0,
                "exit_time_ms": decision_time_ms + 1_000,
                "side": "SELL",
                "horizon_seconds": 1,
                "entry_price": 100.0,
                "exit_price": 99.0,
                "exit_reason": "TARGET" if payoff > 0 else "TIMEOUT",
                "gross_executable_bps": payoff,
                "maximum_favorable_executable_bps": max(payoff, 0.0),
                "maximum_adverse_executable_bps": min(payoff, 0.0),
                "policy_sha256": "a" * 64,
                "feature_vector_sha256": feature_hash,
                "shadow_decision_time": decision_time.isoformat(),
            }
        )
    with decisions_path.open("w", encoding="utf-8") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    pd.DataFrame(settlement_rows).to_parquet(settlements_path, index=False)
    report_path = window_root / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
                "committed": False,
                "dry_run": True,
                "approved_strategy": False,
                "execution_replay": {"latency_ms": 0},
                "shadow_decisions_path": str(decisions_path),
                "settlements_path": str(settlements_path),
                "status": "SHADOW_SETTLEMENT_READY_FOR_PROSPECTIVE_STREAM",
                "status_reasons": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report_path
