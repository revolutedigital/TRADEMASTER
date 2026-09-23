"""Aggregate fixed-gate shadow windows without approving strategies."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.research.summarize_book_fixed_gate_shadow_windows import (
    SettlementReportRef,
    summarize_book_fixed_gate_shadow_windows,
)


def test_shadow_window_summary_weights_executable_net_and_stays_research_only(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_settlement_report(ind_net1=0.5, one_net1=1.0)), encoding="utf-8")
    second.write_text(json.dumps(_settlement_report(ind_net1=1.5, one_net1=2.0)), encoding="utf-8")

    report = summarize_book_fixed_gate_shadow_windows(
        settlement_report_refs=(
            SettlementReportRef("first", first),
            SettlementReportRef("second", second),
        ),
        output_root=tmp_path / "out",
        minimum_windows=2,
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
        cost_levels_bps=(0.0, 1.0, 2.0),
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert (
        report["status"]
        == "SHADOW_WINDOW_SUMMARY_REPLICATED_AT_PROFIT_COST_STRESS_COST_FRAGILE"
    )
    assert report["aggregate_by_mode"]["independent"]["net_mean_by_cost_bps"]["1"] == 1.0
    assert report["aggregate_by_mode"]["one_position"]["net_mean_by_cost_bps"]["1"] == 1.5


def test_shadow_window_summary_rejects_execution_authorization(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    payload = _settlement_report(ind_net1=0.5, one_net1=0.5)
    payload["execution_authorization"] = "testnet"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        summarize_book_fixed_gate_shadow_windows(
            settlement_report_refs=(SettlementReportRef("bad", path),),
            output_root=tmp_path / "out",
        )
    except ValueError as error:
        assert "execution_authorization=none" in str(error)
    else:
        raise AssertionError("expected execution boundary validation to fail")


def _settlement_report(*, ind_net1: float, one_net1: float) -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "status": "SHADOW_SETTLEMENT_READY_FOR_PROSPECTIVE_STREAM",
        "status_reasons": ["prospective_shadow_stream_required_before_testnet"],
        "shadow_selected_count": 3,
        "execution_replay": {
            "latency_ms": 100,
            "target_bps": 5.0,
            "initial_stop_bps": 20.0,
        },
        "independent_summary": _mode_summary(net1=ind_net1, executable_count=2),
        "one_position_summary": _mode_summary(net1=one_net1, executable_count=2),
    }


def _mode_summary(*, net1: float, executable_count: int) -> dict[str, object]:
    return {
        "selected_signal_count": executable_count,
        "executable_count": executable_count,
        "skipped_count": 0,
        "gross_executable_mean_bps": net1 + 1.0,
        "gross_executable_total_bps": (net1 + 1.0) * executable_count,
        "gross_win_rate": 1.0,
        "maximum_drawdown_bps": 1.0,
        "exit_reasons": {"TARGET": executable_count},
        "cost_sensitivity": [
            {
                "explicit_cost_bps": 0.0,
                "net_executable_mean_bps": net1 + 1.0,
                "net_executable_total_bps": (net1 + 1.0) * executable_count,
                "net_positive": True,
            },
            {
                "explicit_cost_bps": 1.0,
                "net_executable_mean_bps": net1,
                "net_executable_total_bps": net1 * executable_count,
                "net_positive": net1 > 0,
            },
            {
                "explicit_cost_bps": 2.0,
                "net_executable_mean_bps": net1 - 1.0,
                "net_executable_total_bps": (net1 - 1.0) * executable_count,
                "net_positive": net1 > 1.0,
            },
        ],
    }
