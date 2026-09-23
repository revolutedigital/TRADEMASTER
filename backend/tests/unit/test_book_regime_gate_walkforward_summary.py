"""Walk-forward matrix summaries for regime-gate candidates."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.research.summarize_book_regime_gate_walkforward import (
    summarize_walkforward_reports,
)


def test_walkforward_summary_selects_candidate_positive_in_every_fold(
    tmp_path: Path,
) -> None:
    report_paths = (
        _write_report(tmp_path / "fold0.json", _candidates(good_gross=1.2, weak_gross=3.0)),
        _write_report(tmp_path / "fold1.json", _candidates(good_gross=1.5, weak_gross=-0.2)),
        _write_report(tmp_path / "fold2.json", _candidates(good_gross=1.8, weak_gross=0.3)),
    )

    report = summarize_walkforward_reports(
        report_paths=report_paths,
        output_root=tmp_path / "out",
        profit_cost_bps=1.0,
        minimum_confirmation_selected_count=10,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["approved_strategy"] is False
    assert report["status"] == "ROBUST_GATE_CANDIDATE_FOR_FUTURE_HOLDOUT"
    selected = report["selected_robust_candidate"]
    assert selected["feature"] == "spot_update_age_ms"
    assert selected["direction"] == "low"
    assert selected["fraction"] == 0.33
    assert selected["positive_profit_cost_fold_count"] == 3


def test_walkforward_summary_fails_closed_with_partial_candidate_export(
    tmp_path: Path,
) -> None:
    source = {
        "status": "READY_FOR_REGIME_GATE_REVIEW",
        "candidates_evaluated": 2,
        "top_candidates_by_gate_development": [_candidate("spot_update_age_ms", 1.5)],
    }
    report_path = _write_report(tmp_path / "fold0.json", source)

    report = summarize_walkforward_reports(
        report_paths=(report_path,),
        output_root=tmp_path / "out",
    )

    assert report["status"] == "INCOMPLETE_CANDIDATE_EXPORT"
    assert report["modeling_ready"] is False
    assert report["selected_robust_candidate"] is None


def _candidates(*, good_gross: float, weak_gross: float) -> dict[str, object]:
    return {
        "status": "READY_FOR_REGIME_GATE_REVIEW",
        "candidates_evaluated": 2,
        "top_candidates_by_gate_development": [
            _candidate("fragile_top_feature", weak_gross, rank_payoff=3.0),
            _candidate("spot_update_age_ms", good_gross, rank_payoff=2.0),
        ],
    }


def _candidate(
    feature: str,
    confirmation_gross_bps: float,
    *,
    rank_payoff: float = 1.0,
) -> dict[str, object]:
    return {
        "gate": {
            "feature": feature,
            "direction": "low",
            "fraction": 0.33,
            "quantile": 0.33,
            "threshold": 123.0,
        },
        "gate_development_metrics": {
            "selected_count": 20,
            "selected_gross_path_payoff_mean_bps": rank_payoff,
        },
        "confirmation_metrics": {
            "selected_count": 12,
            "selected_gross_path_payoff_mean_bps": confirmation_gross_bps,
            "cost_sensitivity": [
                {
                    "total_cost_bps": 0.5,
                    "selected_net_mean_bps": confirmation_gross_bps - 0.5,
                },
                {
                    "total_cost_bps": 1.0,
                    "selected_net_mean_bps": confirmation_gross_bps - 1.0,
                },
            ],
        },
    }


def _write_report(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
